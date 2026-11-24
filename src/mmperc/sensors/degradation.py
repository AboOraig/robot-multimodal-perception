"""Controlled, reproducible sensor degradations, applied at load time to clean data.

Contract
--------
* Stateless and order independent: the corrupted values of an episode depend only on
  (seed, episode id, family, severity levels, the episode's clean arrays). Any subset or ordering
  of episodes gives identical results, so every model is evaluated under identical conditions.
  Training augmentation uses a different `seed` per epoch; evaluation uses a fixed seed.
* Severity is an integer level 0..4 per *family*; level 0 is the exact identity.
* A `Scenario` is a mapping family -> level (missing = 0). Combinations are simultaneous.
* Missing data (dropout) is zero-filled and flagged in `proprio_mask` / `contact_mask`
  (True = available). Models must receive the masks.
* Ground truth (cube pose, target, labels) is never modified.

Families (see configs/degradation.yaml for the calibrated level values)
-----------------------------------------------------------------------
vision : occlusion (targeted at the cube), occlusion_random, noise, blur, illumination
proprio: noise, bias (constant + slow random-walk drift), dropout (burst / Markov)
contact: noise, bias (tare offset + drift), dropout

Image pipeline order (physical): occluder -> illumination -> blur -> sensor noise -> clip to uint8.
Targeted occlusion uses the ground-truth cube position only to *place the corruption*; the models
never see it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter

from mmperc.common.seeding import make_rng
from mmperc.data.dataset import holding_gate
from mmperc.simulation.camera import intrinsics
from mmperc.simulation.scene import build_model

FAMILIES = ("vision_occlusion", "vision_occlusion_random", "vision_noise", "vision_blur",
            "vision_illumination", "proprio_noise", "proprio_bias", "proprio_dropout",
            "contact_noise", "contact_bias", "contact_dropout",
            "vision_dropout")                       # appended last: RNG stream ids of the older families must not change
MODALITY_OF = {f: f.split("_")[0] for f in FAMILIES}
_FAMILY_ID = {f: 100 + i for i, f in enumerate(FAMILIES)}       # fixed RNG stream ids (never reorder)
MODALITIES = ("vision", "proprio", "contact")


@dataclass(frozen=True)
class Scenario:
    """Severity level (0..4) per degradation family; absent families are at level 0."""
    levels: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @staticmethod
    def make(**levels: int) -> "Scenario":
        for k, v in levels.items():
            if k not in FAMILIES:
                raise KeyError(f"unknown degradation family {k!r}")
            if not 0 <= int(v) <= 4:
                raise ValueError(f"level of {k} must be in 0..4")
        return Scenario(tuple(sorted((k, int(v)) for k, v in levels.items() if int(v) > 0)))

    def level(self, family: str) -> int:
        return dict(self.levels).get(family, 0)

    @property
    def is_clean(self) -> bool:
        return len(self.levels) == 0

    @property
    def name(self) -> str:
        return "clean" if self.is_clean else "+".join(f"{k}:{v}" for k, v in self.levels)

    def modalities(self) -> set[str]:
        return {MODALITY_OF[k] for k, _ in self.levels}


def sample_scenario(rng: np.random.Generator, families: tuple[str, ...], p_clean: float = 0.15,
                    p_family: float = 0.3, max_level: int = 4) -> Scenario:
    """Random mixed-degradation scenario for training (each family independently active)."""
    if rng.random() < p_clean:
        return Scenario()
    active = {f: int(rng.integers(1, max_level + 1)) for f in families if rng.random() < p_family}
    if not active:
        active = {families[int(rng.integers(len(families)))]: int(rng.integers(1, max_level + 1))}
    return Scenario.make(**active)


def markov_mask(rng: np.random.Generator, T: int, p_drop: float, mean_burst: float) -> np.ndarray:
    """Availability mask (True = available) from a two-state Markov chain.

    Stationary dropout probability = p_drop; mean dropout burst length = mean_burst frames.
    """
    if p_drop <= 0.0:
        return np.ones(T, dtype=bool)
    p_exit = 1.0 / mean_burst
    p_enter = p_drop * p_exit / (1.0 - p_drop)
    u = rng.random(T)
    dropped = rng.random() < p_drop
    mask = np.empty(T, dtype=bool)
    for t in range(T):
        dropped = (u[t] >= p_exit) if dropped else (u[t] < p_enter)
        mask[t] = not dropped
    return mask


class CubeProjector:
    """Pixel position / apparent width of the cube from the (fixed) camera, in pure numpy."""

    def __init__(self, sim_cfg: dict[str, Any]):
        model = build_model(sim_cfg)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cid = model.camera(sim_cfg["camera"]["name"]).id
        self.pos = data.cam_xpos[cid].copy()
        self.R = data.cam_xmat[cid].reshape(3, 3).copy()
        size = int(sim_cfg["image_size"])
        self.f, _, self.cx, self.cy = intrinsics(float(sim_cfg["camera"]["fovy"]), size, size)
        self.cube = float(sim_cfg["cube"]["size"])
        self.size = size

    def project(self, p_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(u, v, apparent_width_px) for world points (N, 3); u right, v down, pixel units."""
        pc = (np.atleast_2d(p_world) - self.pos) @ self.R
        depth = -pc[:, 2]
        return (self.cx + self.f * pc[:, 0] / depth, self.cy - self.f * pc[:, 1] / depth,
                self.f * self.cube / depth)


class Degrader:
    def __init__(self, deg_cfg: dict[str, Any], sim_cfg: dict[str, Any], dt: float | None = None):
        self.cfg = deg_cfg
        self.levels = {f: list(deg_cfg["families"][f]["levels"]) for f in FAMILIES}
        for f, v in self.levels.items():
            if len(v) != 5:
                raise ValueError(f"{f}: need 5 level values (level 0..4), got {len(v)}")
        self.dt = dt if dt is not None else 1.0 / float(sim_cfg["log_hz"])
        self.burst = float(deg_cfg["dropout_mean_burst_frames"])
        self.proj = CubeProjector(sim_cfg)
        self.cube_size = float(sim_cfg["cube"]["size"])

    # ---------------------------------------------------------------- helpers
    def _val(self, family: str, level: int) -> float:
        return float(self.levels[family][level])

    def _rng(self, seed: int, ep_id: int, family: str) -> np.random.Generator:
        return make_rng(int(seed), int(ep_id), _FAMILY_ID[family])

    def _drift(self, rng, T: int, sigma: float, shape: tuple[int, ...]) -> np.ndarray:
        """Constant offset + slow random walk (std per second = drift_frac * sigma)."""
        b0 = rng.normal(0.0, sigma, shape)
        step = self.cfg["bias_drift_frac"] * sigma * np.sqrt(self.dt)
        walk = np.cumsum(rng.normal(0.0, step, (T,) + shape), axis=0)
        return b0[None] + walk

    # ---------------------------------------------------------------- one episode
    def apply_episode(self, ep: dict[str, np.ndarray], scenario: Scenario, seed: int,
                      ep_id: int) -> dict[str, np.ndarray]:
        """Degrade one episode. `ep` needs: q, qd, finger_q, wrench, cube_pos, and image (if vision
        families are active). Returns new arrays for every sensor channel + the two masks."""
        T = len(ep["q"])
        out: dict[str, np.ndarray] = {
            "q": ep["q"].copy(), "qd": ep["qd"].copy(), "finger_q": ep["finger_q"].copy(),
            "wrench": ep["wrench"].copy(),
            "proprio_mask": np.ones(T, dtype=bool), "contact_mask": np.ones(T, dtype=bool),
            "vision_mask": np.ones(T, dtype=bool)}
        if "image" in ep:
            out["image"] = ep["image"].copy()
        if scenario.is_clean:
            return out
        L = scenario.level
        S = self._val

        # ------------------------------------------------ vision
        if "image" in ep and "vision" in scenario.modalities():
            img = ep["image"].astype(np.float32)
            Hh, Ww = img.shape[1:3]
            if L("vision_occlusion"):
                c = S("vision_occlusion", L("vision_occlusion"))
                rng = self._rng(seed, ep_id, "vision_occlusion")
                u, v, s_px = self.proj.project(ep["cube_pos"])
                jit = rng.uniform(-1, 1, (T, 2)) * self.cfg["occlusion_jitter_frac"] * s_px[:, None]
                # DECOYS: identical grey squares rigidly offset from the cube path by a random per-episode displacement
                # (same height as the cube -> same apparent size). Without them the occluder itself marks the cube position
                # and image models "see through" the occlusion (found with the cube-invisible control, M6b). With n decoys a
                # single frame shows n+1 indistinguishable squares, so the cover leaks at most ~log(n+1) nats.
                n_dec = int(self.cfg.get("occlusion_decoys", 0))
                dec_uv = []
                if n_dec:
                    size_w = self.cube_size
                    ang = rng.uniform(0.0, 2.0 * np.pi, n_dec)
                    rad = max(1.5 * c * size_w, 0.04) + rng.uniform(0.0, 0.06, n_dec)      # clear of the cover square
                    for k in range(n_dec):
                        pk = ep["cube_pos"] + np.array([rad[k] * np.cos(ang[k]), rad[k] * np.sin(ang[k]), 0.0])
                        uk, vk, _ = self.proj.project(pk)
                        dec_uv.append((uk, vk))
                for t in range(T):
                    half = 0.5 * c * s_px[t]
                    centres = [(u[t] + jit[t, 0], v[t] + jit[t, 1])] + [(uk[t], vk[t]) for uk, vk in dec_uv]
                    for cu, cv in centres:
                        x0, x1 = int(np.floor(cu - half)), int(np.ceil(cu + half))
                        y0, y1 = int(np.floor(cv - half)), int(np.ceil(cv + half))
                        img[t, max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = self.cfg["occluder_gray"]
            if L("vision_occlusion_random"):
                a = S("vision_occlusion_random", L("vision_occlusion_random"))
                rng = self._rng(seed, ep_id, "vision_occlusion_random")
                aspect = np.exp(rng.uniform(np.log(0.5), np.log(2.0), T))
                area = a * Hh * Ww
                w = np.clip(np.sqrt(area * aspect), 1, Ww)
                h = np.clip(area / w, 1, Hh)
                x0 = rng.uniform(0, 1, T) * (Ww - w)
                y0 = rng.uniform(0, 1, T) * (Hh - h)
                for t in range(T):
                    img[t, int(y0[t]):int(round(y0[t] + h[t])), int(x0[t]):int(round(x0[t] + w[t]))] = \
                        self.cfg["occluder_gray"]
            if L("vision_illumination"):
                a = S("vision_illumination", L("vision_illumination"))
                rng = self._rng(seed, ep_id, "vision_illumination")
                gain = np.exp(rng.uniform(-np.log1p(a), np.log1p(a)))             # one lighting per episode
                gamma = np.exp(rng.uniform(-np.log1p(a / 2), np.log1p(a / 2)))
                img = 255.0 * np.clip(img / 255.0 * gain, 0.0, 1.0) ** gamma
            if L("vision_blur"):
                s = S("vision_blur", L("vision_blur"))
                img = gaussian_filter(img, sigma=(0, s, s, 0), mode="nearest")
            if L("vision_noise"):
                s = S("vision_noise", L("vision_noise"))
                img = img + self._rng(seed, ep_id, "vision_noise").normal(0.0, s, img.shape).astype(np.float32)
            out["image"] = np.clip(np.rint(img), 0, 255).astype(np.uint8)
            if L("vision_dropout"):                 # lost camera frames (bursty): blank black frame + mask, like the other dropouts
                rng = self._rng(seed, ep_id, "vision_dropout")
                out["vision_mask"] = markov_mask(rng, T, S("vision_dropout", L("vision_dropout")), self.burst)
                out["image"][~out["vision_mask"]] = 0

        # ------------------------------------------------ proprioception
        if L("proprio_noise"):
            k = L("proprio_noise")
            rng = self._rng(seed, ep_id, "proprio_noise")
            out["q"] = out["q"] + rng.normal(0.0, S("proprio_noise", k), out["q"].shape)
            out["finger_q"] = out["finger_q"] + rng.normal(0.0, self.cfg["proprio_finger_noise_per_rad"]
                                                           * S("proprio_noise", k), out["finger_q"].shape)
            out["qd"] = out["qd"] + rng.normal(0.0, self.cfg["proprio_qd_noise_per_rad"]
                                               * S("proprio_noise", k), out["qd"].shape)
        if L("proprio_bias"):
            sb = S("proprio_bias", L("proprio_bias"))
            rng = self._rng(seed, ep_id, "proprio_bias")
            out["q"] = out["q"] + self._drift(rng, T, sb, (7,))
            out["finger_q"] = out["finger_q"] + self._drift(rng, T, self.cfg["proprio_finger_noise_per_rad"] * sb, (2,))
        if L("proprio_dropout"):
            rng = self._rng(seed, ep_id, "proprio_dropout")
            out["proprio_mask"] = markov_mask(rng, T, S("proprio_dropout", L("proprio_dropout")), self.burst)

        # ------------------------------------------------ contact
        if L("contact_noise"):
            sf = S("contact_noise", L("contact_noise"))
            sig = np.array([sf] * 3 + [sf * self.cfg["contact_torque_arm_m"]] * 3)
            out["wrench"] = out["wrench"] + self._rng(seed, ep_id, "contact_noise").normal(0.0, 1.0, out["wrench"].shape) * sig
        if L("contact_bias"):
            sf = S("contact_bias", L("contact_bias"))
            rng = self._rng(seed, ep_id, "contact_bias")
            bf = self._drift(rng, T, sf, (3,))
            bt = self._drift(rng, T, sf * self.cfg["contact_torque_arm_m"], (3,))
            out["wrench"] = out["wrench"] + np.concatenate([bf, bt], axis=1)
        if L("contact_dropout"):
            rng = self._rng(seed, ep_id, "contact_dropout")
            out["contact_mask"] = markov_mask(rng, T, S("contact_dropout", L("contact_dropout")), self.burst)

        # zero-fill missing channels (applied last so that biases/noise never leak through a dropout)
        pm, cm = ~out["proprio_mask"], ~out["contact_mask"]
        for k in ("q", "qd", "finger_q"):
            out[k][pm] = 0.0
        out["wrench"][cm] = 0.0
        return out

    # ---------------------------------------------------------------- whole split
    def apply(self, d: dict[str, np.ndarray],
              scenario: Scenario | Callable[[int], Scenario], seed: int) -> dict[str, np.ndarray]:
        """Degrade a loaded split (flat frame arrays from `load_split`). Returns a new dict that
        shares the untouched arrays (ground truth, labels) with `d` and replaces the sensor arrays.

        `scenario` is a Scenario (same for all episodes) or a callable ep_id -> Scenario."""
        ep_ids = d["ep_id"]
        bounds = np.concatenate([[0], np.flatnonzero(np.diff(ep_ids)) + 1, [len(ep_ids)]])
        keys = ["q", "qd", "finger_q", "wrench"] + (["image"] if "image" in d else [])
        res = {k: np.empty_like(d[k]) for k in keys}
        res["proprio_mask"] = np.ones(len(ep_ids), dtype=bool)
        res["contact_mask"] = np.ones(len(ep_ids), dtype=bool)
        res["vision_mask"] = np.ones(len(ep_ids), dtype=bool)
        for a, b in zip(bounds[:-1], bounds[1:]):
            eid = int(ep_ids[a])
            sc = scenario(eid) if callable(scenario) else scenario
            sl = slice(a, b)
            piece = {k: d[k][sl] for k in keys + ["cube_pos"]}
            o = self.apply_episode(piece, sc, seed, eid)
            for k in res:
                res[k][sl] = o[k]
        # the observable holding detector must be recomputed from the *corrupted* finger readings
        res["holding_gate"] = holding_gate(res["finger_q"]) & res["proprio_mask"]
        out = dict(d)
        out.update(res)
        return out

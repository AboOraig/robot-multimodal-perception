"""Turn a (possibly degraded) data dict into model inputs, and fit normalisation statistics on TRAIN only."""
from __future__ import annotations

import numpy as np

from mmperc.baselines.kinematics import HandKinematics, hand_yaw


def proprio_features(d: dict, hk: HandKinematics | None, use_fk: bool, use_gate: bool, gate_inputs: bool = False):
    """(feat (N,F) float32, ok (N,) bool). Columns: q, qd, finger_q(2)+opening, [FK: tcp pos, cos4g, sin4g], [holding gate]."""
    n = len(d["q"])
    ok = np.asarray(d["proprio_mask"], bool) if "proprio_mask" in d else np.ones(n, bool)
    cols = [d["q"], d["qd"], d["finger_q"], d["finger_q"].sum(1, keepdims=True)]
    if use_fk:
        P, R = hk.fk(d["q"])
        g = hand_yaw(R)
        cols += [P, np.cos(4 * g)[:, None], np.sin(4 * g)[:, None]]
    if use_gate:
        cols.append(np.asarray(d["holding_gate"], float)[:, None])
    if gate_inputs:                                        # learned analogue of B3: proprioception is 'missing' unless holding
        ok = ok & np.asarray(d["holding_gate"], bool)
    return np.concatenate(cols, 1).astype(np.float32), ok


def contact_features(d: dict, hk: HandKinematics):
    """(cfeat (N,13) float32, ok (N,) bool). Conditioned on proprioception: the wrench is only interpretable with the
    arm pose, so the contact stream is 'missing' if EITHER sensor is. Columns: wrench(6), gravity in hand frame(3),
    TCP height z(1), finger opening(1), |qd|(1), holding gate(1).
    The TCP x,y are deliberately NOT included: the scripted hand hovers over the cube, so they predict the cube xy with
    R^2 = 0.998 outside the grasp (found in M7; contact must carry contact information only)."""
    n = len(d["q"])
    pm = np.asarray(d["proprio_mask"], bool) if "proprio_mask" in d else np.ones(n, bool)
    cm = np.asarray(d["contact_mask"], bool) if "contact_mask" in d else np.ones(n, bool)
    P, R = hk.fk(d["q"])
    g_hand = -R[:, 2, :]                                   # R^T [0, 0, -1]
    cols = [d["wrench"], g_hand, P[:, 2:3], d["finger_q"].sum(1, keepdims=True), np.linalg.norm(d["qd"], axis=1, keepdims=True),
            np.asarray(d["holding_gate"], float)[:, None]]
    return np.concatenate(cols, 1).astype(np.float32), pm & cm


def ablate_inputs(inp: dict, which) -> dict:
    """Test-time modality ablation for fusion inputs: the modality is replaced by what a dead sensor delivers."""
    out = dict(inp)
    for w in which:
        if w == "vision":
            out["image"] = np.zeros_like(inp["image"]); out["vmask"] = np.zeros_like(inp["vmask"])
        elif w == "proprio":
            out["feat"] = np.zeros_like(inp["feat"]); out["ok"] = np.zeros_like(inp["ok"])
        elif w == "contact":
            out["cfeat"] = np.zeros_like(inp["cfeat"]); out["cok"] = np.zeros_like(inp["cok"])
        else:
            raise ValueError(f"unknown modality {w!r}")
    return out


def make_inputs(kind: str, d: dict, cfg: dict, hk: HandKinematics | None = None) -> dict:
    if kind == "fusion":
        p = cfg["proprio"]
        feat, ok = proprio_features(d, hk, p["use_fk"], p["use_gate"], gate_inputs=True)     # proprioception valid only while holding
        cfeat, cok = contact_features(d, hk)
        if not cfg.get("fusion", {}).get("use_contact", True):      # ablation: the contact expert never has data (always 'missing')
            cok = np.zeros_like(cok)
        n = len(feat)
        feat[~ok] = 0.0; cfeat[~cok] = 0.0                  # what the sensors deliver: missing -> zero-filled
        vm = np.asarray(d["vision_mask"], bool) if "vision_mask" in d else np.ones(n, bool)
        return {"image": d["image"], "vmask": vm, "feat": feat, "ok": ok, "cfeat": cfeat, "cok": cok}
    if kind == "vision":
        return {"image": d["image"]}
    if kind == "proprio":
        p = cfg["proprio"]
        feat, ok = proprio_features(d, hk, p["use_fk"], p["use_gate"], p.get("gate_inputs", False))
        return {"feat": feat, "ok": ok}
    raise ValueError(kind)


def fit_stats(train: dict, inputs: dict) -> dict:
    t = train["target"].astype(np.float64)
    stats = {"pos_mean": t[:, :3].mean(0).tolist(), "pos_std": t[:, :3].std(0).tolist()}
    for pre, okk in (("feat", "ok"), ("cfeat", "cok")):
        if pre in inputs:
            f, ok = inputs[pre], inputs[okk]
            if not ok.any():                                # stream never available (e.g. contact ablation): neutral statistics, no NaN
                stats[pre + "_mu"] = [0.0] * f.shape[1]; stats[pre + "_sd"] = [1.0] * f.shape[1]
                continue
            stats[pre + "_mu"] = f[ok].mean(0).tolist()
            sd = f[ok].std(0)
            # floor: near-constant features (e.g. the holding gate on valid rows) must not make zero-filled rows explode
            stats[pre + "_sd"] = np.maximum(sd, np.maximum(0.1 * np.abs(f[ok].mean(0)), 1e-3)).tolist()
    return stats

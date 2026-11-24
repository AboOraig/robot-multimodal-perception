"""Scripted pick -> lift -> hold -> place -> release -> retreat episode with synchronized logging.

Design notes (see docs/design.md):
* The motion is a joint-space trajectory obtained by IK along Cartesian segments. The robot is
  driven by the Menagerie position actuators plus a gravity feed-forward on the arm joints.
* Anti-leakage: random start posture, random intermediate waypoint, random null-space posture,
  random in-hand grasp offset and gripper yaw misalignment. Time / phase are logged for analysis
  only and must never be model inputs.
* Logging at `log_hz`: frame k is the state after k * (substeps) physics steps. The wrench is the
  mean over the physics substeps since the previous frame (no aliasing); every other channel is
  the instantaneous value at the frame time; the image is rendered from that same state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from mmperc.common.seeding import make_rng
from mmperc.simulation.kinematics import ArmIK, topdown_rotation, wrap_pi
from mmperc.simulation.scene import (ARM_JOINTS, CUBE_BODY, CUBE_GEOM, CUBE_JOINT, FINGER_JOINTS,
                                     TCP_SITE, build_model)

PHASES = ("approach", "close", "lift", "hold", "lower", "release", "retreat")
SEGMENT_PHASE = {"to_waypoint": 0, "to_pregrasp": 0, "descend": 0, "settle_pre": 0,
                 "close": 1, "grasp_settle": 1, "lift": 2, "hold": 3, "lower": 4,
                 "release": 5, "retreat": 6}


class PlanningError(RuntimeError):
    pass


def min_jerk(s: np.ndarray | float) -> np.ndarray | float:
    """Minimum-jerk time scaling, s in [0, 1]."""
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


@dataclass
class Segment:
    name: str
    duration: float
    q_samples: np.ndarray        # (n, 7) joint-space samples, n >= 1
    grip_start: float
    grip_end: float

    def q_at(self, t: float) -> np.ndarray:
        s = float(min_jerk(min(max(t / self.duration, 0.0), 1.0)))
        n = len(self.q_samples)
        if n == 1:
            return self.q_samples[0]
        x = s * (n - 1)
        i = min(int(x), n - 2)
        a = x - i
        return (1 - a) * self.q_samples[i] + a * self.q_samples[i + 1]

    def grip_at(self, t: float) -> float:
        a = min(max(t / self.duration, 0.0), 1.0)
        return (1 - a) * self.grip_start + a * self.grip_end


def _cartesian_segment(ik: ArmIK, q0, p0, g0, p1, g1, q_ref, step) -> np.ndarray:
    dist = max(np.linalg.norm(np.asarray(p1) - np.asarray(p0)), 1e-9)
    n = max(2, int(math.ceil(dist / step)) + 1)
    qs = [np.asarray(q0, dtype=float)]
    q = qs[0]
    for i in range(1, n):
        a = i / (n - 1)
        p = (1 - a) * np.asarray(p0) + a * np.asarray(p1)
        g = (1 - a) * g0 + a * g1
        q, ok, ep, er = ik.solve(q, p, topdown_rotation(g), q_ref=q_ref)
        if not ok:
            raise PlanningError(f"IK failed (pos err {ep:.4f} m, rot err {er:.4f} rad)")
        qs.append(q)
    return np.array(qs)


def plan_episode(cfg: dict[str, Any], ik: ArmIK, rng: np.random.Generator, q_start: np.ndarray,
                 cube_pos: np.ndarray, cube_yaw: float) -> tuple[list[Segment], dict[str, Any]]:
    """Sample the free parameters of the script and build the joint-space segments."""
    ec = cfg["episode"]
    q_home = np.array(cfg["robot"]["home_qpos"], dtype=float)
    q_ref = np.clip(q_home + rng.uniform(-1, 1, 7) * ec["null_posture_noise_rad"], ik.lo, ik.hi)

    # Gripper closing axis must align with a cube face normal: gamma = yaw + k * 90 deg (+ noise).
    # Reduce the cube yaw modulo 90 deg (4-fold symmetry) to the face-aligned angle in [-45, 45) deg
    # closest to the home orientation, then add the misalignment noise.
    gamma_g = float(np.mod(cube_yaw + np.pi / 4, np.pi / 2) - np.pi / 4)
    gamma_g += float(rng.uniform(-1, 1) * ec["grasp_yaw_noise_rad"])
    ox = float(rng.uniform(*ec["grasp_offset_x"]))
    oz = float(rng.uniform(*ec["grasp_offset_z"]))
    R_g = topdown_rotation(gamma_g)
    # TCP target = cube centre + x_h * ox + z_h * oz   (z_h points down)
    p_grasp = cube_pos + R_g[:, 0] * ox + R_g[:, 2] * oz
    p_pre = p_grasp + np.array([0.0, 0.0, ec["pregrasp_height"]])
    p_wp = np.array([rng.uniform(*ec["waypoint_x"]), rng.uniform(*ec["waypoint_y"]),
                     rng.uniform(*ec["waypoint_z"])])
    g_wp = gamma_g + float(rng.uniform(-1, 1) * ec["waypoint_yaw_noise_rad"])

    step = ec["ik_path_step"]
    tm = ec["timing"]
    go, gc = cfg["robot"]["grip_open"], cfg["robot"]["grip_closed"]

    q_wp, ok, ep, er = ik.solve(q_start, p_wp, topdown_rotation(g_wp), q_ref=q_ref, max_iter=600)
    if not ok:
        raise PlanningError(f"waypoint IK failed ({ep:.4f}, {er:.4f})")

    segs: list[Segment] = []

    def add_hold(name, q, T, g0, g1):
        segs.append(Segment(name, T, np.asarray(q)[None, :], g0, g1))

    segs.append(Segment("to_waypoint", tm["to_waypoint"], np.array([q_start, q_wp]), go, go))
    qs = _cartesian_segment(ik, q_wp, p_wp, g_wp, p_pre, gamma_g, q_ref, step)
    segs.append(Segment("to_pregrasp", tm["to_pregrasp"], qs, go, go))
    qs = _cartesian_segment(ik, qs[-1], p_pre, gamma_g, p_grasp, gamma_g, q_ref, step)
    segs.append(Segment("descend", tm["descend"], qs, go, go))
    q_g = qs[-1]
    add_hold("settle_pre", q_g, tm["settle_pre"], go, go)
    add_hold("close", q_g, tm["close"], go, gc)
    add_hold("grasp_settle", q_g, tm["grasp_settle"], gc, gc)
    p_lift = p_grasp + np.array([0.0, 0.0, ec["lift_height"]])
    qs = _cartesian_segment(ik, q_g, p_grasp, gamma_g, p_lift, gamma_g, q_ref, step)
    segs.append(Segment("lift", tm["lift"], qs, gc, gc))
    add_hold("hold", qs[-1], tm["hold"], gc, gc)
    segs.append(Segment("lower", tm["lower"], qs[::-1].copy(), gc, gc))
    add_hold("release", q_g, tm["release"], gc, go)
    p_ret = p_grasp + np.array([0.0, 0.0, ec["retreat_height"]])
    qs = _cartesian_segment(ik, q_g, p_grasp, gamma_g, p_ret, gamma_g, q_ref, step)
    segs.append(Segment("retreat", tm["retreat"], qs, go, go))

    meta = dict(gamma_grasp=gamma_g, grasp_offset_x=ox, grasp_offset_z=oz,
                waypoint=p_wp.tolist(), q_ref=q_ref.tolist())
    return segs, meta


def _finger_contact_count(model: mujoco.MjModel, data: mujoco.MjData, cube_gid: int,
                          finger_bodies: set[int]) -> int:
    touching: set[int] = set()
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == cube_gid or c.geom2 == cube_gid:
            other = c.geom2 if c.geom1 == cube_gid else c.geom1
            b = int(model.geom_bodyid[other])
            if b in finger_bodies:
                touching.add(b)
    return len(touching)


def sample_cube_pose(cfg: dict[str, Any], seed: int, episode_id: int):
    """First random draws of an episode: cube position and yaw. Cheap (no simulation).

    Returns the generator *after* these draws (run_episode keeps consuming it, so the draw order
    is part of the reproducibility contract) together with the cube position and yaw.
    """
    rng = make_rng(int(seed), int(episode_id))
    ec = cfg["episode"]
    half = 0.5 * float(cfg["cube"]["size"])
    cube_pos = np.array([rng.uniform(*ec["cube_x"]), rng.uniform(*ec["cube_y"]), half])
    cube_yaw = float(rng.uniform(-np.pi, np.pi))
    return rng, cube_pos, cube_yaw


def run_episode(cfg: dict[str, Any], episode_id: int, *, model: mujoco.MjModel | None = None,
                ik: ArmIK | None = None, render: bool = True,
                master_seed: int | None = None) -> dict[str, Any]:
    """Generate one episode. Fully determined by (cfg, master_seed, episode_id)."""
    seed = int(cfg["seed"] if master_seed is None else master_seed)
    model = model if model is not None else build_model(cfg)
    ik = ik if ik is not None else ArmIK(model)
    data = mujoco.MjData(model)
    rng, cube_pos, cube_yaw = sample_cube_pose(cfg, seed, episode_id)
    ec = cfg["episode"]
    half = 0.5 * float(cfg["cube"]["size"])
    q_home = np.array(cfg["robot"]["home_qpos"], dtype=float)
    q_start = np.clip(q_home + rng.uniform(-1, 1, 7) * ec["home_noise_rad"], ik.lo, ik.hi)

    segs = meta = None
    for attempt in range(int(ec["max_plan_attempts"])):
        try:
            segs, meta = plan_episode(cfg, ik, rng, q_start, cube_pos, cube_yaw)
            break
        except PlanningError:
            continue
    if segs is None:
        raise PlanningError(f"episode {episode_id}: no feasible plan in {ec['max_plan_attempts']} attempts")
    meta["plan_attempts"] = attempt + 1

    # ---- initial simulation state -------------------------------------------------------
    arm_q = np.array([model.joint(n).qposadr[0] for n in ARM_JOINTS])
    arm_d = np.array([model.joint(n).dofadr[0] for n in ARM_JOINTS])
    fin_q = np.array([model.joint(n).qposadr[0] for n in FINGER_JOINTS])
    fin_d = np.array([model.joint(n).dofadr[0] for n in FINGER_JOINTS])
    cj = model.joint(CUBE_JOINT)
    cq, cd = int(cj.qposadr[0]), int(cj.dofadr[0])
    kp = np.array([model.actuator_gainprm[model.actuator(f"actuator{i + 1}").id, 0] for i in range(7)])
    grip_act = model.actuator("actuator8").id
    cube_bid = model.body(CUBE_BODY).id
    cube_gid = model.geom(CUBE_GEOM).id
    tcp_sid = model.site(TCP_SITE).id
    finger_bodies = {model.body("left_finger").id, model.body("right_finger").id}
    sens = {model.sensor(i).name: (int(model.sensor(i).adr[0]), int(model.sensor(i).dim[0]))
            for i in range(model.nsensor)}
    f_adr, t_adr = sens["ft_force"][0], sens["ft_torque"][0]

    mujoco.mj_resetData(model, data)
    data.qpos[arm_q] = q_start
    data.qpos[fin_q] = 0.04
    data.qpos[cq:cq + 3] = cube_pos
    data.qpos[cq + 3:cq + 7] = [math.cos(cube_yaw / 2), 0.0, 0.0, math.sin(cube_yaw / 2)]
    mujoco.mj_forward(model, data)

    def command(q_des: np.ndarray, grip: float) -> None:
        data.ctrl[:7] = q_des + data.qfrc_bias[arm_d] / kp      # gravity feed-forward
        data.ctrl[grip_act] = grip

    dt = float(model.opt.timestep)
    sub = int(round(1.0 / (cfg["log_hz"] * dt)))
    assert abs(sub * dt * cfg["log_hz"] - 1.0) < 1e-9, "log_hz must divide 1/timestep"
    go = cfg["robot"]["grip_open"]
    for _ in range(int(0.4 / dt)):                              # settle: robot holds q_start, cube rests
        command(q_start, go)
        mujoco.mj_step(model, data)

    # ---- time grid ----------------------------------------------------------------------
    bounds = np.cumsum([0.0] + [s.duration for s in segs])
    n_frames = int(round(bounds[-1] * cfg["log_hz"])) + 1
    size = int(cfg["image_size"])
    renderer = mujoco.Renderer(model, height=size, width=size) if render else None
    cam = cfg["camera"]["name"]

    out = {k: [] for k in ("t", "phase", "segment", "image", "q", "qd", "finger_q", "wrench",
                           "cube_pos", "cube_quat", "cube_yaw", "cube_tilt", "cube_vel",
                           "tcp_pos", "n_finger_contacts", "q_cmd")}
    wrench_acc = np.zeros(6)
    wrench_n = 0
    q_cmd = q_start.copy()

    def log(w: np.ndarray) -> None:
        # mj_step leaves xpos/xmat/site_xpos/contacts computed from the state *before* its last
        # integration step. Recompute them so every logged channel refers to the same instant.
        mujoco.mj_forward(model, data)
        R = data.xmat[cube_bid].reshape(3, 3)
        out["t"].append(data.time - t_origin)
        out["phase"].append(SEGMENT_PHASE[segs[seg_idx].name])
        out["segment"].append(seg_idx)
        out["image"].append(np.zeros((size, size, 3), np.uint8) if renderer is None else
                            (renderer.update_scene(data, camera=cam) or renderer.render()).copy())
        out["q"].append(data.qpos[arm_q].copy())
        out["qd"].append(data.qvel[arm_d].copy())
        out["finger_q"].append(data.qpos[fin_q].copy())
        out["wrench"].append(w.copy())
        out["cube_pos"].append(data.xpos[cube_bid].copy())
        out["cube_quat"].append(data.xquat[cube_bid].copy())
        out["cube_yaw"].append(math.atan2(R[1, 0], R[0, 0]))
        out["cube_tilt"].append(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))
        out["cube_vel"].append(data.qvel[cd:cd + 6].copy())
        out["tcp_pos"].append(data.site_xpos[tcp_sid].copy())
        out["n_finger_contacts"].append(_finger_contact_count(model, data, cube_gid, finger_bodies))
        out["q_cmd"].append(q_cmd.copy())

    t_origin = data.time
    seg_idx = 0
    command(segs[0].q_at(0.0), segs[0].grip_at(0.0))
    mujoco.mj_forward(model, data)
    log(np.concatenate([data.sensordata[f_adr:f_adr + 3], data.sensordata[t_adr:t_adr + 3]]))

    for k in range(1, n_frames):
        wrench_acc[:] = 0.0
        for j in range(sub):
            t = data.time - t_origin
            seg_idx = int(np.clip(np.searchsorted(bounds, t, side="right") - 1, 0, len(segs) - 1))
            seg = segs[seg_idx]
            q_cmd = seg.q_at(t - bounds[seg_idx])
            command(q_cmd, seg.grip_at(t - bounds[seg_idx]))
            mujoco.mj_step(model, data)
            wrench_acc += np.concatenate([data.sensordata[f_adr:f_adr + 3],
                                          data.sensordata[t_adr:t_adr + 3]])
        t = data.time - t_origin
        seg_idx = int(np.clip(np.searchsorted(bounds, t - 1e-9, side="right") - 1, 0, len(segs) - 1))
        log(wrench_acc / sub)
    if renderer is not None:
        renderer.close()

    ep: dict[str, Any] = {k: np.asarray(v) for k, v in out.items()}
    ep["cube_phi"] = wrap_pi(4.0 * ep["cube_yaw"])                 # symmetric-cube yaw representation
    ep["segment_names"] = [s.name for s in segs]
    ep["segment_bounds"] = bounds
    ep["phase_names"] = PHASES

    # ---- outcome labels (failed grasps are kept and flagged, never silently dropped) -----
    z0 = half
    hold_idx = np.where(ep["phase"] == 3)[0]
    lifted = float(ep["cube_pos"][hold_idx[-1], 2] - z0) if len(hold_idx) else 0.0
    tilt_hold = float(np.degrees(ep["cube_tilt"][hold_idx].max())) if len(hold_idx) else 180.0
    tcp_gap = float(np.linalg.norm(ep["cube_pos"][hold_idx[-1]] - ep["tcp_pos"][hold_idx[-1]])) if len(hold_idx) else 1.0
    succ = (lifted >= ec["success"]["min_lift"] and tilt_hold <= ec["success"]["max_tilt_deg"]
            and tcp_gap < 0.03)
    meta.update(episode_id=int(episode_id), seed=seed, cube_pos0=cube_pos.tolist(), cube_yaw0=cube_yaw,
                q_start=q_start.tolist(), lifted_m=lifted, max_tilt_hold_deg=tilt_hold,
                tcp_cube_gap_m=tcp_gap, success=bool(succ), mujoco_version=mujoco.__version__)
    ep["meta"] = meta
    return ep

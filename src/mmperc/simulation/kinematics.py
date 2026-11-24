"""Rotation helpers and a damped-least-squares inverse-kinematics (IK) solver.

IK is used only to *generate* the scripted trajectory (the control side of the project).
The perception models never see IK outputs: they see measured joint angles, from which
forward kinematics (`ArmIK.fk`) provides a physically meaningful baseline.
"""
from __future__ import annotations

import mujoco
import numpy as np

from mmperc.simulation.scene import ARM_JOINTS, TCP_SITE


def wrap_pi(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def yaw_of(R: np.ndarray) -> float:
    """Yaw of a rotation matrix: heading of its x axis in the world xy-plane."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def topdown_rotation(gamma: float) -> np.ndarray:
    """Hand orientation with z_h = -z_world (fingers down) and closing axis y_h at angle `gamma`.

    Columns are the hand axes (x_h, y_h, z_h) in world coordinates; right-handed.
    """
    y = np.array([np.cos(gamma), np.sin(gamma), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def rotation_error(R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """World-frame rotation vector (small-angle) that rotates R towards R_des."""
    return 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))


class ArmIK:
    """Position + orientation IK for the 7-DoF arm with a null-space posture term.

    Uses a private MjData, so it never disturbs the simulation state.
    """

    def __init__(self, model: mujoco.MjModel, site_name: str = TCP_SITE,
                 joint_names=ARM_JOINTS, damping: float = 0.02, null_gain: float = 1.0):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site_name).id
        joints = [model.joint(n) for n in joint_names]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dadr = np.array([j.dofadr[0] for j in joints])
        self.lo = np.array([model.jnt_range[j.id, 0] for j in joints])
        self.hi = np.array([model.jnt_range[j.id, 1] for j in joints])
        self.damping = damping
        self.null_gain = null_gain
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP position (3,) and rotation matrix (3,3) in the world frame for arm angles q."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def solve(self, q0: np.ndarray, p_des: np.ndarray, R_des: np.ndarray,
              q_ref: np.ndarray | None = None, tol_p: float = 5e-4, tol_r: float = 2e-3,
              max_iter: int = 300, max_step: float = 0.15) -> tuple[np.ndarray, bool, float, float]:
        """Iterate DLS IK from q0. Returns (q, converged, |pos err| m, |rot err| rad)."""
        q = np.clip(np.asarray(q0, dtype=float).copy(), self.lo, self.hi)
        q_ref = q.copy() if q_ref is None else np.asarray(q_ref, dtype=float)
        eye6 = np.eye(6)
        eye7 = np.eye(7)
        ep = er = np.inf
        for _ in range(max_iter):
            self.data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site_id)
            p = self.data.site_xpos[self.site_id]
            R = self.data.site_xmat[self.site_id].reshape(3, 3)
            e = np.concatenate([p_des - p, rotation_error(R, R_des)])
            ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if ep < tol_p and er < tol_r:
                return q, True, ep, er
            J = np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])      # 6 x 7
            J_dls = J.T @ np.linalg.solve(J @ J.T + self.damping ** 2 * eye6, eye6)
            # The null-space projector must use the *exact* pseudo-inverse: with the damped one,
            # (I - J_dls J) is not a true projector and the posture term leaks into the task error.
            N = eye7 - np.linalg.pinv(J, rcond=1e-4) @ J
            dq = J_dls @ e + N @ (self.null_gain * (q_ref - q))
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q, False, ep, er

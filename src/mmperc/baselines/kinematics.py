"""Batch forward kinematics / Jacobians of the TCP from (possibly corrupted) joint angles."""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mmperc.simulation.scene import ARM_JOINTS, TCP_SITE, build_model


class HandKinematics:
    def __init__(self, sim_cfg: dict[str, Any]):
        self.model = build_model(sim_cfg)
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site(TCP_SITE).id
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.dadr = np.array([self.model.joint(n).dofadr[0] for n in ARM_JOINTS])
        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros((3, self.model.nv))

    def fk(self, Q: np.ndarray, jac: bool = False):
        """TCP positions (N,3), rotations (N,3,3) [columns = hand axes in world] and, if `jac`,
        position Jacobians (N,3,7) and the yaw-rate Jacobian row (N,7)."""
        n = len(Q)
        P, R = np.empty((n, 3)), np.empty((n, 3, 3))
        Jp, Jz = (np.empty((n, 3, 7)), np.empty((n, 7))) if jac else (None, None)
        m, d = self.model, self.data
        for i in range(n):
            d.qpos[self.qadr] = Q[i]
            mujoco.mj_kinematics(m, d)
            P[i] = d.site_xpos[self.site]
            R[i] = d.site_xmat[self.site].reshape(3, 3)
            if jac:
                mujoco.mj_comPos(m, d)
                mujoco.mj_jacSite(m, d, self._jp, self._jr, self.site)
                Jp[i] = self._jp[:, self.dadr]
                Jz[i] = self._jr[2, self.dadr]
        return (P, R, Jp, Jz) if jac else (P, R)


def hand_yaw(R: np.ndarray) -> np.ndarray:
    """Heading gamma of the closing axis y_h in the world xy-plane (matches the scripted grasp)."""
    return np.arctan2(R[:, 1, 1], R[:, 0, 1])


def to_hand(R: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """Express world vectors in the hand frame: R^T v."""
    return np.einsum("nji,nj->ni", R, v_world)


def to_world(R: np.ndarray, v_hand: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", R, v_hand)

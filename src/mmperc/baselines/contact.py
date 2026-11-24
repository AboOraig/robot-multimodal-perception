"""B5: physics-based in-hand offset from the wrist force/torque (valid only while holding, statically).

Model (all vectors in the sensor frame S): a rigidly held cube of weight W and centre of mass r
contributes force F_c and torque tau_c = r x F_c. The sensor also carries the hand's own wrench,
which depends only on the gravity direction in S:   F_h = a * g_S + b_F,   tau_h = c_h x F_h + b_T.
(a, b_F, c_h, b_T are fitted on free, static frames of the train split: a 'tare' calibration.)
Residual wrench: F_r = F - F_h, tau_r = tau - tau_h. Then r_perp = (F_r x tau_r) / |F_r|^2 gives the
component of r perpendicular to F_r; the component along F_r is unobservable and is fixed by the
nominal height of the cube centre below the flange. The in-hand offset is delta = r - (0, 0, tcp_z).

What it can and cannot do: it measures the in-hand offset (the 2 dof that dominate the kinematic
baseline's uncertainty) but NOT the world pose; and only when the hand is static (accelerations add
wrench components), the gripper is closed on the cube, and the wrench channel is available.
"""
from __future__ import annotations

import numpy as np

G_WORLD = np.array([0.0, 0.0, -9.81])


def skew(v: np.ndarray) -> np.ndarray:
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1), np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def static_mask(qd: np.ndarray, episode: np.ndarray, dt: float, v0: float = 0.01, a0: float = 0.1,
                sigma_qd: float = 0.0) -> np.ndarray:
    """Causal 'hand is static' detector from joint velocities: |qd| small AND |qd_t - qd_{t-1}|/dt small.

    The acceleration test matters: a minimum-jerk motion starts with zero velocity but large
    acceleration, and the resulting inertial wrench is as large as the in-hand torque signal.
    Thresholds widen with the (assumed known) velocity-noise std `sigma_qd` of the 7 joints.
    """
    v = np.linalg.norm(qd, axis=1)
    dq = np.empty_like(qd)
    dq[1:] = qd[1:] - qd[:-1]
    first = np.ones(len(qd), bool)
    first[1:] = episode[1:] != episode[:-1]
    a = np.linalg.norm(dq, axis=1) / dt
    thr_v = v0 + 3.0 * sigma_qd * np.sqrt(7.0)
    thr_a = a0 + 3.0 * sigma_qd * np.sqrt(14.0) / dt
    return (v < thr_v) & (a < thr_a) & ~first


class ContactOffset:
    def __init__(self, tcp_z: float, dt: float = 0.05, static_qd: float = 0.03, min_force_frac: float = 0.5,
                 v0: float = 0.01, a0: float = 0.1, force_band: tuple = (0.8, 1.2), lift_z_min: float = 0.05):
        self.tcp_z, self.dt, self.static_qd, self.min_force_frac = tcp_z, dt, static_qd, min_force_frac
        self.v0, self.a0, self.force_band, self.lift_z_min = v0, a0, force_band, lift_z_min

    def fit(self, train: dict, R: np.ndarray, delta_nom: np.ndarray, mass: float) -> "ContactOffset":
        """`R`: hand rotations of the train frames (from clean q). Uses clean wrenches."""
        gS = np.einsum("nji,j->ni", R, G_WORLD)
        free = (train["n_finger_contacts"] == 0) & (np.linalg.norm(train["qd"], axis=1) < self.static_qd)
        F, T, g = train["wrench"][free, :3], train["wrench"][free, 3:], gS[free]
        # F = a g + bF  (least squares for the scalar a and the vector bF)
        A = np.concatenate([g.reshape(-1, 1), np.tile(np.eye(3), (len(g), 1))], axis=1)
        sol = np.linalg.lstsq(A, F.reshape(-1), rcond=None)[0]
        self.a, self.bF = sol[0], sol[1:]
        Fh = self.a * g + self.bF
        # tau = c x F + bT = -[F]x c + bT
        A = np.concatenate([-skew(Fh).reshape(-1, 3), np.tile(np.eye(3), (len(g), 1))], axis=1)
        sol = np.linalg.lstsq(A, T.reshape(-1), rcond=None)[0]
        self.c, self.bT = sol[:3], sol[3:]
        self.mass, self.delta_z_nom = mass, float(delta_nom[2])
        # empirical residual of the offset estimate on clean, valid, held frames
        st = static_mask(train["qd"], train["episode"], self.dt, self.v0, self.a0)
        d_hat, ok = self.estimate(train["wrench"], train["qd"], R, np.ones(len(R), bool), st, train["tcp_pos"][:, 2])
        gate = ok & train["holding_gate"]
        d_true = np.einsum("nji,nj->ni", R, train["cube_pos"] - train["tcp_pos"])
        res = (d_hat - d_true)[gate]
        self.res_bias, self.res_std = res.mean(0), res.std(0)
        self.fit_frames = int(free.sum()), int(gate.sum())
        return self

    def hand_wrench(self, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gS = np.einsum("nji,j->ni", R, G_WORLD)
        Fh = self.a * gS + self.bF
        Th = np.cross(self.c, Fh) + self.bT
        return Fh, Th

    def estimate(self, wrench: np.ndarray, qd: np.ndarray, R: np.ndarray,
                 channel_ok: np.ndarray, static: np.ndarray | None = None,
                 tcp_height: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (delta_hat (N,3) in the hand frame about the TCP, valid (N,)). The caller must
        also require the holding gate (not known here: it depends on finger_q)."""
        Fh, Th = self.hand_wrench(R)
        Fr, Tr = wrench[:, :3] - Fh, wrench[:, 3:] - Th
        n2 = (Fr ** 2).sum(1)
        r_perp = np.cross(Fr, Tr) / np.maximum(n2, 1e-9)[:, None]
        Fhat = Fr / np.sqrt(np.maximum(n2, 1e-9))[:, None]
        lam = ((self.tcp_z + self.delta_z_nom) - r_perp[:, 2]) / np.where(np.abs(Fhat[:, 2]) > 0.3, Fhat[:, 2], 1.0)
        r = r_perp + lam[:, None] * Fhat
        delta = r - np.array([0.0, 0.0, self.tcp_z])
        is_static = (np.linalg.norm(qd, axis=1) < self.static_qd) if static is None else static
        ratio = np.sqrt(n2) / (self.mass * 9.81)
        # Valid only if the cube hangs entirely from the hand: static, residual force ~ cube weight,
        # and the hand is lifted clear of the table (otherwise the table carries part of the load).
        valid = (channel_ok & is_static & (ratio > self.force_band[0]) & (ratio < self.force_band[1])
                 & (np.abs(Fhat[:, 2]) > 0.3))
        if tcp_height is not None:
            valid &= tcp_height > self.lift_z_min
        return delta, valid

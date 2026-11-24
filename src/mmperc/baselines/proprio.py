"""B3: physically meaningful kinematic estimate of the held cube from joint angles.

While the gripper is closed on the cube (observable: finger opening ~ cube width, `holding_gate`),
the cube is rigidly attached to the hand:   p_cube = p_tcp(q) + R(q) delta,   phi = 4 gamma(q) + dphi,
with the in-hand offset delta random (nominal mean delta_nom, covariance Sigma_delta calibrated on the
train split) and gamma the heading of the closing axis. Measurement noise on q is propagated through
the Jacobian when its standard deviation `sigma_q` is known (an assumption: sensor specification).

gated=True  (default): informative only while holding -> no policy-leakage benefit.
gated=False ("leaky"): applies the same formula in every frame; where the hand merely hovers above the
cube this exploits the scripted policy (see scripts/check_leakage.py) and must be reported as such.
Optional contact refinement replaces delta by the wrench-based estimate where that is valid.
"""
from __future__ import annotations

import numpy as np

from mmperc.baselines.common import Estimate, Prior, block_cov, wrap
from mmperc.baselines.contact import ContactOffset, static_mask
from mmperc.baselines.kinematics import HandKinematics, hand_yaw, to_hand, to_world


class FKProprio:
    def __init__(self, hk: HandKinematics, gated: bool = True, use_contact: bool = False, dt: float = 0.05):
        self.hk, self.gated, self.use_contact, self.dt = hk, gated, use_contact, dt
        self.contact: ContactOffset | None = None

    def fit(self, train: dict, tcp_z: float, mass: float) -> "FKProprio":
        P, R = self.hk.fk(train["q"])
        self.prior = Prior.fit(train)
        g = train["holding_gate"]
        d = to_hand(R[g], train["cube_pos"][g] - P[g])
        self.delta_nom, self.cov_delta = d.mean(0), np.cov(d.T)
        dphi = wrap(train["cube_phi"][g] - 4 * hand_yaw(R[g]))
        self.phi_off = float(np.angle(np.exp(1j * dphi).mean()))
        self.var_phi = float(np.var(wrap(dphi - self.phi_off)))
        if self.use_contact:
            self.contact = ContactOffset(tcp_z, dt=self.dt).fit({**train, "q": train["q"]}, R, self.delta_nom, mass)
            self.cov_delta_contact = np.diag(self.contact.res_std ** 2)
            self.contact_bias = self.contact.res_bias
        return self

    def predict(self, d: dict, sigma_q: float = 1e-3, sigma_qd: float = 0.0, sigma_wrench_torque: float = 0.0,
                sigma_wrench_force: float = 0.0, dt: float = 0.05) -> Estimate:
        n = len(d["q"])
        prior = self.prior.estimate(n)
        P, R, Jp, Jz = self.hk.fk(d["q"], jac=True)
        avail = d["proprio_mask"] if "proprio_mask" in d else np.ones(n, bool)
        ok = avail & d["holding_gate"] if self.gated else avail
        delta = np.tile(self.delta_nom, (n, 1))
        cov_delta = np.tile(self.cov_delta, (n, 1, 1))
        if self.use_contact:
            chan = d["contact_mask"] if "contact_mask" in d else np.ones(n, bool)
            st = static_mask(d["qd"], d["episode"], dt, self.contact.v0, self.contact.a0, sigma_qd)
            dh, valid = self.contact.estimate(d["wrench"], d["qd"], R, chan, st, P[:, 2])
            valid &= d["holding_gate"] & avail
            dh = dh - self.contact_bias
            # Precision-weighted (information form) fusion of the nominal in-hand offset prior with the
            # wrench measurement, for the x,y components; z (the gravity axis) stays nominal.
            # Offset error from wrench noise: torque noise acts through 1/|F|; force noise tilts the
            # force direction and acts through the lever arm z_c (~10.7 cm) from sensor to cube.
            z_c = self.contact.tcp_z + self.contact.delta_z_nom
            noise = (sigma_wrench_torque ** 2 + (z_c * sigma_wrench_force) ** 2) / max(self.contact.mass * 9.81, 1e-6) ** 2
            S0 = self.cov_delta[:2, :2]
            Sc = np.diag(self.contact.res_std[:2] ** 2 + noise)
            S0i, Sci = np.linalg.inv(S0), np.linalg.inv(Sc)
            Sp = np.linalg.inv(S0i + Sci)
            mean_xy = (Sp @ S0i @ self.delta_nom[:2])[None, :] + dh[:, :2] @ (Sp @ Sci).T
            delta[valid, :2] = mean_xy[valid]
            cov_delta[valid, :2, :2] = Sp
            cov_delta[valid, :2, 2] = 0.0
            cov_delta[valid, 2, :2] = 0.0
            self.last_contact_valid = valid
        mean = prior.mean.copy()
        mean[:, :3] = P + to_world(R, delta)
        mean[:, 3] = wrap(4 * hand_yaw(R) + self.phi_off)
        cov_p = np.einsum("nij,njk,nlk->nil", R, cov_delta, R)
        cov_p = cov_p + (sigma_q ** 2) * np.einsum("nij,nkj->nik", Jp, Jp)
        var_phi = self.var_phi + 16 * (sigma_q ** 2) * (Jz ** 2).sum(1)
        cov = block_cov(cov_p, var_phi)
        mean[~ok] = prior.mean[~ok]
        cov[~ok] = prior.cov[~ok]
        return Estimate(mean, cov, ok)

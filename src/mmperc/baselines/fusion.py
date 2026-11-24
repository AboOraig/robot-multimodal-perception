"""B1 (vision-only MAP) and B6 (classical WLS fusion of vision + kinematics [+ contact offset]).

All position estimates minimise a sum of squared Mahalanobis residuals (Gauss-Newton, batched over
frames):  prior + pinhole pixel residual (+ kinematic position measurement).
Assumptions of this classical fusion (valid in clean conditions, expected to break otherwise):
  * measurement errors are Gaussian, zero-mean and independent between sensors,
  * the noise covariances are those calibrated on clean train data (or, optionally, oracle values),
  * outliers are caught by a chi-square innovation test of the pixel measurement against the
    kinematic prediction (only possible while the kinematic estimate is informative).
Independence is NOT exactly true here: the kinematic estimate and the pixel measurement share the
physical state and the scripted trajectory, so the posterior covariance is optimistic by construction.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from mmperc.baselines.common import Estimate, Prior, block_cov, wrap
from mmperc.baselines.vision import ColourVision, centroid, ray_plane
from mmperc.sensors.degradation import CubeProjector


def _jac_pix(proj: CubeProjector, p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    J = np.empty((len(p), 2, 3))
    for k in range(3):
        dp = np.zeros(3); dp[k] = eps
        u1, v1, _ = proj.project(p + dp)
        u0, v0, _ = proj.project(p - dp)
        J[:, 0, k] = (u1 - u0) / (2 * eps)
        J[:, 1, k] = (v1 - v0) / (2 * eps)
    return J


def solve_position(proj: CubeProjector, prior: Prior, uv: np.ndarray, uv_use: np.ndarray, cov_pix: np.ndarray,
                   p_meas: np.ndarray | None = None, cov_meas: np.ndarray | None = None,
                   meas_use: np.ndarray | None = None, iters: int = 4, prior_alt: Prior | None = None,
                   use_alt: np.ndarray | None = None):
    """Batched Gauss-Newton. Returns (p (N,3), posterior covariance (N,3,3)).
    Frames flagged in `use_alt` use `prior_alt` instead of `prior` (e.g. table prior vs broad prior)."""
    n = len(uv)
    mean_n = np.tile(prior.mean_p, (n, 1))
    prec_n = np.tile(np.linalg.inv(prior.cov_p), (n, 1, 1))
    if prior_alt is not None and use_alt is not None and use_alt.any():
        mean_n[use_alt] = prior_alt.mean_p
        prec_n[use_alt] = np.linalg.inv(prior_alt.cov_p)
    Rinv = np.linalg.inv(cov_pix)
    p = mean_n.copy()
    have_v = uv_use
    if have_v.any():
        p[have_v] = ray_plane(proj, uv[have_v, 0], uv[have_v, 1], mean_n[have_v, 2])
    if meas_use is not None and meas_use.any():
        p[meas_use] = p_meas[meas_use]
    for _ in range(iters):
        H = prec_n.copy()
        b = np.einsum("nij,nj->ni", prec_n, mean_n - p)
        if have_v.any():
            J = _jac_pix(proj, p)
            u, v, _ = proj.project(p)
            r = uv - np.stack([u, v], 1)
            Jv = J * have_v[:, None, None]
            H += np.einsum("nki,kl,nlj->nij", Jv, Rinv, Jv)
            b += np.einsum("nki,kl,nl->ni", Jv, Rinv, r * have_v[:, None])
        if meas_use is not None and meas_use.any():
            W = np.zeros((n, 3, 3))
            W[meas_use] = np.linalg.inv(cov_meas[meas_use])
            H += W
            b += np.einsum("nij,nj->ni", W, p_meas - p)
        p = p + np.linalg.solve(H, b[..., None])[..., 0]
    return p, np.linalg.inv(H)


class VisionOnly:
    """B1."""

    def __init__(self, vision: ColourVision, prior: Prior, cube_half: float | None = None):
        """cube_half given -> 'table' variant (cube assumed resting on the table); None -> broad prior."""
        self.vision, self.broad = vision, prior
        self.prior = prior.table(cube_half) if cube_half is not None else prior

    def predict(self, d: dict, sigma_pix_scale: float = 1.0) -> Estimate:
        n = len(d["image"])
        proj = self.vision.proj
        # first pass (position only) -> hint for the yaw linearisation
        u0, v0, m0 = centroid(d["image"], self.vision.thr)
        use0 = m0 > self.vision.min_mass
        hint = np.tile(self.prior.mean_p, (n, 1))
        if use0.any():
            hint[use0] = ray_plane(proj, (u0 - self.vision.bias[0])[use0], (v0 - self.vision.bias[1])[use0], self.prior.mean_p[2])
        ms = self.vision.measure(d, hint)
        uv = np.stack([ms["u"], ms["v"]], 1)
        p, cov = solve_position(proj, self.prior, uv, ms["valid"], self.vision.cov_pix * sigma_pix_scale ** 2)
        phi = np.where(ms["valid"], ms["phi"], 0.0)
        var_phi = np.where(ms["valid"], self.vision.var_phi, np.pi ** 2 / 3)
        mean = np.concatenate([p, phi[:, None]], 1)
        cov4 = block_cov(cov, var_phi)
        base = self.prior.estimate(n)
        mean[~ms["valid"]] = base.mean[~ms["valid"]]
        cov4[~ms["valid"]] = base.cov[~ms["valid"]]
        return Estimate(mean, cov4, ms["valid"])


class WLSFusion:
    """B6: vision pixel measurement + kinematic (optionally contact-refined) estimate."""

    def __init__(self, vision: ColourVision, kin, prior: Prior, cube_half: float, gate_p: float = 0.99,
                 gate: bool = True):
        self.vision, self.kin, self.prior = vision, kin, prior
        self.table_prior = prior.table(cube_half)
        self.gate_thr = chi2.ppf(gate_p, 2)
        self.gate_thr_phi = chi2.ppf(gate_p, 1)
        self.gate = gate

    def predict(self, d: dict, kin_kwargs: dict | None = None, sigma_pix_scale: float = 1.0):
        n = len(d["image"])
        proj = self.vision.proj
        kin = self.kin.predict(d, **(kin_kwargs or {}))
        vis_only = VisionOnly(self.vision, self.prior, self.table_prior.mean_p[2]).predict(d, sigma_pix_scale)
        hint = np.where(vis_only.informative[:, None], vis_only.mean[:, :3], kin.mean[:, :3])
        ms = self.vision.measure(d, hint)
        uv = np.stack([ms["u"], ms["v"]], 1)
        cov_pix = self.vision.cov_pix * sigma_pix_scale ** 2
        use_v = ms["valid"].copy()
        # chi-square innovation test of the pixel measurement against the kinematic prediction
        gated_out = np.zeros(n, bool)
        if self.gate:
            k = kin.informative & use_v
            if k.any():
                u, v, _ = proj.project(kin.mean[:, :3])
                J = _jac_pix(proj, kin.mean[:, :3])
                S = cov_pix[None] + np.einsum("nij,njk,nlk->nil", J, kin.cov[:, :3, :3], J)
                nu = uv - np.stack([u, v], 1)
                d2 = np.einsum("ni,nij,nj->n", nu, np.linalg.inv(S), nu)
                gated_out = k & (d2 > self.gate_thr)
                use_v &= ~gated_out
        # while the gripper is not holding, the cube rests on the table (table prior); once holding is
        # detected it may be lifted, so the broad prior is used and the kinematics supply the height
        # Missing proprioception (dropout) means the holding state is UNKNOWN, not 'not holding': use the broad prior.
        proprio_ok = d["proprio_mask"] if "proprio_mask" in d else np.ones(n, bool)
        p, cov = solve_position(proj, self.table_prior, uv, use_v, cov_pix, kin.mean[:, :3], kin.cov[:, :3, :3],
                                kin.informative, prior_alt=self.prior, use_alt=kin.informative | ~proprio_ok)
        # yaw: circular WLS of the two yaw measurements
        wv = np.where(use_v, 1.0 / self.vision.var_phi, 0.0)
        wk = np.where(kin.informative, 1.0 / np.maximum(kin.cov[:, 3, 3], 1e-9), 0.0)
        if self.gate:
            both = (wv > 0) & (wk > 0)
            dphi = wrap(ms["phi"] - kin.mean[:, 3])
            out = both & (dphi ** 2 * (1 / (1 / np.maximum(wv, 1e-12) + 1 / np.maximum(wk, 1e-12))) > self.gate_thr_phi)
            wv = np.where(out, 0.0, wv)
        ref = np.where(wk > 0, kin.mean[:, 3], ms["phi"])
        wsum = wv + wk
        phi = np.where(wsum > 0, wrap(ref + (wv * wrap(ms["phi"] - ref)) / np.maximum(wsum, 1e-12)), 0.0)
        var_phi = np.where(wsum > 0, 1 / np.maximum(wsum, 1e-12), np.pi ** 2 / 3)
        informative = use_v | kin.informative
        mean = np.concatenate([p, phi[:, None]], 1)
        cov4 = block_cov(cov, var_phi)
        self.last = {"gated_out": gated_out, "vision_used": use_v, "kin_informative": kin.informative}
        return Estimate(mean, cov4, informative)

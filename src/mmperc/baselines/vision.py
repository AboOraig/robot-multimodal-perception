"""B1: classical geometric vision. Colour segmentation -> sub-pixel centroid -> pinhole MAP estimate.

Position: the (bias-corrected) pixel centroid is a 2-D measurement of the cube centre's projection,
with covariance calibrated on clean train images. One camera cannot observe depth, so the 3-D
estimate is the MAP solution with a weak Gaussian prior on position (train statistics); its
covariance is therefore long along the viewing ray whenever the cube is off the table plane.
Yaw: 4-fold harmonic of the edge-gradient orientation, mapped to the table plane with the local
image->plane Jacobian (phi = arg sum |g|^2 exp(i 4 theta); this is 4*yaw for a square).

Known weaknesses (by design, not bugs): fixed colour thresholds are brittle under strong
illumination / noise; partial occlusion biases the centroid without being detected; yaw from a
6-9 px blob is coarse.
"""
from __future__ import annotations

import numpy as np

from mmperc.baselines.common import wrap
from mmperc.sensors.degradation import CubeProjector


def redness(img: np.ndarray) -> np.ndarray:
    i = img.astype(np.float32)
    return i[..., 0] - 0.5 * (i[..., 1] + i[..., 2])


def centroid(img: np.ndarray, thr: float = 60.0, chunk: int = 2000):
    """Weighted centroid (u, v) in continuous pixel coordinates (pixel j centre at j + 0.5), mass
    (sum of weights) and the weight images' gradient harmonic inputs are computed elsewhere."""
    n, H, W = img.shape[:3]
    u, v, m = np.empty(n), np.empty(n), np.empty(n)
    jj = np.arange(W, dtype=np.float32) + 0.5
    ii = np.arange(H, dtype=np.float32) + 0.5
    for a in range(0, n, chunk):
        w = np.clip(redness(img[a:a + chunk]) - thr, 0, None)
        s = w.sum((1, 2))
        m[a:a + chunk] = s
        ok = np.maximum(s, 1e-6)
        u[a:a + chunk] = (w.sum(1) * jj).sum(1) / ok
        v[a:a + chunk] = (w.sum(2) * ii).sum(1) / ok
    return u, v, m


def plane_jacobian(proj: CubeProjector, p: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """A = d(u,v)/d(x,y) at world points p (N,3): image displacement per metre on the table plane."""
    A = np.empty((len(p), 2, 2))
    for k in range(2):
        dp = np.zeros(3); dp[k] = eps
        u1, v1, _ = proj.project(p + dp)
        u0, v0, _ = proj.project(p - dp)
        A[:, 0, k] = (u1 - u0) / (2 * eps)
        A[:, 1, k] = (v1 - v0) / (2 * eps)
    return A


def yaw_harmonic(img: np.ndarray, A: np.ndarray, thr: float = 60.0, chunk: int = 2000) -> np.ndarray:
    """Complex harmonic S4 = sum |g_w|^2 exp(i 4 theta_w) per frame; phi_hat = angle(S4).
    `A`: (N,2,2) image-per-plane Jacobians (gradients map covariantly: g_w = A^T g_img)."""
    n = len(img)
    out = np.empty(n, complex)
    for a in range(0, n, chunk):
        w = np.clip(redness(img[a:a + chunk]) - thr, 0, None)
        gv, gu = np.gradient(w, axis=(1, 2))
        Ac = A[a:a + chunk]
        gx = Ac[:, 0, 0, None, None] * gu + Ac[:, 1, 0, None, None] * gv
        gy = Ac[:, 0, 1, None, None] * gu + Ac[:, 1, 1, None, None] * gv
        g2 = gx ** 2 + gy ** 2
        # exp(i 4 theta) = ((gx + i gy)/|g|)^4 = (gx + i gy)^4 / |g|^4  -> weight |g|^2 gives (.)^4/|g|^2
        z = (gx + 1j * gy) ** 4 / np.maximum(g2, 1e-9)
        out[a:a + chunk] = z.sum((1, 2))
    return out


def ray_plane(proj: CubeProjector, u: np.ndarray, v: np.ndarray, z_plane: float | np.ndarray) -> np.ndarray:
    """World point where the pixel ray meets the plane z = z_plane."""
    d_cam = np.stack([(u - proj.cx) / proj.f, -(v - proj.cy) / proj.f, -np.ones_like(u)], axis=1)
    d_w = d_cam @ proj.R.T
    t = (z_plane - proj.pos[2]) / d_w[:, 2]
    return proj.pos + t[:, None] * d_w


class ColourVision:
    """Fit on train images; `measure` returns pixel measurement, yaw measurement and validity."""

    def __init__(self, proj: CubeProjector, thr: float = 60.0):
        self.proj, self.thr = proj, thr

    def fit(self, train: dict) -> "ColourVision":
        u, v, m = centroid(train["image"], self.thr)
        self.min_mass = 0.25 * float(np.median(m))
        ok = m > self.min_mass
        ut, vt, _ = self.proj.project(train["cube_pos"])
        r = np.stack([u - ut, v - vt], 1)[ok]
        self.bias = np.median(r, axis=0)
        r = r - self.bias
        self.cov_pix = np.cov(r.T)
        self.fit_frames = int(ok.sum())
        # yaw calibration on frames where the position is well determined
        A = plane_jacobian(self.proj, train["cube_pos"])
        S4 = yaw_harmonic(train["image"], A, self.thr)
        dphi = wrap(train["cube_phi"] - np.angle(S4))[ok]
        self.phi_off = float(np.angle(np.exp(1j * dphi).mean()))
        self.var_phi = float(np.var(wrap(dphi - self.phi_off)))
        return self

    def measure(self, d: dict, p_hint: np.ndarray) -> dict:
        """p_hint (N,3): approximate cube position used only to linearise the image->plane map for
        yaw (the vision-only MAP estimate, never the ground truth)."""
        u, v, m = centroid(d["image"], self.thr)
        valid = m > self.min_mass
        S4 = yaw_harmonic(d["image"], plane_jacobian(self.proj, p_hint), self.thr)
        return {"u": u - self.bias[0], "v": v - self.bias[1], "valid": valid, "mass": m,
                "phi": wrap(np.angle(S4) + self.phi_off)}

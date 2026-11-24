"""Per-episode sufficient statistics: everything needed to recompute (and bootstrap) the metrics offline.

For an estimate and the ground truth, `episode_stats` returns an array (n_episodes, 6, K): axis 1 is the 5 stages
(free, hover, grasp, coupled, post) plus index 5 = all frames; axis 2 holds the SUMS over frames listed in `FIELDS`.
Sums (not means) are stored so that any resampling of episodes gives exact pooled metrics: RMSE = sqrt(sum/n).
`calibration_stats` returns pooled reliability counts and the binned error-vs-predicted-sigma table.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from mmperc.baselines.common import Estimate
from mmperc.evaluation.metrics import per_frame

FIELDS = ("n", "err3_sq", "errxy_sq", "errz_abs", "yaw_abs_deg", "d2", "nll", "in68", "in95")
GROUPS = ("free", "hover", "grasp", "coupled", "post", "all")
LEVELS = np.linspace(0.05, 0.95, 19)                       # nominal coverage levels of the reliability diagram
SIGMA_EDGES_MM = np.logspace(-1, 3, 25)                    # 0.1 mm .. 1 m, 24 bins


def _groups(stage: np.ndarray) -> np.ndarray:
    return np.concatenate([np.stack([stage == s for s in range(5)], 1), np.ones((len(stage), 1), bool)], 1)   # (N, 6)


def episode_stats(est: Estimate, target: np.ndarray, episode: np.ndarray, stage: np.ndarray, n_ep: int) -> np.ndarray:
    f = per_frame(est, target)
    vals = np.stack([np.ones(len(target)), f["err3_sq"], f["errxy_sq"], f["errz_abs"], f["yaw_abs_deg"], f["d2"], f["nll"],
                     (f["d2"] <= chi2.ppf(0.68, 4)).astype(float), (f["d2"] <= chi2.ppf(0.95, 4)).astype(float)], 1)   # (N, K)
    g = _groups(stage).astype(float)
    out = np.zeros((n_ep, 6, len(FIELDS)))
    for k in range(len(FIELDS)):
        for j in range(6):
            out[:, j, k] = np.bincount(episode, weights=vals[:, k] * g[:, j], minlength=n_ep)
    return out


def calibration_stats(est: Estimate, target: np.ndarray, stage: np.ndarray) -> dict[str, np.ndarray]:
    """reliab (6, n_levels): frames with d2 <= chi2 quantile; counts (6,); sigma_bins (6, n_bins, 3) = [count, sum errxy^2, sum sigma_xy^2]."""
    f = per_frame(est, target)
    g = _groups(stage)
    thr = chi2.ppf(LEVELS, 4)
    sig = 1e3 * np.sqrt(0.5 * (est.cov[:, 0, 0] + est.cov[:, 1, 1]))          # predicted xy sigma (mm)
    b = np.clip(np.digitize(sig, SIGMA_EDGES_MM) - 1, 0, len(SIGMA_EDGES_MM) - 2)
    errsq = 1e6 * f["errxy_sq"]
    reliab = np.stack([(f["d2"][g[:, j], None] <= thr[None]).sum(0) for j in range(6)])
    counts = g.sum(0)
    nb = len(SIGMA_EDGES_MM) - 1
    sb = np.zeros((6, nb, 3))
    for j in range(6):
        m = g[:, j]
        sb[j, :, 0] = np.bincount(b[m], minlength=nb)
        sb[j, :, 1] = np.bincount(b[m], weights=errsq[m], minlength=nb)
        sb[j, :, 2] = np.bincount(b[m], weights=sig[m] ** 2, minlength=nb)
    return {"reliab": reliab.astype(float), "counts": counts.astype(float), "sigma_bins": sb}

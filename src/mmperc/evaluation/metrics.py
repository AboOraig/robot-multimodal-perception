"""Point-estimate, probabilistic and calibration metrics, with episode-level cluster bootstrap.

Frames within an episode are strongly autocorrelated, so the unit of replication is the episode:
confidence intervals resample episodes, never frames.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from mmperc.baselines.common import Estimate, wrap

LOG2PI = np.log(2 * np.pi)


def tangent_error(mean: np.ndarray, target: np.ndarray) -> np.ndarray:
    """e = [p - p_hat; wrap(phi - phi_hat)], shape (N, 4). `target` = [x y z phi]."""
    e = target - mean
    e[:, 3] = wrap(e[:, 3])
    return e


def per_frame(est: Estimate, target: np.ndarray) -> dict[str, np.ndarray]:
    e = tangent_error(est.mean, target)
    L = np.linalg.cholesky(est.cov + 1e-12 * np.eye(4))
    z = np.linalg.solve(L, e[..., None])[..., 0]                    # L^-1 e
    d2 = (z ** 2).sum(1)                                            # Mahalanobis distance squared
    nll = 0.5 * d2 + np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1) + 2 * LOG2PI
    return {"err3_sq": (e[:, :3] ** 2).sum(1), "errxy_sq": (e[:, :2] ** 2).sum(1), "errz_abs": np.abs(e[:, 2]),
            "yaw_abs_deg": np.degrees(np.abs(e[:, 3])) / 4.0, "d2": d2, "nll": nll}


def summarize(est: Estimate, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    f = per_frame(est, target)
    m = np.ones(len(target), bool) if mask is None else mask
    if not m.any():
        return {}
    cov95 = (f["d2"][m] <= chi2.ppf(0.95, 4)).mean()
    cov68 = (f["d2"][m] <= chi2.ppf(0.68, 4)).mean()
    return {"n": int(m.sum()), "rmse3_mm": 1e3 * float(np.sqrt(f["err3_sq"][m].mean())),
            "rmse_xy_mm": 1e3 * float(np.sqrt(f["errxy_sq"][m].mean())),
            "mae_z_mm": 1e3 * float(f["errz_abs"][m].mean()),
            "yaw_mae_deg": float(f["yaw_abs_deg"][m].mean()),
            "nll": float(f["nll"][m].mean()), "mean_d2": float(f["d2"][m].mean()),     # 4 if calibrated
            "cov68": float(cov68), "cov95": float(cov95)}


def cluster_bootstrap_ci(values: np.ndarray, episode: np.ndarray, mask: np.ndarray | None = None,
                         stat: str = "rmse", n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(estimate, lo, hi) of sqrt(mean(values)) ('rmse', values are squared errors) or mean, 95% CI,
    resampling whole episodes."""
    m = np.ones(len(values), bool) if mask is None else mask
    eps = np.unique(episode[m])
    sums = np.array([values[m & (episode == e)].sum() for e in eps])
    cnts = np.array([(m & (episode == e)).sum() for e in eps], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(eps), (n_boot, len(eps)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    f = (lambda x: np.sqrt(x)) if stat == "rmse" else (lambda x: x)
    return float(f(sums.sum() / cnts.sum())), float(f(np.percentile(boot, 2.5))), float(f(np.percentile(boot, 97.5)))

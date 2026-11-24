"""Paired episode-level cluster bootstrap and Holm correction (the decision statistics of docs/preregistration.md).

A method is described by per-episode sums `S` (n_ep, K) from `suffstats.episode_stats` for ONE (scenario, group);
several training seeds are combined by averaging the sums (= the pooled metric of the seed-averaged squared errors).
Two methods evaluated on the SAME episodes are compared with the SAME resampled episode indices, which removes the
between-episode variance (difficulty of each episode) from the comparison.
"""
from __future__ import annotations

import numpy as np

from mmperc.evaluation.suffstats import FIELDS

_IDX = {k: i for i, k in enumerate(FIELDS)}


def metric(S: np.ndarray, name: str) -> float:
    """Pooled metric from per-episode sums S (n_ep, K)."""
    n = S[:, _IDX["n"]].sum()
    if name == "rmse_xy_mm":
        return 1e3 * float(np.sqrt(S[:, _IDX["errxy_sq"]].sum() / n))
    if name == "rmse3_mm":
        return 1e3 * float(np.sqrt(S[:, _IDX["err3_sq"]].sum() / n))
    if name == "yaw_mae_deg":
        return float(S[:, _IDX["yaw_abs_deg"]].sum() / n)
    if name == "nll":
        return float(S[:, _IDX["nll"]].sum() / n)
    if name == "mean_d2":
        return float(S[:, _IDX["d2"]].sum() / n)
    if name == "cov95":
        return float(S[:, _IDX["in95"]].sum() / n)
    if name == "cov68":
        return float(S[:, _IDX["in68"]].sum() / n)
    if name == "abs_d2_gap":                                # |mean d2 - 4|: 0 = calibrated
        return abs(float(S[:, _IDX["d2"]].sum() / n) - 4.0)
    raise ValueError(name)


def _boot_metric(S: np.ndarray, idx: np.ndarray, name: str) -> np.ndarray:
    """metric on each bootstrap resample; idx (B, n_ep)."""
    T = S[idx]                                             # (B, n_ep, K)
    n = T[..., _IDX["n"]].sum(1)
    if name in ("rmse_xy_mm", "rmse3_mm"):
        k = "errxy_sq" if name == "rmse_xy_mm" else "err3_sq"
        return 1e3 * np.sqrt(T[..., _IDX[k]].sum(1) / n)
    key = {"yaw_mae_deg": "yaw_abs_deg", "nll": "nll", "mean_d2": "d2", "cov95": "in95", "cov68": "in68"}.get(name)
    if key:
        return T[..., _IDX[key]].sum(1) / n
    if name == "abs_d2_gap":
        return np.abs(T[..., _IDX["d2"]].sum(1) / n - 4.0)
    raise ValueError(name)


def paired_bootstrap(A: np.ndarray, B: np.ndarray, name: str, n_boot: int = 5000, seed: int = 0, ratio: bool = False) -> dict:
    """Difference metric(A) - metric(B) (or the ratio metric(A)/metric(B) if ratio=True; null value 1), 95% CI and
    one-sided p-values: p_better = P*(diff >= null) (small = A lower/better), p_worse = P*(diff <= null) (small = A higher/worse).
    A and B: per-episode sums (n_ep, K) on the same episodes."""
    assert A.shape == B.shape
    n_ep = len(A)
    idx = np.random.default_rng(seed).integers(0, n_ep, (n_boot, n_ep))
    a, b = _boot_metric(A, idx, name), _boot_metric(B, idx, name)
    d = a / b if ratio else a - b
    null = 1.0 if ratio else 0.0
    est = metric(A, name) / metric(B, name) if ratio else metric(A, name) - metric(B, name)
    return {"diff": float(est), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_better": float((np.sum(d >= null) + 1) / (n_boot + 1)), "p_worse": float((np.sum(d <= null) + 1) / (n_boot + 1)),
            "a": metric(A, name), "b": metric(B, name)}


def holm(pvals: list[float], alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Holm-Bonferroni step-down. Returns (adjusted p, reject) in the input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return [(float(a), bool(a < alpha)) for a in adj]


def seed_average(stats_per_seed: list[np.ndarray]) -> np.ndarray:
    """Average of per-episode sums over training seeds (same episodes, same frame counts)."""
    return np.mean(np.stack(stats_per_seed), 0)


def seed_sign_consistency(per_seed_A: list[np.ndarray], per_seed_B: list[np.ndarray], name: str) -> tuple[int, int]:
    """(# seeds where metric(A_s) < metric(B_s), # seeds); seeds are paired by index (same data, different init)."""
    wins = sum(metric(a, name) < metric(b, name) for a, b in zip(per_seed_A, per_seed_B))
    return int(wins), len(per_seed_A)


def relative_gain_contrast(A1, B1, A0, B0, name: str = "rmse_xy_mm", n_boot: int = 5000, seed: int = 0) -> dict:
    """'Particularly under X' test. Relative gain rho = 1 - metric(A)/metric(B) (A = candidate, B = reference) is computed in
    condition 1 (e.g. occlusion) and condition 0 (e.g. clean) on the SAME resampled episodes; returns rho1, rho0 and the
    contrast rho1 - rho0 with a 95% CI and one-sided p = P*(rho1 - rho0 <= 0). Needs the same episodes in both conditions."""
    assert A1.shape == B1.shape == A0.shape == B0.shape
    idx = np.random.default_rng(seed).integers(0, len(A1), (n_boot, len(A1)))
    rho = lambda A, B: 1.0 - _boot_metric(A, idx, name) / _boot_metric(B, idx, name)
    d = rho(A1, B1) - rho(A0, B0)
    r1, r0 = 1.0 - metric(A1, name) / metric(B1, name), 1.0 - metric(A0, name) / metric(B0, name)
    return {"rho1": float(r1), "rho0": float(r0), "contrast": float(r1 - r0), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "p_greater": float((np.sum(d <= 0) + 1) / (n_boot + 1))}

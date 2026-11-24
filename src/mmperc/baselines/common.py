"""Shared types for the classical baselines.

State x = [px, py, pz, phi] with phi = wrap(4 * yaw) (symmetric cube). An `Estimate` carries a mean
and a covariance in the tangent space (error e = [p - p_hat; wrap(phi - phi_hat)]).
Frames that an estimator cannot say anything about are filled with the *prior* (train mean and a
wide covariance) and flagged `informative = False`; that way every metric is defined everywhere and
nothing is silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHI_PRIOR_VAR = np.pi ** 2 / 3.0           # variance of a uniform angle on [-pi, pi)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Estimate:
    mean: np.ndarray          # (N, 4)
    cov: np.ndarray           # (N, 4, 4)
    informative: np.ndarray   # (N,) bool

    def __len__(self):
        return len(self.mean)


@dataclass
class Prior:
    mean_p: np.ndarray
    cov_p: np.ndarray

    @staticmethod
    def fit(train: dict) -> "Prior":
        return Prior(train["cube_pos"].mean(0), np.cov(train["cube_pos"].T))

    def table(self, cube_half: float, sd_z: float = 0.004) -> "Prior":
        """Prior for a cube resting on the table: height known to a few mm, xy as before."""
        c = self.cov_p.copy()
        c[2, :] = 0.0; c[:, 2] = 0.0; c[2, 2] = sd_z ** 2
        m = self.mean_p.copy(); m[2] = cube_half
        return Prior(m, c)

    def estimate(self, n: int) -> Estimate:
        mean = np.zeros((n, 4)); mean[:, :3] = self.mean_p
        cov = np.zeros((n, 4, 4)); cov[:, :3, :3] = self.cov_p; cov[:, 3, 3] = PHI_PRIOR_VAR
        return Estimate(mean, cov, np.zeros(n, bool))


def block_cov(cov_p: np.ndarray, var_phi: np.ndarray | float) -> np.ndarray:
    n = len(cov_p)
    out = np.zeros((n, 4, 4))
    out[:, :3, :3] = cov_p
    out[:, 3, 3] = var_phi
    return out

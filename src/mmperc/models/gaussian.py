"""Gaussian output head with a full positive-definite covariance, and the NLL / MSE losses.

State y = [x, y, z, phi]; the error lives in the tangent space: e = [p - p_hat; wrap(phi - phi_hat)].
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

LOG2PI = math.log(2.0 * math.pi)
TRIL = torch.tril_indices(4, 4, offset=-1)         # 6 strictly-lower entries


def wrap_t(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi


def tangent_error(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.cat([target[:, :3] - mean[:, :3], wrap_t(target[:, 3:4] - mean[:, 3:4])], 1)


class GaussianHead(nn.Module):
    """features -> (mean (B,4), L (B,4,4) lower-triangular with positive diagonal), physical units."""

    def __init__(self, in_dim: int, pos_mean, pos_std, hidden: int = 128, eps: float = 1e-3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 3 + 2 + 4 + 6))
        self.register_buffer("pos_mean", torch.as_tensor(pos_mean, dtype=torch.float32))
        self.register_buffer("scale", torch.cat([torch.as_tensor(pos_std, dtype=torch.float32), torch.ones(1)]))
        self.eps = eps
        # start with unit-ish normalised covariance and a centred mean
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].weight.data.mul_(0.1)

    def forward(self, h: torch.Tensor):
        with torch.autocast(device_type=h.device.type, enabled=False):     # covariance maths always in fp32
            o = self.net(h.float())
            p = self.pos_mean + self.scale[:3] * o[:, :3]
            phi = torch.atan2(o[:, 3] + 1e-6, o[:, 4] + 1.0)               # +1 on cos: starts near phi = 0, regular at init
            mean = torch.cat([p, phi[:, None]], 1)
            diag = F.softplus(o[:, 5:9]) + self.eps
            L = torch.diag_embed(diag)
            L = self._fill(L, o[:, 9:15])
            L = self.scale[None, :, None] * L                              # physical units: S L
        return mean, L

    @staticmethod
    def _fill(L, off):
        L = L.clone()
        L[:, TRIL[0], TRIL[1]] = off
        return L


def covariance(L: torch.Tensor) -> torch.Tensor:
    return L @ L.transpose(1, 2)


def gaussian_nll(mean, L, target, beta: float = 0.0, reduce: bool = True):
    """Heteroscedastic Gaussian NLL (includes the 2 log 2 pi constant). beta > 0 -> beta-NLL weighting."""
    e = tangent_error(mean, target)
    z = torch.linalg.solve_triangular(L, e[..., None], upper=False)[..., 0]
    logdiag = torch.log(torch.diagonal(L, dim1=1, dim2=2))
    nll = 0.5 * (z ** 2).sum(1) + logdiag.sum(1) + 2 * LOG2PI
    if beta > 0:
        w = torch.exp(beta * 2.0 * logdiag.mean(1)).detach()               # det(Sigma)^(beta/4), no gradient
        nll = w * nll
    return nll.mean() if reduce else nll


def mse_warmup(mean, target, scale):
    """MSE on the tangent error normalised by the training std (warm-up phase)."""
    e = tangent_error(mean, target) / scale
    return (e ** 2).mean()

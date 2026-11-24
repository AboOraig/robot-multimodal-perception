"""Modality encoders. M5: vision (spatial-softmax CNN) and proprioception (MLP)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _block(cin, cout, k, s, groups=8):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.GroupNorm(groups, cout), nn.SiLU())


class SpatialSoftmax(nn.Module):
    """K feature maps -> K expected (x, y) coordinates in [-1, 1] (learned temperature)."""

    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        b, k, h, w = f.shape
        a = torch.softmax(f.float().flatten(2) * torch.exp(self.log_temp), dim=-1).view(b, k, h, w)
        ys = torch.linspace(-1, 1, h, device=f.device)
        xs = torch.linspace(-1, 1, w, device=f.device)
        return torch.cat([(a.sum(2) * xs).sum(-1), (a.sum(3) * ys).sum(-1)], 1)      # (B, 2K)


class VisionEncoder(nn.Module):
    """96x96 RGB uint8 (B,H,W,3) -> embedding. Keypoints (position cues) + global pooled features (size / appearance cues)."""

    def __init__(self, width: int = 32, keypoints: int = 32, embed: int = 128):
        super().__init__()
        w = width
        self.trunk = nn.Sequential(_block(3, w, 5, 2), _block(w, w, 3, 1), _block(w, 2 * w, 3, 2),
                                   _block(2 * w, 2 * w, 3, 1), _block(2 * w, 2 * w, 3, 1))
        self.kp = nn.Conv2d(2 * w, keypoints, 1)
        self.ss = SpatialSoftmax()
        self.mlp = nn.Sequential(nn.Linear(2 * keypoints + 2 * w, 256), nn.SiLU(), nn.Linear(256, embed), nn.SiLU())
        self.out_dim = embed

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        f = self.trunk(x)
        return self.mlp(torch.cat([self.ss(self.kp(f)), f.mean((2, 3)).float()], 1))


class ProprioEncoder(nn.Module):
    """Feature vector (+ availability mask) -> embedding. Features are standardised with TRAIN statistics; missing -> 0."""

    def __init__(self, in_dim: int, mu, sd, hidden: int = 256, layers: int = 3, embed: int = 128):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        dims = [in_dim + 1] + [hidden] * (layers - 1) + [embed]
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.SiLU()]
        self.mlp = nn.Sequential(*mods)
        self.out_dim = embed

    def forward(self, feat: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        okf = ok.float()[:, None]
        x = ((feat.float() - self.mu) / self.sd).clamp(-10.0, 10.0) * okf    # clamp: robust to out-of-range / zero-filled inputs
        return self.mlp(torch.cat([x, okf], 1))


class ContactEncoder(ProprioEncoder):
    """Wrench + hand-frame gravity + TCP pose etc. (conditioned on proprioception) -> embedding. Same maths as ProprioEncoder."""

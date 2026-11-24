"""Fusion baselines F0-F2: image + proprioception + contact -> Gaussian over [x, y, z, phi].

mode = concat : naive. Missing data arrives zero-filled (as the sensors deliver it); no reliability signals anywhere.
mode = learned: encoders and the fusion MLP are told which modalities are available (explicit masks).
mode = gated  : learned + a per-modality reliability gate sigma(g([e_m, mask_m])) multiplying each embedding.
All variants end in the same full-covariance Gaussian head, so they are directly comparable under NLL / calibration.
"""
from __future__ import annotations

import torch
from torch import nn

from mmperc.models.encoders import ContactEncoder, ProprioEncoder, VisionEncoder
from mmperc.models.gaussian import GaussianHead

MODES = ("concat", "learned", "gated")


class FusionNet(nn.Module):
    kind = "fusion"

    def __init__(self, stats: dict, cfg: dict):
        super().__init__()
        f, v, p, c, h = cfg["fusion"], cfg["vision"], cfg["proprio"], cfg["contact"], cfg["head"]
        self.mode = f["mode"]
        if self.mode not in MODES:
            raise ValueError(f"fusion mode must be one of {MODES}")
        E = f["embed"]
        self.ev = VisionEncoder(v["width"], v["keypoints"], E)
        self.ep = ProprioEncoder(len(stats["feat_mu"]), stats["feat_mu"], stats["feat_sd"], p["hidden"], p["layers"], E)
        self.ec = ContactEncoder(len(stats["cfeat_mu"]), stats["cfeat_mu"], stats["cfeat_sd"], c["hidden"], c["layers"], E)
        n_mask = 0 if self.mode == "concat" else 3
        if self.mode == "gated":
            self.gates = nn.ModuleList([nn.Linear(E + 1, 1) for _ in range(3)])
            for g in self.gates:
                nn.init.zeros_(g.weight); nn.init.constant_(g.bias, 2.0)        # start (almost) open
        self.mlp = nn.Sequential(nn.Linear(3 * E + n_mask, f["hidden"]), nn.SiLU(), nn.Linear(f["hidden"], 128), nn.SiLU())
        self.head = GaussianHead(128, stats["pos_mean"], stats["pos_std"], h["hidden"], h["eps"])

    def forward(self, b: dict):
        ev = self.ev(b["image"])
        if self.mode == "concat":
            one = torch.ones_like(b["ok"])
            hcat = torch.cat([ev, self.ep(b["feat"], one), self.ec(b["cfeat"], one)], 1)
        else:
            vm, pm, cm = (b[k].float()[:, None] for k in ("vmask", "ok", "cok"))
            es = [ev * vm, self.ep(b["feat"], b["ok"]), self.ec(b["cfeat"], b["cok"])]
            ms = [vm, pm, cm]
            if self.mode == "gated":
                es = [e * torch.sigmoid(g(torch.cat([e, m], 1))) for e, m, g in zip(es, ms, self.gates)]
            hcat = torch.cat(es + ms, 1)
        return self.head(self.mlp(hcat))


# ----------------------------------------------------------------------------------------------------------------------
# Uncertainty-aware fusion: one calibrated Gaussian EXPERT per sensing route, combined in information form.
# ----------------------------------------------------------------------------------------------------------------------
import math  # noqa: E402

from mmperc.models.gaussian import gaussian_nll, mse_warmup, tangent_error, wrap_t  # noqa: E402

UA_MODES = ("poe", "tpoe", "ci")
EXPERTS = ("vision", "proprio", "contact")           # contact expert is CONDITIONED on proprioception (input = proprio + contact features)


def fuse_gaussians(mu, L, w, mask, prior_mean, prior_prec):
    """Information-form fusion of M Gaussian experts and a broad prior over y = [x, y, z, phi].

        Lam = Lam0 + sum_m a_m Lam_m ,   Lam_m = (L_m L_m^T)^-1 ,   a_m = w_m * mask_m
        mean = Lam^-1 (Lam0 mu0 + sum_m a_m Lam_m mu_m)

    w_m = 1 for all m           -> product of experts (assumes conditionally independent experts);
    w_m in (0, 1) (tempering)   -> tempered PoE: tolerates correlated experts (each expert counted at most once);
    sum_m w_m = 1               -> covariance intersection: consistent for ANY unknown correlation between experts.
    A missing expert (mask 0) has zero precision, so it is excluded exactly; with every expert missing the prior remains.
    phi is an angle: expert phis are unwrapped around the precision-weighted circular mean before averaging.
    Shapes: mu (B,M,4), L (B,M,4,4) lower Cholesky of each expert covariance, w/mask (B,M), prior_mean (3,), prior_prec (4,).
    Returns (mean (B,4), L_fused (B,4,4)) in float32.
    """
    mu, L = mu.float(), L.float()
    B, M, _ = mu.shape
    eye = torch.eye(4, device=mu.device, dtype=mu.dtype)
    lam = torch.cholesky_solve(eye.expand(B, M, 4, 4).contiguous(), L)                 # (B,M,4,4) expert precisions
    a = (w.float() * mask.float())[..., None, None]
    lam = a * lam
    p0 = prior_prec.float()
    lam_phi = lam[..., 3, 3]
    s = (lam_phi * torch.sin(mu[..., 3])).sum(1)
    c = (lam_phi * torch.cos(mu[..., 3])).sum(1) + p0[3]                               # prior phi mean 0 with precision p0[3]
    ref = torch.atan2(s, c)
    phi_u = ref[:, None] + wrap_t(mu[..., 3] - ref[:, None])
    mu_u = torch.cat([mu[..., :3], phi_u[..., None]], -1)
    mu0 = torch.cat([prior_mean.float().expand(B, 3), (ref + wrap_t(-ref))[:, None]], 1)
    lam0 = torch.diag(p0)
    lam_tot = lam0 + lam.sum(1)
    eta = (lam0 @ mu0[..., None])[..., 0] + (lam @ mu_u[..., None]).sum(1)[..., 0]
    Lc = torch.linalg.cholesky(0.5 * (lam_tot + lam_tot.transpose(1, 2)))
    mean = torch.cholesky_solve(eta[..., None], Lc)[..., 0]
    cov = torch.cholesky_solve(eye.expand(B, 4, 4).contiguous(), Lc)
    L_f = torch.linalg.cholesky(0.5 * (cov + cov.transpose(1, 2)))
    mean = torch.cat([mean[:, :3], wrap_t(mean[:, 3:4])], 1)
    return mean, L_f


class UAFusionNet(nn.Module):
    """Three Gaussian experts (vision / proprioception / proprioception-conditioned contact) + fusion rule `mode`.

    Experts are supervised individually (aux loss, weight fusion.aux_weight) as well as through the fused NLL, so each one
    is a calibrated unimodal estimator. The fusion weights (tpoe, ci) come from a small network that sees the experts'
    log-uncertainties, availability and pairwise disagreement (conflict detection, e.g. a biased proprioception)."""

    kind = "fusion"

    def __init__(self, stats: dict, cfg: dict):
        super().__init__()
        f, v, p, c, h = cfg["fusion"], cfg["vision"], cfg["proprio"], cfg["contact"], cfg["head"]
        self.mode = f["mode"]
        if self.mode not in UA_MODES:
            raise ValueError(f"UA fusion mode must be one of {UA_MODES}")
        self.aux_weight = float(f.get("aux_weight", 0.5))
        E, hh = f["embed"], h["hidden"]
        nf, nc = len(stats["feat_mu"]), len(stats["cfeat_mu"])
        self.ev = VisionEncoder(v["width"], v["keypoints"], E)
        self.ep = ProprioEncoder(nf, stats["feat_mu"], stats["feat_sd"], p["hidden"], p["layers"], E)
        self.ec = ProprioEncoder(nf + nc, list(stats["feat_mu"]) + list(stats["cfeat_mu"]),
                                 list(stats["feat_sd"]) + list(stats["cfeat_sd"]), c["hidden"] * 2, c["layers"], E)
        self.heads = nn.ModuleList([GaussianHead(E, stats["pos_mean"], stats["pos_std"], hh, h["eps"]) for _ in range(3)])
        self.register_buffer("prior_mean", torch.as_tensor(stats["pos_mean"], dtype=torch.float32))
        self.register_buffer("prior_prec", torch.cat([1.0 / torch.as_tensor(stats["pos_std"], dtype=torch.float32) ** 2,
                                                      torch.tensor([3.0 / math.pi ** 2])]))
        if self.mode != "poe":
            self.wnet = nn.Sequential(nn.Linear(3 * 5 + 3, f.get("weight_hidden", 64)), nn.SiLU(), nn.Linear(f.get("weight_hidden", 64), 3))
            nn.init.zeros_(self.wnet[-1].weight)
            nn.init.constant_(self.wnet[-1].bias, 3.0 if self.mode == "tpoe" else 0.0)   # tpoe starts ~PoE, ci starts uniform
        self.last: dict = {}

    @staticmethod
    def _disagreement(mu, L, mask):
        """log(1 + d^2_ij) for the 3 expert pairs; d^2 = e^T (S_i + S_j)^-1 e. Zero when either expert is missing."""
        S = L @ L.transpose(-1, -2)
        out = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            e = tangent_error(mu[:, i], mu[:, j])
            d2 = (e[:, None, :] @ torch.linalg.solve(S[:, i] + S[:, j] + 1e-9 * torch.eye(4, device=mu.device), e[..., None]))[:, 0, 0]
            out.append(torch.log1p(d2.clamp(min=0.0)) * (mask[:, i] & mask[:, j]).float())
        return torch.stack(out, 1)

    def forward(self, b: dict):
        vm, pm, cm = b["vmask"], b["ok"], b["ok"] & b["cok"]                       # contact needs proprioception AND the wrench
        embs = [self.ev(b["image"]) * vm.float()[:, None], self.ep(b["feat"], pm),
                self.ec(torch.cat([b["feat"], b["cfeat"]], 1), cm)]
        outs = [hd(e) for hd, e in zip(self.heads, embs)]
        mu = torch.stack([o[0] for o in outs], 1)
        L = torch.stack([o[1] for o in outs], 1)
        mask = torch.stack([vm, pm, cm], 1)
        with torch.autocast(device_type=mu.device.type, enabled=False):
            if self.mode == "poe":
                w = torch.ones(mu.shape[:2], device=mu.device)
            else:
                logd = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).detach()     # (B,3,4)
                feats = torch.cat([logd, mask[..., None].float()], -1).flatten(1)  # (B,15)
                z = self.wnet(torch.cat([feats, self._disagreement(mu.detach(), L.detach(), mask)], 1))
                if self.mode == "tpoe":
                    w = torch.sigmoid(z)
                else:                                                              # ci: convex weights over PRESENT experts
                    w = torch.softmax(z.masked_fill(~mask, -1e9), dim=1)
            mean, L_f = fuse_gaussians(mu, L, w, mask, self.prior_mean, self.prior_prec)
        self.last = {"mu": mu.detach(), "L": L.detach(), "mask": mask.detach(), "w": w.detach()}
        self._experts = (mu, L, mask)
        return mean, L_f

    def aux_loss(self, y, use_nll: bool, scale, cov_weight: float = 0.1):
        """Per-expert supervision on frames where the expert is available (sum over experts of the masked mean)."""
        mu, L, mask = self._experts
        total = y.new_zeros(())
        for k in range(mu.shape[1]):
            n = mask[:, k].float().sum().clamp(min=1.0)
            if use_nll:
                per = gaussian_nll(mu[:, k], L[:, k], y, reduce=False)
            else:
                e = tangent_error(mu[:, k], y) / scale
                per = (e ** 2).mean(1) + cov_weight * gaussian_nll(mu[:, k].detach(), L[:, k], y, reduce=False)
            total = total + (per * mask[:, k].float()).sum() / n
        return total

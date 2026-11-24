"""Unimodal learned baselines: B2 (vision) and B4 (proprioception), each with a Gaussian head."""
from __future__ import annotations

import torch
from torch import nn

from mmperc.models.encoders import ProprioEncoder, VisionEncoder
from mmperc.models.gaussian import GaussianHead


class VisionNet(nn.Module):
    kind = "vision"

    def __init__(self, stats: dict, cfg: dict):
        super().__init__()
        v, h = cfg["vision"], cfg["head"]
        self.enc = VisionEncoder(v["width"], v["keypoints"], v["embed"])
        self.head = GaussianHead(self.enc.out_dim, stats["pos_mean"], stats["pos_std"], h["hidden"], h["eps"])

    def forward(self, batch: dict):
        return self.head(self.enc(batch["image"]))


class ProprioNet(nn.Module):
    kind = "proprio"

    def __init__(self, stats: dict, cfg: dict):
        super().__init__()
        p, h = cfg["proprio"], cfg["head"]
        self.enc = ProprioEncoder(len(stats["feat_mu"]), stats["feat_mu"], stats["feat_sd"], p["hidden"], p["layers"], p["embed"])
        self.head = GaussianHead(self.enc.out_dim, stats["pos_mean"], stats["pos_std"], h["hidden"], h["eps"])

    def forward(self, batch: dict):
        return self.head(self.enc(batch["feat"], batch["ok"]))


from mmperc.models.fusion import UA_MODES, FusionNet, UAFusionNet  # noqa: E402


def _fusion(stats, cfg):
    return (UAFusionNet if cfg["fusion"]["mode"] in UA_MODES else FusionNet)(stats, cfg)


MODELS = {"vision": VisionNet, "proprio": ProprioNet, "fusion": _fusion}


def build_model(kind: str, stats: dict, cfg: dict) -> nn.Module:
    return MODELS[kind](stats, cfg)

"""Deterministic seeding utilities.

`make_rng` derives an independent generator from a tuple of integer keys, e.g.
(master_seed, episode_id, frame_id, degradation_id). This is how degradations
will be made identical across models and runs, regardless of data-loader order.
"""
from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for a training run."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_rng(*keys: int) -> np.random.Generator:
    """Return a Generator determined only by the (non-negative integer) keys."""
    return np.random.default_rng(np.random.SeedSequence(list(keys)))
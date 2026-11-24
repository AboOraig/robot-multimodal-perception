"""Arm-only control: a transparent cube changes the pixels but not the physics."""
import copy

import numpy as np

from mmperc.common.config import load_config
from mmperc.simulation.episode import run_episode


def _red(im):
    return int(((im[..., 0] > 150) & (im[..., 1] < 90) & (im[..., 2] < 90)).sum(axis=(1, 2)).max())


def test_transparent_cube_is_invisible_but_trajectory_is_identical():
    sim = load_config("configs/simulation.yaml")
    inv = copy.deepcopy(sim)
    inv["cube"]["rgba"] = list(sim["cube"]["rgba"][:3]) + [0.0]
    a, b = run_episode(sim, 7, render=True), run_episode(inv, 7, render=True)
    assert _red(a["image"]) > 10 and _red(b["image"]) == 0
    for k in ("q", "cube_pos", "wrench", "cube_phi"):
        np.testing.assert_array_equal(a[k], b[k])

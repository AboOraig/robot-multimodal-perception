import numpy as np

from mmperc.common.config import load_config
from mmperc.common.seeding import make_rng


def test_make_rng_is_deterministic():
    a = make_rng(0, 12, 34).normal(size=8)
    b = make_rng(0, 12, 34).normal(size=8)
    assert np.array_equal(a, b)


def test_make_rng_keys_are_independent():
    a = make_rng(0, 12, 34).normal(size=8)
    assert not np.array_equal(a, make_rng(0, 12, 35).normal(size=8))
    assert not np.array_equal(a, make_rng(0, 13, 34).normal(size=8))


def test_load_config_relative_to_repo_root():
    cfg = load_config("configs/simulation.yaml")
    assert cfg["image_size"] == 96
    assert cfg["physics"]["timestep"] == 0.002

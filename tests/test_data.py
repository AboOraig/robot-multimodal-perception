"""Data pipeline tests: splits, storage round trip, generation, resume, loader."""
import copy
import json

import numpy as np
import pytest

from mmperc.common.config import load_config
from mmperc.data.dataset import SEGMENT_NAMES, STAGE_NAMES, holding_gate, load_manifest, load_split
from mmperc.data.generate import generate_dataset
from mmperc.data.splits import _yaw_dist_deg, config_hash, make_splits
from mmperc.data.storage import load_episode, read_meta, save_episode
from mmperc.simulation.episode import sample_cube_pose


@pytest.fixture(scope="module")
def cfgs():
    sim, ds = load_config("configs/simulation.yaml"), load_config("configs/dataset.yaml")
    ds = copy.deepcopy(ds)
    ds["splits"].update(n_train=6, n_val=3, n_test=3, n_test_ood=3)
    ds["store_images"] = False
    return sim, ds


def test_splits_disjoint_deterministic_and_ood(cfgs):
    sim, ds = cfgs
    a, b = make_splits(sim, ds), make_splits(sim, ds)
    assert a == b
    ids = a["ids"]
    allids = [e for v in ids.values() for e in v]
    assert len(allids) == len(set(allids)) == 15
    xr, yr = ds["splits"]["ood_region"]["x"], ds["splits"]["ood_region"]["y"]
    for name, v in ids.items():
        for e in v:
            pos = sample_cube_pose(sim, ds["seed"], e)[1]
            inside = xr[0] <= pos[0] <= xr[1] and yr[0] <= pos[1] <= yr[1]
            assert inside == (name == "test_ood"), (name, e, pos)


def test_yaw_distance_respects_cube_symmetry():
    d = _yaw_dist_deg(0.0, np.radians([0.0, 90.0, 180.0, 45.0, 89.0, -1.0]))
    assert np.allclose(d, [0, 0, 0, 45, 1, 1])


def test_storage_round_trip_with_images(tmp_path):
    from mmperc.data.storage import ARRAY_KEYS
    T = 5
    ep = {k: np.random.default_rng(0).normal(size=(T, 3)) for k in ARRAY_KEYS}
    ep["image"] = np.random.default_rng(1).integers(0, 255, (T, 96, 96, 3), dtype=np.uint8)
    ep.update(meta={"success": True, "lifted_m": 0.1}, segment_names=["a"], segment_bounds=np.array([0., 1.]),
              phase_names=("x",))
    p = tmp_path / "e.h5"
    save_episode(p, ep, "abc", store_images=True)
    assert not list(tmp_path.glob("*.tmp"))
    back = load_episode(p)
    assert np.array_equal(back["image"], ep["image"]) and np.array_equal(back["q"], ep["q"])
    assert read_meta(p)["config_hash"] == "abc"


def test_generate_resume_and_load(cfgs, tmp_path):
    sim, ds = cfgs
    m = generate_dataset(sim, ds, tmp_path, workers=1)
    assert m["n_episodes"] == 15 and m["last_run_status_counts"] == {"ok": 15}
    m2 = generate_dataset(sim, ds, tmp_path, workers=1)          # resume: nothing is recomputed
    assert m2["last_run_status_counts"] == {"skipped": 15}

    d = load_split(tmp_path, "val", load_images=False)
    n = len(d["t"])
    assert n == 3 * 166 and d["target"].shape == (n, 4) and d["wrench"].shape == (n, 6)
    assert set(np.unique(d["stage"])) == set(range(len(STAGE_NAMES)))
    assert len(set(d["ep_id"])) == 3 and set(d["episode"]) == {0, 1, 2}
    # observable holding gate agrees with the ground-truth contact label (generalises M2 finding)
    truth = d["n_finger_contacts"] == 2
    assert (d["holding_gate"] & truth).sum() / truth.sum() > 0.99

    bad = copy.deepcopy(ds)
    bad["seed"] = 999
    with pytest.raises(RuntimeError):
        generate_dataset(sim, bad, tmp_path, workers=1)           # different config, same folder


def test_segment_and_stage_tables_consistent():
    from mmperc.data.dataset import SEGMENT_TO_STAGE
    assert len(SEGMENT_NAMES) == len(SEGMENT_TO_STAGE) == 11
    assert holding_gate(np.array([[0.02, 0.02], [0.04, 0.04]])).tolist() == [True, False]

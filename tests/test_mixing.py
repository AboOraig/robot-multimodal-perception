"""Mixed-degradation training: grammar, held-out combinations, reproducibility, warm-up stability."""
import copy

import numpy as np
import pytest

from mmperc.baselines.kinematics import HandKinematics
from mmperc.common.config import load_config
from mmperc.common.seeding import make_rng
from mmperc.data.dataset import load_split
from mmperc.data.generate import generate_dataset
from mmperc.sensors.degradation import FAMILIES, Degrader
from mmperc.training.features import make_inputs
from mmperc.training.mixing import DEFAULT_FORBIDDEN, Mixer, is_forbidden, relevant_families, sample_train_scenario


@pytest.fixture(scope="module")
def sim():
    return load_config("configs/simulation.yaml")


@pytest.fixture(scope="module")
def tiny(sim, tmp_path_factory):
    ds = copy.deepcopy(load_config("configs/dataset.yaml"))
    ds["splits"].update(n_train=6, n_val=3, n_test=2, n_test_ood=1)
    ds["store_images"] = False
    out = tmp_path_factory.mktemp("tiny_mix")
    generate_dataset(sim, ds, out, workers=1)
    return out


def test_forbidden_pairs_never_sampled_but_each_family_and_other_pairs_are():
    seen, pairs = set(), set()
    for i in range(4000):
        sc = sample_train_scenario(make_rng(0, 0, i), FAMILIES, DEFAULT_FORBIDDEN, p_family=0.4)
        assert not is_forbidden(sc, DEFAULT_FORBIDDEN)
        fams = [f for f, _ in sc.levels]
        seen.update(fams)
        pairs.update((a, b) for a in fams for b in fams if a < b)
    assert seen == set(FAMILIES)                                      # every family is trained on
    assert ("proprio_noise", "vision_occlusion") not in pairs and ("proprio_bias", "vision_occlusion") not in pairs
    assert ("proprio_dropout", "vision_occlusion") in pairs           # other combinations ARE seen


def test_relevant_families_follow_the_model_modality():
    assert all(f.startswith("vision") for f in relevant_families("vision"))
    assert all(f.startswith("proprio") for f in relevant_families("proprio"))


def test_mixer_is_reproducible_changes_across_blocks_and_never_touches_labels(tiny, sim):
    cfg = load_config("configs/training.yaml")
    cfg["mixing"].update(enabled=True, refresh_every=2)
    hk = HandKinematics(sim)
    tr = load_split(tiny, "train", load_images=False)
    mk = lambda k, d, c: make_inputs(k, d, c, hk)
    dg = Degrader(load_config("configs/degradation.yaml"), sim)
    a, b = Mixer("proprio", cfg, dg, mk, 3), Mixer("proprio", cfg, dg, mk, 3)
    x0, x0b = a.train_inputs(tr, 0), b.train_inputs(tr, 0)
    np.testing.assert_array_equal(x0["feat"], x0b["feat"])            # same (seed, epoch) -> identical degraded copy
    np.testing.assert_array_equal(a.train_inputs(tr, 1)["feat"], x0["feat"])   # refresh_every = 2 -> epoch 1 reuses epoch 0
    assert not np.array_equal(a.train_inputs(tr, 2)["feat"], x0["feat"])      # new block -> new degradation
    clean = make_inputs("proprio", tr, cfg, hk)
    assert not np.array_equal(x0["feat"], clean["feat"]) or not x0["ok"].all()  # something was actually degraded
    assert len(x0["feat"]) == len(tr["target"])


def test_mixed_val_inputs_are_fixed(tiny, sim):
    cfg = load_config("configs/training.yaml"); cfg["mixing"]["enabled"] = True
    hk = HandKinematics(sim)
    va = load_split(tiny, "val", load_images=False)
    m = Mixer("proprio", cfg, Degrader(load_config("configs/degradation.yaml"), sim), lambda k, d, c: make_inputs(k, d, c, hk), 0)
    np.testing.assert_array_equal(m.val_inputs(va)["feat"], m.val_inputs(va)["feat"])


def test_warmup_trains_the_covariance_head_on_a_detached_mean(tmp_path):
    """Mechanism behind the fix for the loss spike at the MSE->NLL switch seen on real data: without it the covariance
    head is untrained during warm-up (d2 << 1, i.e. wildly under-confident, and free to drift); with it, it already learns."""
    import torch
    from test_models import _stats, _synthetic, _tiny_cfg
    from mmperc.evaluation.metrics import summarize
    from mmperc.training.engine import load_checkpoint, predict, train
    d2 = {}
    for w in (0.0, 0.1):
        cfg = _tiny_cfg(); cfg["train"].update(epochs=10, warmup_epochs=10, warmup_cov_weight=w)   # warm-up only
        data, inp = _synthetic(128)
        train("proprio", cfg, data, data, inp, inp, _stats(data, inp), tmp_path / str(w), seed=1, log=lambda *_: None)
        m, _ = load_checkpoint(tmp_path / str(w) / "last.pt")
        d2[w] = summarize(predict(m, inp, torch.device("cpu")), data["target"].astype(float))["mean_d2"]
    assert d2[0.1] > 2.0 * d2[0.0]

"""M7: contact features, fusion models (concat / learned / gated), mask semantics, mixing for all modalities."""
import copy

import numpy as np
import pytest
import torch

from mmperc.baselines.kinematics import HandKinematics
from mmperc.common.config import load_config
from mmperc.data.dataset import load_split
from mmperc.data.generate import generate_dataset
from mmperc.models.gaussian import covariance, gaussian_nll
from mmperc.models.unimodal import build_model
from mmperc.sensors.degradation import Degrader, Scenario
from mmperc.training.features import contact_features, fit_stats, make_inputs
from mmperc.training.mixing import relevant_families


@pytest.fixture(scope="module")
def sim():
    return load_config("configs/simulation.yaml")


@pytest.fixture(scope="module")
def tiny(sim, tmp_path_factory):
    ds = copy.deepcopy(load_config("configs/dataset.yaml"))
    ds["splits"].update(n_train=6, n_val=3, n_test=2, n_test_ood=1)
    ds["store_images"] = False
    out = tmp_path_factory.mktemp("tiny_fusion")
    generate_dataset(sim, ds, out, workers=1)
    return out


def _cfg(mode):
    cfg = load_config("configs/training.yaml")
    cfg["fusion"]["mode"] = mode
    cfg["vision"].update(width=8, keypoints=8)                # small, CPU-friendly
    return cfg


def _fake_stats(nf=24, nc=13):
    return {"pos_mean": [0.5, 0.0, 0.05], "pos_std": [0.06, 0.12, 0.05], "feat_mu": [0.0] * nf, "feat_sd": [1.0] * nf,
            "cfeat_mu": [0.0] * nc, "cfeat_sd": [1.0] * nc}


def _batch(n=6, nf=24, nc=13, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"image": torch.randint(0, 255, (n, 96, 96, 3), dtype=torch.uint8, generator=g),
            "vmask": torch.ones(n, dtype=torch.bool), "feat": torch.randn(n, nf, generator=g), "ok": torch.ones(n, dtype=torch.bool),
            "cfeat": torch.randn(n, nc, generator=g), "cok": torch.ones(n, dtype=torch.bool)}


def test_contact_features_shape_gravity_and_masks(tiny, sim):
    hk = HandKinematics(sim)
    d = load_split(tiny, "val", load_images=False)
    cf, ok = contact_features(d, hk)
    assert cf.shape == (len(d["q"]), 13) and np.isfinite(cf).all() and ok.all()
    assert np.allclose(np.linalg.norm(cf[:, 6:9], axis=1), 1.0, atol=1e-5)               # unit gravity direction in the hand frame
    dg = Degrader(load_config("configs/degradation.yaml"), sim)
    o = dg.apply(d, Scenario.make(proprio_dropout=4, contact_dropout=4), 3)
    _, ok2 = contact_features(o, hk)
    assert np.array_equal(ok2, o["proprio_mask"] & o["contact_mask"])                    # missing if EITHER sensor is missing
    assert 0 < ok2.mean() < 1


@pytest.mark.parametrize("mode", ["concat", "learned", "gated"])
def test_fusion_net_shapes_positive_definite_and_gradients(mode):
    net = build_model("fusion", _fake_stats(), _cfg(mode))
    b = _batch()
    m, L = net(b)
    S = covariance(L).double()
    assert m.shape == (6, 4) and (torch.linalg.eigvalsh(S) / torch.linalg.eigvalsh(S)[:, -1:] > -1e-9).all()
    gaussian_nll(m, L, m.detach() + 0.01).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_masks_matter_for_learned_and_gated_but_concat_has_no_reliability_input():
    torch.manual_seed(0)
    b = _batch()
    b2 = dict(b); b2["vmask"] = torch.zeros_like(b["vmask"])
    for mode, expect_change in (("concat", False), ("learned", True), ("gated", True)):
        torch.manual_seed(1)
        net = build_model("fusion", _fake_stats(), _cfg(mode)).eval()
        with torch.no_grad():
            m1, _ = net(b); m2, _ = net(b2)
        assert (not torch.allclose(m1, m2)) == expect_change, mode
    # and a dropped vision stream must not leak the (black) image into learned/gated outputs
    net = build_model("fusion", _fake_stats(), _cfg("learned")).eval()
    b3, b4 = dict(b2), dict(b2)
    b4["image"] = torch.zeros_like(b["image"])
    with torch.no_grad():
        assert torch.allclose(net(b3)[0], net(b4)[0], atol=1e-6)


@pytest.mark.parametrize("mode", ["concat", "learned", "gated"])
def test_fusion_overfits_a_tiny_batch_with_the_real_training_recipe(mode):
    """MSE warm-up (+ covariance on a detached mean) then NLL, with gradient clipping, exactly as the trainer does.
    NLL alone from scratch does NOT fit the mean (variance inflates to excuse it) -- that is why the warm-up exists."""
    from mmperc.models.gaussian import mse_warmup, tangent_error
    torch.manual_seed(0)
    net = build_model("fusion", _fake_stats(), _cfg(mode))
    b = _batch(8)
    y = torch.tensor([[0.5, 0.05, 0.03, 0.4]] * 4 + [[0.45, -0.1, 0.12, -0.8]] * 4, dtype=torch.float32)
    scale = torch.tensor([0.06, 0.12, 0.05, 1.0])
    opt = torch.optim.AdamW(net.parameters(), 2e-3)
    for i in range(300):
        m, L = net(b)
        loss = mse_warmup(m, y, scale) + 0.1 * gaussian_nll(m.detach(), L, y) if i < 150 else gaussian_nll(m, L, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    rmse_mm = 1e3 * float(tangent_error(m.detach(), y)[:, :3].pow(2).sum(1).mean().sqrt())
    assert rmse_mm < 15.0 and loss.item() < -8.0


def test_fusion_mixing_uses_all_modalities_and_inputs_have_every_key(tiny, sim):
    assert {f.split("_")[0] for f in relevant_families("fusion")} == {"vision", "proprio", "contact"}
    d = load_split(tiny, "val", load_images=False)
    d["image"] = np.zeros((len(d["q"]), 96, 96, 3), np.uint8)
    cfg = load_config("configs/training.yaml")
    inp = make_inputs("fusion", d, cfg, HandKinematics(sim))
    assert set(inp) == {"image", "vmask", "feat", "ok", "cfeat", "cok"}
    assert len({len(v) for v in inp.values()}) == 1
    assert (inp["feat"][~inp["ok"]] == 0).all() and (inp["cfeat"][~inp["cok"]] == 0).all()   # zero-filled
    st = fit_stats(d, inp)
    assert len(st["feat_mu"]) == inp["feat"].shape[1] and len(st["cfeat_mu"]) == 13


def test_standardisation_is_safe_for_constant_features_and_zero_filled_rows():
    """Regression: a feature that is constant on valid rows (holding gate = 1) had sd ~ 1e-6, so zero-filled rows gave
    inputs of ~1e6 in the unmasked (concat) model, which then diverged."""
    f = np.zeros((50, 3), np.float32); f[:, 0] = 1.0; f[:, 1] = np.linspace(-1, 1, 50); f[:, 2] = 0.3
    st = fit_stats({"target": np.zeros((50, 4), np.float32) + 0.1}, {"feat": f, "ok": np.ones(50, bool)})
    sd = np.array(st["feat_sd"])
    assert (sd >= 1e-3).all() and abs((0.0 - 1.0) / sd[0]) <= 10.0 + 1e-6      # zero-filled gate -> at most ~ -10 sd
    from mmperc.models.encoders import ProprioEncoder
    enc = ProprioEncoder(3, st["feat_mu"], st["feat_sd"], 16, 2, 8)
    out = enc(torch.zeros(4, 3), torch.ones(4, dtype=torch.bool))
    assert torch.isfinite(out).all() and out.abs().max() < 50
    big = enc(torch.full((4, 3), 1e9), torch.ones(4, dtype=torch.bool))          # absurd inputs are clamped
    assert torch.isfinite(big).all()


def test_contact_features_carry_no_tcp_xy_and_ablation_makes_a_dead_sensor(tiny, sim):
    """Regression (M7): TCP x,y in the contact stream predicted the cube xy with R^2 = 0.998 (scripted hand hovers over the cube)."""
    hk = HandKinematics(sim)
    d = load_split(tiny, "val", load_images=False)
    cf, _ = contact_features(d, hk)
    P, _ = hk.fk(d["q"])
    for j in range(cf.shape[1]):
        for k in (0, 1):
            assert not np.allclose(cf[:, j], P[:, k], atol=1e-5), f"column {j} equals TCP {'xy'[k]}"
    assert np.allclose(cf[:, 9], P[:, 2], atol=1e-5)                 # only the height is kept
    from mmperc.training.features import ablate_inputs
    d["image"] = np.ones((len(d["q"]), 96, 96, 3), np.uint8)
    inp = make_inputs("fusion", d, load_config("configs/training.yaml"), hk)
    ab = ablate_inputs(inp, ["vision", "contact"])
    assert not ab["vmask"].any() and not ab["cok"].any() and (ab["image"] == 0).all() and (ab["cfeat"] == 0).all()
    assert np.array_equal(ab["feat"], inp["feat"]) and ab["ok"].any()      # proprioception untouched

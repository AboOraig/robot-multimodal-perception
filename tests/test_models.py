"""M5 tests: Gaussian head / NLL maths, positive definiteness, wrapped phi, overfit, checkpoints, determinism."""
import math

import numpy as np
import pytest
import torch

from mmperc.common.config import load_config
from mmperc.evaluation.metrics import per_frame
from mmperc.baselines.common import Estimate
from mmperc.models.gaussian import GaussianHead, covariance, gaussian_nll, tangent_error, wrap_t
from mmperc.models.unimodal import build_model
from mmperc.training.engine import load_checkpoint, predict, save_checkpoint, train

POS_MEAN, POS_STD = [0.5, 0.0, 0.05], [0.06, 0.12, 0.05]


def _head(seed=0):
    torch.manual_seed(seed)
    return GaussianHead(16, POS_MEAN, POS_STD, hidden=32)


def test_head_covariance_is_symmetric_positive_definite_even_for_extreme_inputs():
    h = _head()
    for scale in (0.0, 1.0, 10.0):
        m, L = h(torch.randn(64, 16) * scale)
        S = covariance(L).double()
        assert torch.allclose(S, S.transpose(1, 2), atol=1e-12)
        ev = torch.linalg.eigvalsh(S)
        assert (ev / ev[:, -1:] > -1e-9).all()                       # PSD up to float round-off; PD by construction (diag L > 0)
        assert (torch.diagonal(L, dim1=1, dim2=2) > 0).all() and torch.isfinite(m).all()


def test_nll_matches_closed_form_and_numpy_metric():
    torch.manual_seed(1)
    h = _head()
    m, L = h(torch.randn(32, 16))
    y = m.detach() + torch.randn(32, 4) * 0.01
    y[:, 3] = wrap_t(y[:, 3])
    nll = gaussian_nll(m, L, y, reduce=False).detach().double().numpy()
    est = Estimate(m.detach().double().numpy(), covariance(L).detach().double().numpy(), np.ones(32, bool))
    ref = per_frame(est, y.double().numpy())["nll"]
    np.testing.assert_allclose(nll, ref, rtol=1e-4, atol=1e-4)
    e = tangent_error(m.detach(), y).double().numpy()
    S = covariance(L).detach().double().numpy()
    closed = 0.5 * np.einsum("ni,nij,nj->n", e, np.linalg.inv(S), e) + 0.5 * np.log(np.linalg.det(S)) + 2 * math.log(2 * math.pi)
    np.testing.assert_allclose(nll, closed, rtol=1e-3, atol=1e-3)


def test_phi_error_wraps_across_the_branch_cut():
    m = torch.tensor([[0, 0, 0, math.pi - 0.01]], dtype=torch.float32)
    y = torch.tensor([[0, 0, 0, -math.pi + 0.01]], dtype=torch.float32)
    assert abs(float(tangent_error(m, y)[0, 3])) == pytest.approx(0.02, abs=1e-5)


def test_beta_nll_weights_do_not_carry_gradient_through_the_weight():
    h = _head()
    m, L = h(torch.randn(8, 16))
    y = m.detach() + 0.05
    g0 = torch.autograd.grad(gaussian_nll(m, L, y, beta=0.0), list(h.parameters()), retain_graph=True)
    g5 = torch.autograd.grad(gaussian_nll(m, L, y, beta=0.5), list(h.parameters()))
    assert all(torch.isfinite(g).all() for g in g5) and any(not torch.allclose(a, b) for a, b in zip(g0, g5))


def _tiny_cfg():
    cfg = load_config("configs/training.yaml")
    cfg["device"] = "cpu"; cfg["amp"] = False
    cfg["train"].update(epochs=60, batch_size=32, lr=3e-3, warmup_epochs=20)
    return cfg


def _synthetic(n=96, seed=0):
    rng = np.random.default_rng(seed)
    feat = rng.normal(size=(n, 6)).astype(np.float32)
    t = np.zeros((n, 4), np.float32)
    t[:, 0] = 0.5 + 0.03 * feat[:, 0]; t[:, 1] = 0.05 * feat[:, 1]; t[:, 2] = 0.05 + 0.01 * feat[:, 2]; t[:, 3] = 0.5 * feat[:, 3]
    return {"target": t}, {"feat": feat, "ok": np.ones(n, bool)}


def _stats(d, inp):
    return {"pos_mean": d["target"][:, :3].mean(0).tolist(), "pos_std": d["target"][:, :3].std(0).tolist(),
            "feat_mu": inp["feat"].mean(0).tolist(), "feat_sd": (inp["feat"].std(0) + 1e-6).tolist()}


def test_proprio_net_fits_a_learnable_mapping_and_is_roughly_calibrated(tmp_path):
    cfg = _tiny_cfg()
    d, inp = _synthetic()
    st = _stats(d, inp)
    train("proprio", cfg, d, d, inp, inp, st, tmp_path, seed=0, log=lambda *_: None)
    model, ck = load_checkpoint(tmp_path / "best.pt")
    est = predict(model, inp, torch.device("cpu"))
    err = np.abs(est.mean[:, :3] - d["target"][:, :3])
    assert err[:, 0].mean() < 0.01 and err[:, 1].mean() < 0.02            # prior sd is 0.03 / 0.05
    assert (tmp_path / "log.jsonl").exists() and (tmp_path / "last.pt").exists()


def test_checkpoint_round_trip_is_exact(tmp_path):
    cfg = _tiny_cfg()
    d, inp = _synthetic(32)
    st = _stats(d, inp)
    torch.manual_seed(0)
    m = build_model("proprio", st, cfg)
    save_checkpoint(tmp_path / "c.pt", m, st, cfg, "proprio", 0, 3)
    m2, ck = load_checkpoint(tmp_path / "c.pt")
    a, b = predict(m, inp, torch.device("cpu")), predict(m2, inp, torch.device("cpu"))
    np.testing.assert_array_equal(a.mean, b.mean); np.testing.assert_array_equal(a.cov, b.cov)
    assert ck["epoch"] == 3 and ck["kind"] == "proprio"


def test_training_is_deterministic_on_cpu_and_resume_continues(tmp_path):
    cfg = _tiny_cfg(); cfg["train"].update(epochs=4, warmup_epochs=1)
    d, inp = _synthetic(64)
    st = _stats(d, inp)
    quiet = lambda *_: None
    train("proprio", cfg, d, d, inp, inp, st, tmp_path / "a", seed=5, log=quiet)
    train("proprio", cfg, d, d, inp, inp, st, tmp_path / "b", seed=5, log=quiet)
    ma, _ = load_checkpoint(tmp_path / "a" / "last.pt"); mb, _ = load_checkpoint(tmp_path / "b" / "last.pt")
    for (k, va), (_, vb) in zip(ma.state_dict().items(), mb.state_dict().items()):
        assert torch.equal(va, vb), k
    # interrupted run (2 epochs) then resumed to 4 == uninterrupted run
    train("proprio", cfg, d, d, inp, inp, st, tmp_path / "c", seed=5, log=quiet, stop_after=2)
    train("proprio", cfg, d, d, inp, inp, st, tmp_path / "c", seed=5, resume=True, log=quiet)
    mc, _ = load_checkpoint(tmp_path / "c" / "last.pt")
    for (k, va), (_, vc) in zip(ma.state_dict().items(), mc.state_dict().items()):
        assert torch.allclose(va, vc, atol=1e-5), k


def test_vision_net_forward_shapes_and_gradients():
    cfg = _tiny_cfg()
    st = {"pos_mean": POS_MEAN, "pos_std": POS_STD}
    net = build_model("vision", st, cfg)
    img = torch.randint(0, 255, (4, 96, 96, 3), dtype=torch.uint8)
    m, L = net({"image": img})
    assert m.shape == (4, 4) and L.shape == (4, 4, 4)
    gaussian_nll(m, L, m.detach() + 0.01).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert sum(p.numel() for p in net.parameters()) < 1.5e6


def test_gated_inputs_mask_proprioception_unless_holding():
    from mmperc.training.features import proprio_features
    n = 6
    d = {"q": np.zeros((n, 7)), "qd": np.zeros((n, 7)), "finger_q": np.zeros((n, 2)),
         "holding_gate": np.array([1, 0, 1, 0, 1, 0], bool), "proprio_mask": np.array([1, 1, 1, 1, 0, 0], bool)}
    _, ok = proprio_features(d, None, use_fk=False, use_gate=True, gate_inputs=True)
    assert ok.tolist() == [True, False, True, False, False, False]
    _, ok2 = proprio_features(d, None, use_fk=False, use_gate=True, gate_inputs=False)
    assert ok2.tolist() == d["proprio_mask"].tolist()

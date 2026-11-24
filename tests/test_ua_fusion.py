"""M8: uncertainty-aware fusion. Closed-form checks of the fusion maths, then model-level properties."""
import math

import numpy as np
import pytest
import torch

from mmperc.models.fusion import fuse_gaussians
from mmperc.models.gaussian import covariance, gaussian_nll, mse_warmup, tangent_error
from mmperc.models.unimodal import build_model

PM = torch.tensor([0.5, 0.0, 0.05])
PP = torch.cat([1.0 / torch.tensor([0.06, 0.12, 0.05]) ** 2, torch.tensor([3.0 / math.pi ** 2])])
BROAD = torch.tensor([1e-8, 1e-8, 1e-8, 1e-8])            # practically no prior


def _L(sd):
    return torch.diag(torch.as_tensor(sd, dtype=torch.float32))


def _one(mu, sd, B=1):
    return torch.as_tensor(mu, dtype=torch.float32).expand(B, 4), _L(sd).expand(B, 4, 4)


def test_poe_matches_the_closed_form_precision_weighted_mean():
    m1, L1 = _one([0.50, 0.10, 0.02, 0.3], [0.01, 0.02, 0.03, 0.1])
    m2, L2 = _one([0.52, 0.06, 0.04, 0.4], [0.02, 0.01, 0.01, 0.2])
    mu, L = torch.stack([m1, m2], 1), torch.stack([L1, L2], 1)
    mean, Lf = fuse_gaussians(mu, L, torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool), PM, BROAD)
    P1, P2 = 1 / np.array([0.01, 0.02, 0.03, 0.1]) ** 2, 1 / np.array([0.02, 0.01, 0.01, 0.2]) ** 2
    ref = (P1 * m1[0].numpy() + P2 * m2[0].numpy()) / (P1 + P2)
    np.testing.assert_allclose(mean[0].numpy(), ref, atol=1e-5)
    np.testing.assert_allclose(torch.diagonal(covariance(Lf)[0]).numpy(), 1 / (P1 + P2), rtol=1e-4)


def test_two_identical_experts_double_count_under_poe_but_not_under_covariance_intersection():
    """The reason independence must not be assumed: two perfectly correlated copies of one estimate."""
    m, L = _one([0.5, 0.0, 0.05, 0.0], [0.02] * 4)
    mu, Ls = torch.stack([m, m], 1), torch.stack([L, L], 1)
    ok = torch.ones(1, 2, dtype=torch.bool)
    _, Lp = fuse_gaussians(mu, Ls, torch.ones(1, 2), ok, PM, BROAD)                 # PoE: variance halves (over-confident)
    _, Lc = fuse_gaussians(mu, Ls, torch.full((1, 2), 0.5), ok, PM, BROAD)          # CI (w sums to 1): variance unchanged
    assert covariance(Lp)[0, 0, 0].item() == pytest.approx(0.02 ** 2 / 2, rel=1e-3)
    assert covariance(Lc)[0, 0, 0].item() == pytest.approx(0.02 ** 2, rel=1e-3)


def test_missing_experts_are_excluded_exactly_and_all_missing_gives_the_prior():
    m1, L1 = _one([0.50, 0.10, 0.02, 0.3], [0.01] * 4)
    m2, L2 = _one([0.90, -0.5, 0.30, -2.0], [1e-4] * 4)                              # absurdly confident but missing
    mu, L = torch.stack([m1, m2], 1), torch.stack([L1, L2], 1)
    mask = torch.tensor([[True, False]])
    a, La = fuse_gaussians(mu, L, torch.ones(1, 2), mask, PM, BROAD)
    b, Lb = fuse_gaussians(mu[:, :1], L[:, :1], torch.ones(1, 1), mask[:, :1], PM, BROAD)
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-6)
    np.testing.assert_allclose(covariance(La).numpy(), covariance(Lb).numpy(), rtol=1e-5)
    none, Ln = fuse_gaussians(mu, L, torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool), PM, PP)
    np.testing.assert_allclose(none[0, :3].numpy(), PM.numpy(), atol=1e-6)
    np.testing.assert_allclose(torch.diagonal(covariance(Ln)[0]).numpy(), (1 / PP).numpy(), rtol=1e-4)


def test_phi_is_fused_on_the_circle_across_the_branch_cut():
    m1, L1 = _one([0.5, 0, 0.05, math.pi - 0.01], [0.01] * 4)
    m2, L2 = _one([0.5, 0, 0.05, -math.pi + 0.01], [0.01] * 4)
    mean, _ = fuse_gaussians(torch.stack([m1, m2], 1), torch.stack([L1, L2], 1), torch.ones(1, 2),
                             torch.ones(1, 2, dtype=torch.bool), PM, BROAD)
    assert abs(abs(float(mean[0, 3])) - math.pi) < 0.02                              # NOT ~0 (the naive average)


def test_fused_covariance_is_positive_definite_and_gradients_flow_to_all_experts():
    torch.manual_seed(0)
    mu = torch.randn(5, 3, 4, requires_grad=True)
    raw = torch.randn(5, 3, 4, 4) * 0.3
    L = (torch.tril(raw, -1) + torch.diag_embed(torch.nn.functional.softplus(raw.diagonal(dim1=-2, dim2=-1)) + 1e-2)).requires_grad_()
    w = torch.rand(5, 3).requires_grad_()
    mean, Lf = fuse_gaussians(mu, L, w, torch.ones(5, 3, dtype=torch.bool), PM, PP)
    ev = torch.linalg.eigvalsh(covariance(Lf).double())
    assert (ev > 0).all()
    gaussian_nll(mean, Lf, torch.zeros(5, 4)).backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (mu, L, w))


# ---- model level ---------------------------------------------------------------------------------------------------
def _cfg(mode):
    from mmperc.common.config import load_config
    cfg = load_config("configs/training.yaml")
    cfg["fusion"]["mode"] = mode
    cfg["vision"].update(width=8, keypoints=8)
    return cfg


def _stats(nf=24, nc=13):
    return {"pos_mean": [0.5, 0.0, 0.05], "pos_std": [0.06, 0.12, 0.05], "feat_mu": [0.0] * nf, "feat_sd": [1.0] * nf,
            "cfeat_mu": [0.0] * nc, "cfeat_sd": [1.0] * nc}


def _batch(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"image": torch.randint(0, 255, (n, 96, 96, 3), dtype=torch.uint8, generator=g), "vmask": torch.ones(n, dtype=torch.bool),
            "feat": torch.randn(n, 24, generator=g), "ok": torch.ones(n, dtype=torch.bool),
            "cfeat": torch.randn(n, 13, generator=g), "cok": torch.ones(n, dtype=torch.bool)}


@pytest.mark.parametrize("mode", ["poe", "tpoe", "ci"])
def test_ua_net_outputs_valid_gaussians_and_correct_weight_ranges(mode):
    net = build_model("fusion", _stats(), _cfg(mode))
    m, L = net(_batch())
    assert m.shape == (6, 4) and (torch.linalg.eigvalsh(covariance(L).double()) > 0).all()
    w = net.last["w"]
    if mode == "poe":
        assert (w == 1).all()
    elif mode == "tpoe":
        assert ((w > 0) & (w < 1)).all()
    else:
        assert torch.allclose(w.sum(1), torch.ones(6), atol=1e-5)
    net.aux_loss(torch.zeros(6, 4), True, torch.tensor([0.06, 0.12, 0.05, 1.0]))


def test_ci_weights_are_supported_on_present_experts_only_and_missing_inputs_cannot_leak():
    torch.manual_seed(1)
    net = build_model("fusion", _stats(), _cfg("ci")).eval()
    b = _batch()
    b["ok"] = torch.zeros_like(b["ok"])                                               # proprio missing (-> contact missing too)
    with torch.no_grad():
        m1, L1 = net(b)
        w = net.last["w"]
        assert (w[:, 1:] < 1e-6).all() and torch.allclose(w[:, 0], torch.ones(6), atol=1e-5)
        b2 = dict(b); b2["feat"] = torch.randn_like(b["feat"]) * 50; b2["cfeat"] = torch.randn_like(b["cfeat"]) * 50
        m2, L2 = net(b2)
    assert torch.allclose(m1, m2, atol=1e-5) and torch.allclose(L1, L2, atol=1e-5)    # dead experts do not influence the output


@pytest.mark.parametrize("mode", ["poe", "tpoe", "ci"])
def test_ua_fusion_overfits_a_tiny_batch_with_the_real_recipe_including_aux_loss(mode):
    torch.manual_seed(0)
    net = build_model("fusion", _stats(), _cfg(mode))
    b = _batch(8)
    y = torch.tensor([[0.5, 0.05, 0.03, 0.4]] * 4 + [[0.45, -0.1, 0.12, -0.8]] * 4, dtype=torch.float32)
    scale = torch.tensor([0.06, 0.12, 0.05, 1.0])
    opt = torch.optim.AdamW(net.parameters(), 2e-3)
    for i in range(300):
        m, L = net(b)
        use_nll = i >= 150
        loss = gaussian_nll(m, L, y) if use_nll else mse_warmup(m, y, scale) + 0.1 * gaussian_nll(m.detach(), L, y)
        loss = loss + net.aux_weight * net.aux_loss(y, use_nll, scale)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    rmse_mm = 1e3 * float(tangent_error(m.detach(), y)[:, :3].pow(2).sum(1).mean().sqrt())
    assert rmse_mm < 15.0 and torch.isfinite(loss)

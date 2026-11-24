"""M9: decision statistics, sufficient statistics, protocol plan, ablation switches, and an end-to-end tiny protocol."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import chi2

from mmperc.baselines.common import Estimate
from mmperc.common.config import load_config, repo_root
from mmperc.data.generate import generate_dataset
from mmperc.evaluation.metrics import summarize
from mmperc.evaluation.stats import holm, metric, paired_bootstrap, seed_sign_consistency
from mmperc.evaluation.suffstats import FIELDS, GROUPS, LEVELS, calibration_stats, episode_stats
from mmperc.training.naming import run_name

ROOT = repo_root()


def _synthetic(n=600, n_ep=12, scale=1.0, seed=0):
    """Gaussian errors with covariance diag(s^2) and a matching prediction (scale=1 -> calibrated)."""
    rng = np.random.default_rng(seed)
    sd = np.array([0.01, 0.02, 0.005, 0.2])
    target = np.column_stack([rng.uniform(0.4, 0.6, n), rng.uniform(-0.2, 0.2, n), rng.uniform(0.02, 0.1, n), rng.uniform(-3, 3, n)])
    mean = target - rng.standard_normal((n, 4)) * sd
    cov = np.tile(np.diag((sd * scale) ** 2), (n, 1, 1))
    episode = np.sort(rng.integers(0, n_ep, n))
    stage = rng.integers(0, 5, n)
    return Estimate(mean, cov, np.ones(n, bool)), target, episode, stage, n_ep


def test_episode_stats_reproduce_summarize_per_stage_and_overall():
    est, tgt, ep, st, n_ep = _synthetic()
    S = episode_stats(est, tgt, ep, st, n_ep)
    assert S.shape == (n_ep, 6, len(FIELDS))
    for j, name in enumerate(GROUPS):
        ref = summarize(est, tgt, np.ones(len(tgt), bool) if name == "all" else st == j)
        got = S[:, j, :]
        assert metric(got, "rmse_xy_mm") == pytest.approx(ref["rmse_xy_mm"], rel=1e-9)
        assert metric(got, "rmse3_mm") == pytest.approx(ref["rmse3_mm"], rel=1e-9)
        assert metric(got, "nll") == pytest.approx(ref["nll"], rel=1e-9)
        assert metric(got, "mean_d2") == pytest.approx(ref["mean_d2"], rel=1e-9)
        assert metric(got, "cov95") == pytest.approx(ref["cov95"], rel=1e-9)
        assert metric(got, "yaw_mae_deg") == pytest.approx(ref["yaw_mae_deg"], rel=1e-9)


def test_reliability_counts_match_nominal_levels_when_calibrated_and_fall_short_when_overconfident():
    est, tgt, _, st, _ = _synthetic(n=20000, seed=1)
    c = calibration_stats(est, tgt, st)
    emp = c["reliab"][5] / c["counts"][5]
    assert np.abs(emp - LEVELS).max() < 0.02
    est2, tgt2, _, st2, _ = _synthetic(n=20000, scale=0.3, seed=1)                       # 3x too confident
    c2 = calibration_stats(est2, tgt2, st2)
    assert (c2["reliab"][5] / c2["counts"][5] < LEVELS - 0.05)[3:].all()
    sb = c2["sigma_bins"][5]
    assert sb[:, 0].sum() == 20000 and sb[:, 1].sum() > 0


def test_paired_bootstrap_detects_a_real_difference_and_not_a_null_one():
    est, tgt, ep, st, n_ep = _synthetic(n=3000, n_ep=40, seed=2)
    A = episode_stats(est, tgt, ep, st, n_ep)[:, 5]
    r0 = paired_bootstrap(A, A.copy(), "rmse_xy_mm", n_boot=500)
    assert r0["diff"] == 0 and r0["lo"] <= 0 <= r0["hi"]
    worse = Estimate(est.mean + np.array([0.02, 0.02, 0, 0]), est.cov, est.informative)       # a clearly worse method B
    B = episode_stats(worse, tgt, ep, st, n_ep)[:, 5]
    r = paired_bootstrap(A, B, "rmse_xy_mm", n_boot=1000)
    assert r["diff"] < 0 and r["hi"] < 0 and r["p_better"] < 0.01
    rr = paired_bootstrap(B, A, "rmse_xy_mm", n_boot=1000)
    assert rr["diff"] > 0 and rr["p_better"] > 0.99                                     # direction matters (one-sided p)
    assert seed_sign_consistency([A, A], [B, B], "rmse_xy_mm") == (2, 2)


def test_bootstrap_ci_covers_the_truth_at_roughly_nominal_rate():
    """Calibration of the interval itself: A and B share the true RMSE, so the paired CI must contain 0 ~95% of the time."""
    hits = 0
    for s in range(60):
        est, tgt, ep, st, n_ep = _synthetic(n=800, n_ep=20, seed=100 + s)
        est2, _, _, _, _ = _synthetic(n=800, n_ep=20, seed=100 + s)
        rng = np.random.default_rng(s)
        est2 = Estimate(tgt - rng.standard_normal((800, 4)) * np.array([0.01, 0.02, 0.005, 0.2]), est.cov, est.informative)
        A = episode_stats(est, tgt, ep, st, n_ep)[:, 5]; B = episode_stats(est2, tgt, ep, st, n_ep)[:, 5]
        r = paired_bootstrap(A, B, "rmse_xy_mm", n_boot=300, seed=s)
        hits += r["lo"] <= 0 <= r["hi"]
    assert hits / 60 > 0.85


def test_relative_gain_contrast_distinguishes_a_condition_specific_gain_from_a_uniform_one():
    from mmperc.evaluation.stats import relative_gain_contrast
    est, tgt, ep, st, n_ep = _synthetic(n=3000, n_ep=40, seed=3)
    good = episode_stats(est, tgt, ep, st, n_ep)[:, 5]
    def worse(k):                                                                         # reference with error scaled by k
        e = Estimate(tgt - (tgt - est.mean) * k, est.cov, est.informative)
        return episode_stats(e, tgt, ep, st, n_ep)[:, 5]
    # candidate = good everywhere; reference is 3x worse in condition 1 but only equal in condition 0 -> gain is condition-specific
    r = relative_gain_contrast(good, worse(3.0), good, worse(1.0), n_boot=500)
    assert r["rho1"] > 0.5 and abs(r["rho0"]) < 1e-9 and r["lo"] > 0 and r["p_greater"] < 0.01
    u = relative_gain_contrast(good, worse(3.0), good, worse(3.0), n_boot=500)               # same gain in both conditions
    assert abs(u["contrast"]) < 1e-9 and u["lo"] <= 0 <= u["hi"]


def test_holm_matches_hand_computation_and_is_monotone():
    adj = holm([0.01, 0.04, 0.03], 0.05)
    assert [round(a, 4) for a, _ in adj] == [0.03, 0.06, 0.06]
    assert [r for _, r in adj] == [True, False, False]
    assert holm([0.5])[0] == (0.5, False)


def test_run_names_in_the_protocol_match_what_train_py_writes():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_protocol as rp
    parsed = {}
    for name, args in rp.RUNS.items():
        kw = {"kind": args[args.index("--kind") + 1], "mode": args[args.index("--mode") + 1] if "--mode" in args else "learned",
              "gate_inputs": "--gate-inputs" in args, "mixed": "--mixed" in args, "tag": args[args.index("--tag") + 1] if "--tag" in args else ""}
        assert run_name(**kw) == name
        parsed[name] = kw
    assert len(set(rp.RUNS)) == len(rp.RUNS)
    p = rp.plan([0, 1], None)
    assert [s for s, _, _ in p] == [0] * len(rp.RUNS) + [1] * len(rp.RUNS)            # seed-major
    assert rp.plan([0], ["vision"]) == [(0, "vision", rp.RUNS["vision"])]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    sim = load_config("configs/simulation.yaml")
    ds = copy.deepcopy(load_config("configs/dataset.yaml"))
    ds["splits"].update(n_train=6, n_val=3, n_test=2, n_test_ood=1)
    out = tmp_path_factory.mktemp("tiny_protocol")
    generate_dataset(sim, ds, out, workers=1)
    return out


def test_no_contact_switch_removes_the_contact_expert_completely(tiny):
    from mmperc.baselines.kinematics import HandKinematics
    from mmperc.data.dataset import load_split
    from mmperc.models.unimodal import build_model
    from mmperc.training.features import fit_stats, make_inputs
    sim = load_config("configs/simulation.yaml")
    cfg = load_config("configs/training.yaml"); cfg["fusion"]["mode"] = "ci"; cfg["vision"].update(width=8, keypoints=8)
    d = load_split(tiny, "val", load_images=True)
    hk = HandKinematics(sim)
    on = make_inputs("fusion", d, cfg, hk)
    cfg["fusion"]["use_contact"] = False
    off = make_inputs("fusion", d, cfg, hk)
    assert on["cok"].any() and not off["cok"].any() and not off["cfeat"].any()
    assert (on["ok"] == off["ok"]).all() and np.array_equal(on["feat"], off["feat"])      # nothing else changed
    st = fit_stats(d, off)
    assert np.isfinite(st["cfeat_mu"]).all() and np.isfinite(st["cfeat_sd"]).all()        # regression: empty stream gave NaN statistics
    net = build_model("fusion", st, cfg).eval()
    b = {k: torch.as_tensor(v[:16]) for k, v in off.items()}
    net(b)
    assert (net.last["w"][:, 2] == 0).all() and not net.last["mask"][:, 2].any()


def test_allcombos_flag_empties_the_forbidden_pairs():
    """The H3 control model must be trained WITH the combinations the main model never sees."""
    from mmperc.training.mixing import Mixer, is_forbidden
    from mmperc.sensors.degradation import Scenario
    sim, deg = load_config("configs/simulation.yaml"), load_config("configs/degradation.yaml")
    from mmperc.sensors.degradation import Degrader
    cfg = load_config("configs/training.yaml")
    m_main = Mixer("fusion", cfg, Degrader(deg, sim), lambda *a: {}, 0)
    cfg["mixing"]["forbidden_pairs"] = []
    m_all = Mixer("fusion", cfg, Degrader(deg, sim), lambda *a: {}, 0)
    combo = Scenario.make(vision_occlusion=2, proprio_noise=2)
    assert is_forbidden(combo, m_main.forbidden) and not is_forbidden(combo, m_all.forbidden)
    seen = sum(any(f == "vision_occlusion" for f, _ in m_all._scenario_fn(5)(i).levels) and
               any(f == "proprio_noise" for f, _ in m_all._scenario_fn(5)(i).levels) for i in range(3000))
    assert seen > 0                                                                      # the control does see them


def test_protocol_end_to_end_on_a_tiny_dataset(tiny, tmp_path):
    """run_protocol (train 2 models, DONE markers, resume skip, config-fingerprint guard) -> evaluate_all -> Results."""
    py = sys.executable
    root = tmp_path / "runs"
    extra = ["--extra", "--epochs", "4", "--train-episodes", "4", "--val-episodes", "2"]
    cmd = [py, str(ROOT / "scripts/run_protocol.py"), "--seeds", "0", "--out-root", str(root), "--data", str(tiny),
           "--only", "fusion_ci_mixed_nocontact", "proprio_gated"]
    r = subprocess.run(cmd + extra, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for n in ("fusion_ci_mixed_nocontact", "proprio_gated"):
        assert (root / n / "seed0" / "DONE").exists() and (root / n / "seed0" / "best.pt").exists()
    assert json.loads((root / "fusion_ci_mixed_nocontact/seed0/config.json").read_text())["cfg"]["fusion"]["use_contact"] is False
    r2 = subprocess.run(cmd + extra, cwd=ROOT, capture_output=True, text=True)           # second call: nothing to do
    assert r2.returncode == 0 and ">>>" not in r2.stdout
    m = root / "protocol_manifest.json"
    d = json.loads(m.read_text()); d["fingerprint"]["configs/training.yaml"] = "0" * 16; m.write_text(json.dumps(d))
    r3 = subprocess.run(cmd + extra, cwd=ROOT, capture_output=True, text=True)
    assert r3.returncode != 0 and "REFUSED" in (r3.stdout + r3.stderr)                    # changed config is refused

    out = tmp_path / "results"
    ev = [py, str(ROOT / "scripts/evaluate_all.py"), "--runs-root", str(root), "--data", str(tiny), "--split", "test", "--out", str(out),
          "--train-episodes", "4", "--only", "clean", "occl L4 + proprio bias L3", "proprio dropout L3"]
    r4 = subprocess.run(ev, cwd=ROOT, capture_output=True, text=True)
    assert r4.returncode == 0, r4.stdout[-2000:] + r4.stderr[-2000:]
    from mmperc.evaluation.results import Results
    res = Results(out / "test")
    assert res.meta["n_episodes"] == 2 and "fusion_ci_mixed_nocontact" in res.models() and "B6 WLS" in res.models()
    ep = res.ep("clean", "fusion_ci_mixed_nocontact", "all")
    assert len(ep) == 1 and ep[0].shape == (2, len(FIELDS)) and np.isfinite(ep[0]).all()
    # the sufficient statistics must agree with a direct evaluation of the same checkpoint
    from mmperc.data.dataset import load_split
    from mmperc.baselines.kinematics import HandKinematics
    from mmperc.training.engine import load_checkpoint, predict
    from mmperc.training.features import make_inputs
    model, ck = load_checkpoint(root / "fusion_ci_mixed_nocontact/seed0/best.pt")
    te = load_split(tiny, "test", load_images=True)
    est = predict(model, make_inputs("fusion", te, ck["cfg"], HandKinematics(load_config("configs/simulation.yaml"))), torch.device("cpu"))
    ref = summarize(est, te["target"].astype(float))
    assert metric(ep[0], "rmse_xy_mm") == pytest.approx(ref["rmse_xy_mm"], rel=1e-6)
    w, av = res.get("clean", "fusion_ci_mixed_nocontact", "w_sum", 0), res.get("clean", "fusion_ci_mixed_nocontact", "avail_sum", 0)
    assert w[5, 2] == 0 and av[5, 2] == 0                                                # no contact expert anywhere
    assert not (out / "test" / "s05.npz").exists()                                       # --only respected
    assert (out / "test" / "s00.npz").exists()

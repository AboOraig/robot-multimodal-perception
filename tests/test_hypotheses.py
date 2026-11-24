"""M10a: the decision rules applied to synthetic result files whose ground truth is known."""
import json
import zlib

import numpy as np
import pytest

from mmperc.baselines.common import Estimate
from mmperc.evaluation import hypotheses as H
from mmperc.evaluation.results import Results
from mmperc.evaluation.scenarios import SCENARIOS
from mmperc.evaluation.suffstats import FIELDS, GROUPS, calibration_stats, episode_stats

N, NEP = 1500, 30
SD = np.array([1.0, 1.0, 0.5, 1.0])
CLASSICAL = ["B6 WLS", "B6c WLS+contact [oracle]"]


def _est(target, sd_mm, conf, rng):
    sd = sd_mm * 1e-3 * SD
    mean = target - rng.standard_normal(target.shape) * sd
    cov = np.tile(np.diag((sd * conf) ** 2), (len(target), 1, 1))
    cov[:, 3, 3] = 0.01
    mean[:, 3] = target[:, 3]
    return Estimate(mean, cov, np.ones(len(target), bool))


def _world(tmp_path, rule, seeds=(0, 1, 2)):
    """rule(model, scenario) -> (sd_mm, confidence ratio). All frames are in stage 3 ('coupled')."""
    rng0 = np.random.default_rng(0)
    target = np.column_stack([rng0.uniform(.4, .6, N), rng0.uniform(-.2, .2, N), rng0.uniform(.02, .1, N), rng0.uniform(-3, 3, N)])
    episode, stage = np.sort(rng0.integers(0, NEP, N)), np.full(N, 3)
    models = {m: list(seeds) for m in set(H.M.values()) - set(CLASSICAL)}
    meta = {"split": "test", "data": "x", "scenarios": list(SCENARIOS), "fields": FIELDS, "groups": GROUPS, "n_episodes": NEP,
            "ep_id": list(range(NEP)), "models": models, "classical": CLASSICAL, "deg_seed": 0}
    d = tmp_path / "test"; d.mkdir()
    (d / "meta.json").write_text(json.dumps(meta))
    for si, s in enumerate(SCENARIOS):
        out = {}
        for m, sl in list(models.items()) + [(c, [0]) for c in CLASSICAL]:
            for seed in sl:
                sd, conf = rule(m, s)
                rng = np.random.default_rng(zlib.crc32(f"{m}|{s}|{seed}".encode()))
                e = _est(target, sd, conf, rng)
                out[f"{m}__{seed}__ep"] = episode_stats(e, target, episode, stage, NEP)
                cal = calibration_stats(e, target, stage)
                for k, v in cal.items():
                    out[f"{m}__{seed}__{k}"] = v
        np.savez_compressed(d / f"s{si:02d}.npz", **out)
    return Results(d)


def _good_world(m, s):
    """The world the hypotheses describe: fusion >> vision under occlusion, CI honest, non-UA overconfident, clean-trained fails on combos."""
    occl = "occl" in s or "dropout" in s
    if m == "vision_mixed":
        return (20.0 if occl else 1.0), 1.0
    if m in ("fusion_learned_mixed", "fusion_gated_mixed"):
        return 4.0, 0.4                                   # accurate enough, overconfident
    if m == "fusion_ci":
        return (25.0 if "+" in s else 2.0), 0.4           # clean-trained: breaks on combinations
    if m == "fusion_ci_mixed_nocontact":
        return 3.0, 1.0
    return 3.0, 1.0                                       # F-CI and its variants


def test_all_three_hypotheses_are_supported_in_a_world_where_they_are_true(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "N_BOOT", 400)
    res = _world(tmp_path, _good_world)
    h1, h2, h3 = H.h1(res), H.h2(res), H.h3(res)
    # H1: gain under occlusion, but NOT on clean (vision 1 mm vs fusion 3 mm) -> 'particularly under occlusion' holds
    assert h1["verdict"] == "SUPPORTED" and h1["contrast"]["lo"] > 0
    assert h2["verdict"] == "SUPPORTED" and all(v["nll_gains"] == 9 for v in h2["per_comparator"].values())
    assert h3["verdict"] == "SUPPORTED" and h3["H3a"] and h3["H3b_noninferior"]


def test_h1_is_falsified_when_fusion_is_no_better_than_vision(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "N_BOOT", 400)
    res = _world(tmp_path, lambda m, s: (3.0, 1.0))                            # every model identical in distribution
    h1 = H.h1(res)
    assert h1["verdict"] == "FALSIFIED"
    assert H.h2(res)["verdict"] == "FALSIFIED" and H.h3(res)["verdict"] == "FALSIFIED"


def test_h3_is_only_partially_supported_when_seeing_the_combination_matters(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "N_BOOT", 400)
    def rule(m, s):
        if m == "fusion_ci_mixed_allcombos":
            return 3.0, 1.0
        if m == "fusion_ci_mixed" and "+" in s:
            return 9.0, 1.0                                                    # better than clean-trained (25 mm), clearly worse than allcombos
        return _good_world(m, s)
    h3 = H.h3(_world(tmp_path, rule))
    assert h3["H3a"] and not h3["H3b_noninferior"] and h3["verdict"].startswith("PARTIALLY")


def test_a_gain_that_is_not_sign_consistent_over_seeds_is_not_significant(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "N_BOOT", 300)
    res = _world(tmp_path, _good_world)
    r = H._pair(res, "occlusion L4", "fusion_ci_mixed", "vision_mixed", "rmse_xy_mm")
    assert r["seeds_a_lower"] == r["n_seeds"] == 3 and r["diff"] < -10
    r2 = H._pair(res, "occlusion L4", "vision_mixed", "fusion_ci_mixed", "rmse_xy_mm")
    assert r2["seeds_a_lower"] == 0 and r2["p_worse"] < 0.01 and not H._ok(r2)

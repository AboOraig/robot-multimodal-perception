"""Applies the decision rules of docs/preregistration.md to the output of scripts/evaluate_all.py.

Every threshold below is copied from the pre-registration; changing one here is a deviation that must be logged there.
"""
from __future__ import annotations

import numpy as np

from mmperc.evaluation.results import Results
from mmperc.evaluation.stats import holm, paired_bootstrap, relative_gain_contrast, seed_average, seed_sign_consistency

GROUP = "coupled"
MIN_MM = 1.0                       # practical-significance floor for RMSE differences
H2_MIN_SCENARIOS = 6               # of 9
H2_MAX_RMSE_WORSE = 2
NONINF_NLL, NONINF_RATIO = 0.5, 1.25
N_BOOT = 5000

M = {"vision": "vision_mixed", "ci": "fusion_ci_mixed", "f1": "fusion_learned_mixed", "f2": "fusion_gated_mixed",
     "poe": "fusion_poe_mixed", "tpoe": "fusion_tpoe_mixed", "ci_clean": "fusion_ci", "allcombos": "fusion_ci_mixed_allcombos",
     "nocontact": "fusion_ci_mixed_nocontact", "wls": "B6 WLS", "wls_oracle": "B6c WLS+contact [oracle]"}
SET_O = ["occlusion L2", "occlusion L4", "random occl L4"]
SET_U = ["occlusion L4", "random occl L4", "vis noise L3", "illumination L3", "blur L3", "vision dropout L3",
         "proprio noise L3", "proprio bias L3", "proprio dropout L3"]
SET_C = ["occl L3 + proprio noise L2", "occl L4 + proprio bias L3"]


def _pair(res: Results, scen, a, b, name, group=GROUP, ratio=False):
    """paired comparison a vs b on one scenario; seeds paired by index; adds sign-consistency over seeds."""
    sa, sb = res.ep(scen, a, group), res.ep(scen, b, group)
    out = paired_bootstrap(seed_average(sa), seed_average(sb), name, N_BOOT, ratio=ratio)
    k = min(len(sa), len(sb))
    out["seeds_a_lower"], out["n_seeds"] = seed_sign_consistency(sa[:k], sb[:k], name)
    out["scenario"], out["a_model"], out["b_model"], out["metric"] = scen, a, b, name
    return out


def _ok(r, better=True):
    """sign-consistent over seeds (all seeds in the stated direction)."""
    return r["seeds_a_lower"] == r["n_seeds"] if better else r["seeds_a_lower"] == 0


def h1(res: Results) -> dict:
    rows = [_pair(res, s, M["ci"], M["vision"], "rmse_xy_mm") for s in SET_O]
    adj = holm([r["p_better"] for r in rows])
    for r, (p, rej) in zip(rows, adj):
        r["p_holm"] = p
        r["significant"] = bool(rej and _ok(r) and r["diff"] <= -MIN_MM)
    a = sum(r["significant"] for r in rows) >= 2
    sa = lambda m, sc: seed_average(res.ep(sc, m, GROUP))
    c = relative_gain_contrast(sa(M["ci"], "occlusion L4"), sa(M["vision"], "occlusion L4"), sa(M["ci"], "clean"), sa(M["vision"], "clean"), n_boot=N_BOOT)
    b = c["lo"] > 0
    extra = [_pair(res, s, M["ci"], M["vision"], "rmse_xy_mm") for s in ("clean", "vision dropout L3", "occl L4 + proprio bias L3")]
    verdict = "SUPPORTED" if a and b else ("PARTIALLY SUPPORTED (gain under occlusion, but not larger than clean)" if a else "FALSIFIED")
    return {"verdict": verdict, "a_two_of_three": a, "b_particularly_under_occlusion": b, "contrast": c, "tests": rows, "reported_not_decisive": extra}


def h2(res: Results) -> dict:
    tests = []
    for comp in ("f1", "f2"):
        for s in SET_U:
            nll = _pair(res, s, M["ci"], M[comp], "nll"); rm = _pair(res, s, M["ci"], M[comp], "rmse_xy_mm")
            tests.append({"comparator": M[comp], "scenario": s, "nll": nll, "rmse": rm})
    for key, p in (("nll", "p_better"), ("rmse", "p_worse")):
        adj = holm([t[key][p] for t in tests])
        for t, (pv, rej) in zip(tests, adj):
            t[key]["p_holm"] = pv
            t[key]["significant"] = bool(rej and _ok(t[key], better=(key == "nll")))      # nll: lower in every seed; rmse 'worse': higher in every seed
    for t in tests:
        t["nll_gain"] = t["nll"]["significant"] and t["nll"]["diff"] < 0
        t["rmse_worse"] = t["rmse"]["significant"] and t["rmse"]["diff"] >= MIN_MM
    per = {}
    for comp in ("f1", "f2"):
        ts = [t for t in tests if t["comparator"] == M[comp]]
        per[M[comp]] = {"nll_gains": sum(t["nll_gain"] for t in ts), "rmse_worse": sum(t["rmse_worse"] for t in ts)}
    sup = all(v["nll_gains"] >= H2_MIN_SCENARIOS and v["rmse_worse"] <= H2_MAX_RMSE_WORSE for v in per.values())
    fals = any(v["nll_gains"] < H2_MIN_SCENARIOS for v in per.values())
    # H2b: independence assumption (secondary)
    b = []
    for other in ("poe", "tpoe"):
        for s in SET_U:
            g = _pair(res, s, M["ci"], M[other], "abs_d2_gap"); n = _pair(res, s, M["ci"], M[other], "nll")
            b.append({"other": M[other], "scenario": s, "gap_ci": g["a"], "gap_other": g["b"], "nll_diff": n["diff"], "gap_diff_ci": [g["lo"], g["hi"]]})
    c = [_pair(res, s, M["ci"], M[w], m) for w in ("wls", "wls_oracle") for s in SET_U for m in ("rmse_xy_mm", "nll")] if M["wls"] in res.models() else []
    return {"verdict": "SUPPORTED" if sup else ("FALSIFIED" if fals else "NOT SUPPORTED (NLL gains yes, but RMSE worse in >2 scenarios)"),
            "per_comparator": per, "tests": tests, "H2b_independence": b, "H2c_classical": c}


def h3(res: Results) -> dict:
    rows = [_pair(res, s, M["ci"], M["ci_clean"], "nll") for s in SET_C]
    for r, (p, rej) in zip(rows, holm([r["p_better"] for r in rows])):
        r["p_holm"] = p; r["significant"] = bool(rej and _ok(r))
    a = all(r["significant"] for r in rows)
    ctrl = []
    for s in SET_C:
        n = _pair(res, s, M["ci"], M["allcombos"], "nll"); q = _pair(res, s, M["ci"], M["allcombos"], "rmse_xy_mm", ratio=True)
        ctrl.append({"scenario": s, "nll": n, "rmse_ratio": q, "noninferior": bool(n["hi"] <= NONINF_NLL and q["hi"] <= NONINF_RATIO)})
    bb = all(c["noninferior"] for c in ctrl)
    verdict = "SUPPORTED" if a and bb else ("PARTIALLY SUPPORTED (mixed training helps; seeing the combination helps further)" if a else "FALSIFIED")
    cost = _pair(res, "clean", M["ci"], M["ci_clean"], "rmse_xy_mm")
    return {"verdict": verdict, "H3a": a, "H3b_noninferior": bb, "tests": rows, "control": ctrl, "cost_on_clean": cost}


def contact_ablation(res: Results) -> list[dict]:
    return [_pair(res, s, M["ci"], M["nocontact"], m) for s in res.scenarios() for m in ("nll", "rmse_xy_mm")
            if M["nocontact"] in res.models()]

"""Figures for the report from evaluate_all.py results (+ decisions.json from analyze.py). Every figure is labelled simulation-only.

    python scripts/make_figures.py --results F:/results/final --split test --analysis F:/results/analysis_test --out F:/results/figures_test
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from scipy.stats import chi2                                      # noqa: E402

from mmperc.data.dataset import resolve                           # noqa: E402
from mmperc.evaluation import hypotheses as H                     # noqa: E402
from mmperc.evaluation.results import Results                     # noqa: E402
from mmperc.evaluation.stats import metric                        # noqa: E402
from mmperc.evaluation.suffstats import GROUPS, LEVELS, SIGMA_EDGES_MM   # noqa: E402

G = GROUPS.index("coupled")
# Okabe-Ito colour-blind-safe palette
COL = {"vision_mixed": "#0072B2", "proprio_gated_mixed": "#999999", "fusion_learned_mixed": "#E69F00", "fusion_gated_mixed": "#D55E00",
       "fusion_ci_mixed": "#009E73", "fusion_poe_mixed": "#CC79A7", "fusion_tpoe_mixed": "#56B4E9", "fusion_ci": "#000000",
       "B6c WLS+contact [oracle]": "#8C564B"}
LAB = {"vision_mixed": "vision only (mixed)", "proprio_gated_mixed": "proprioception (FK, mixed)", "fusion_learned_mixed": "F1 learned fusion",
       "fusion_gated_mixed": "F2 gated fusion", "fusion_ci_mixed": "F-CI (uncertainty-aware)", "fusion_poe_mixed": "F-PoE", "fusion_tpoe_mixed": "F-tPoE",
       "fusion_ci": "F-CI trained clean", "B6c WLS+contact [oracle]": "classical WLS (oracle noise)"}
SIM = "SIMULATION ONLY (MuJoCo, Franka Panda, one cube)"


def _fin(fig, path, title=None):
    if title:
        fig.suptitle(title, fontsize=11, y=1.04)
    fig.text(0.99, 0.995, SIM, ha="right", va="top", fontsize=7, color="#666666")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_rmse(res, out, split):
    scen = [s for s in ("clean", "occlusion L4", "random occl L4", "proprio noise L3", "proprio bias L3", "proprio dropout L3",
                        "occl L3 + proprio noise L2", "occl L4 + proprio bias L3") if s in res.scenarios()]
    models = [m for m in ("vision_mixed", "proprio_gated_mixed", "fusion_learned_mixed", "fusion_gated_mixed", "fusion_ci_mixed", "B6c WLS+contact [oracle]") if m in res.models()]
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        v = np.array([[metric(e, "rmse_xy_mm") for e in res.ep(s, m, "coupled")] for s in scen], dtype=object)
        mu = [np.mean(x) for x in v]; sd = [np.std(x) for x in v]
        ax.bar(np.arange(len(scen)) + i * w, mu, w, yerr=sd, color=COL[m], label=LAB[m], capsize=1.5)
    ax.set_yscale("log"); ax.set_ylabel("xy RMSE (mm), log scale, hold stage"); ax.set_xticks(np.arange(len(scen)) + 0.4 - w / 2)
    ax.set_xticklabels([s.replace(" + ", "\n+ ") for s in scen], fontsize=8); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22)); ax.grid(axis="y", alpha=.3)
    _fin(fig, out / "fig1_rmse_by_scenario.png", f"Position error by scenario ({split}; bars = mean over seeds, whiskers = std)")


def _pooled_reliab(res, model, scens):
    k = n = 0
    for s in scens:
        for sd in res.seeds(model):
            k = k + res.get(s, model, "reliab", sd)[G]; n += res.get(s, model, "counts", sd)[G]
    return k / n


def fig_calibration(res, out, split):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (name, scens) in zip(axs, (("single-modality failures (set U)", [s for s in H.SET_U if s in res.scenarios()]),
                                       ("unseen combinations (set C)", H.SET_C))):
        ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
        for m in ("fusion_ci_mixed", "fusion_tpoe_mixed", "fusion_poe_mixed", "fusion_learned_mixed", "vision_mixed", "fusion_ci"):
            if m in res.models():
                ax.plot(LEVELS, _pooled_reliab(res, m, scens), marker="o", ms=3, color=COL[m], label=LAB[m])
        ax.set_title(name, fontsize=9); ax.set_xlabel("nominal coverage"); ax.grid(alpha=.3)
    axs[0].set_ylabel("empirical coverage (4-D ellipsoid)"); axs[0].legend(fontsize=7)
    _fin(fig, out / "fig2_reliability.png", f"Calibration ({split}, hold stage): on the diagonal = honest uncertainty; below = over-confident")


def fig_sigma(res, out, split):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    mid = np.sqrt(SIGMA_EDGES_MM[:-1] * SIGMA_EDGES_MM[1:])
    ax.plot([0.1, 1e3], [0.1, 1e3], color="#999", ls="--", lw=1)
    scens = [s for s in H.SET_U if s in res.scenarios()]
    for m in ("fusion_ci_mixed", "fusion_poe_mixed", "fusion_learned_mixed", "vision_mixed"):
        if m not in res.models():
            continue
        acc = sum(res.get(s, m, "sigma_bins", sd)[G] for s in scens for sd in res.seeds(m))
        ok = acc[:, 0] >= 200
        ax.plot(mid[ok], np.sqrt(acc[ok, 1] / acc[ok, 0] / 2), marker="o", ms=3, color=COL[m], label=LAB[m])   # per-axis RMS error (2 coordinates)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("predicted xy sigma (mm, per axis)"); ax.set_ylabel("realised RMS error per axis (mm)")
    ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
    _fin(fig, out / "fig3_error_vs_sigma.png", f"Error vs predicted uncertainty ({split}, set U, hold stage)")


def fig_train_matrix(res, out, split):
    scen = [s for s in res.scenarios() if s not in ("contact noise L2", "contact noise L4")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, (nm, name, f) in zip(axs, (("rmse_xy_mm", "xy RMSE (mm)", np.log10), ("mean_d2", "mean d2 (colour: distance from 4 = calibrated)", np.log10))):
        mat = np.array([[np.mean([metric(e, nm) for e in res.ep(s, m, "coupled")]) for m in ("fusion_ci", "fusion_ci_mixed")] for s in scen])
        shade = f(mat) if nm == "rmse_xy_mm" else np.abs(np.log10(mat / 4.0))      # d2: colour = distance from calibration (4), not size
        im = ax.imshow(shade, cmap="viridis_r", aspect="auto")
        for i in range(len(scen)):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.3g}", ha="center", va="center", color="white" if shade[i, j] > np.median(shade) else "black", fontsize=7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["train: clean", "train: mixed"]); ax.set_title(name, fontsize=9)
    axs[0].set_yticks(range(len(scen))); axs[0].set_yticklabels(scen, fontsize=7)
    _fin(fig, out / "fig4_clean_vs_mixed_training.png", f"F-CI trained on clean vs mixed degradations, tested on each scenario ({split}, hold stage)")


def fig_weights(res, out, split):
    m = "fusion_ci_mixed"
    scen = [s for s in res.scenarios() if s not in ("contact noise L2",)]
    W = np.array([[np.mean([res.get(s, m, "w_sum", sd)[G, k] / res.get(s, m, "counts", sd)[G] for sd in res.seeds(m)]) for k in range(3)] for s in scen])
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(scen))
    for k, (nm, c) in enumerate((("vision", "#0072B2"), ("proprioception", "#999999"), ("contact", "#E69F00"))):
        ax.bar(range(len(scen)), W[:, k], bottom=bottom, color=c, label=nm); bottom += W[:, k]
    ax.set_xticks(range(len(scen))); ax.set_xticklabels([s.replace(" + ", "\n+ ") for s in scen], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("mean covariance-intersection weight"); ax.legend(fontsize=8)
    _fin(fig, out / "fig5_expert_weights.png", f"How F-CI distributes trust among experts ({split}, hold stage)")


def fig_ablation(res, out, split):
    if "fusion_ci_mixed_nocontact" not in res.models():
        return
    scen = [s for s in ("clean", "occlusion L4", "random occl L4", "proprio noise L3", "proprio dropout L3", "occl L4 + proprio bias L3") if s in res.scenarios()]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (m, c, l) in enumerate((("fusion_ci_mixed", "#009E73", "F-CI (all experts)"), ("fusion_ci_mixed_nocontact", "#CC79A7", "F-CI without contact expert"))):
        v = [[metric(e, "rmse_xy_mm") for e in res.ep(s, m, "coupled")] for s in scen]
        ax.bar(np.arange(len(scen)) + i * 0.4, [np.mean(x) for x in v], 0.4, yerr=[np.std(x) for x in v], color=c, label=l, capsize=2)
    ax.set_xticks(np.arange(len(scen)) + .2); ax.set_xticklabels([s.replace(" + ", "\n+ ") for s in scen], fontsize=7)
    ax.set_ylabel("xy RMSE (mm), hold stage"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    _fin(fig, out / "fig6_contact_ablation.png", f"Contact-expert ablation ({split})")


def fig_forest(dec, out, split):
    rows = [(f"H1  {t['scenario']}", t["diff"], t["lo"], t["hi"]) for t in dec["h1"]["tests"]]
    rows += [(f"H1* {t['scenario']}", t["diff"], t["lo"], t["hi"]) for t in dec["h1"]["reported_not_decisive"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (n, d, lo, hi) in enumerate(rows[::-1]):
        ax.plot([lo, hi], [i, i], color="#009E73", lw=2); ax.plot(d, i, "o", color="#009E73")
    ax.axvline(0, color="#999", lw=1); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
    ax.set_xlabel("RMSE(F-CI) - RMSE(vision only), mm  (negative = fusion better)   [* = reported, not decisive]"); ax.set_xscale("symlog", linthresh=5)
    _fin(fig, out / "fig7_h1_forest.png", f"H1: paired episode-bootstrap 95% CI ({split}, hold stage)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--analysis", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = Results(resolve(a.results) / a.split)
    dec = json.loads((resolve(a.analysis) / "decisions.json").read_text())
    out = resolve(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in (fig_rmse, fig_calibration, fig_sigma, fig_train_matrix, fig_weights, fig_ablation):
        f(res, out, a.split); print("ok", f.__name__)
    fig_forest(dec, out, a.split); print("ok fig_forest ->", out)


if __name__ == "__main__":
    main()

"""Evaluate the classical baselines under clean and degraded conditions, stratified by stage.

    python scripts/run_baselines.py --data data/v1 --split val
    python scripts/run_baselines.py --data data/v1 --split test --oracle      # known noise levels

Calibration data (priors, in-hand offset statistics, tare, pixel noise) come from the TRAIN split only.
Estimators:  B1a vision (table plane) | B1b vision (broad height prior) | B3 kinematics (gated by the
holding detector) | B3c kinematics + wrench in-hand offset | B6 WLS fusion (vision + B3) | B6c (vision + B3c).
Default covariances assume CLEAN sensors (the classical assumption); --oracle passes the true noise
standard deviations of the proprioception / contact degradations (an upper bound for this family).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from mmperc.common.config import load_config
from mmperc.data.dataset import STAGE_NAMES, load_split, resolve
from mmperc.evaluation.classical import build_classical, oracle_kwargs
from mmperc.evaluation.metrics import summarize
from mmperc.evaluation.scenarios import SCENARIOS
from mmperc.sensors.degradation import Degrader, Scenario

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--train-episodes", type=int, default=None, help="limit train episodes (debug)")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out", default="experiments/results/baselines.json")
    a = ap.parse_args()

    sim, deg = load_config("configs/simulation.yaml"), load_config("configs/degradation.yaml")
    tr = load_split(a.data, "train", load_images=True, max_episodes=a.train_episodes)
    te = load_split(a.data, a.split, load_images=True, max_episodes=a.max_episodes)
    estimators, cal = build_classical(sim, tr)
    vis, b3, b3c = cal["vis"], cal["b3"], cal["b3c"]
    print(f"calibration (train): sigma_pix={np.round(np.sqrt(np.diag(vis.cov_pix)), 2)} px | in-hand offset sd (mm)="
          f"{np.round(np.sqrt(np.diag(b3.cov_delta)) * 1e3, 1)} | contact residual sd (mm)={np.round(b3c.contact.res_std * 1e3, 2)}")
    dg = Degrader(deg, sim)
    seed = int(deg["seed"])
    target = te["target"].astype(float)
    results = {}
    for sname, levels in SCENARIOS.items():
        sc = Scenario.make(**levels)
        d = dg.apply(te, sc, seed) if levels else te
        kw = {}
        if a.oracle:
            kw = oracle_kwargs(dg, sc, deg)
        print(f"\n=== {sname}  ({sc.name})")
        print(f"{'estimator':<20}" + "".join(f"{s:>20}" for s in STAGE_NAMES))
        results[sname] = {}
        for ename, fn in estimators.items():
            est = fn(d, kw)
            row, rec = "", {}
            for si, sn in enumerate(STAGE_NAMES):
                r = summarize(est, target, te["stage"] == si)
                rec[sn] = r
                row += f"{r['rmse_xy_mm']:6.1f}|{r['yaw_mae_deg']:4.1f}|{r['mean_d2']:6.1f}".rjust(20)
            results[sname][ename] = rec
            print(f"{ename:<20}{row}")
    p = resolve(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"split": a.split, "oracle": a.oracle, "results": results}, indent=1))
    print("\n(cells: xy RMSE mm | yaw MAE deg | mean Mahalanobis d2, 4 = calibrated)  saved", p)


if __name__ == "__main__":
    main()

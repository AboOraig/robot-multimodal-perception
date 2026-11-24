"""Evaluate trained learned baselines under the shared degradation scenarios (same table as run_baselines.py).

    python scripts/evaluate.py --ckpt experiments/runs/vision/seed*/best.pt --split val
Several checkpoints (seeds) are averaged cell-wise; the JSON keeps per-seed values and the std.
Use --split test only for final numbers (validation is used for model selection).
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json

import numpy as np

from mmperc.baselines.kinematics import HandKinematics
from mmperc.common.config import load_config
from mmperc.data.dataset import STAGE_NAMES, load_split, resolve
from mmperc.evaluation.metrics import summarize
from mmperc.evaluation.scenarios import SCENARIOS
from mmperc.sensors.degradation import Degrader, Scenario
from mmperc.training.engine import load_checkpoint, pick_device, predict
from mmperc.training.features import ablate_inputs, make_inputs
from mmperc.training.mixing import relevant_families


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="checkpoint paths or globs")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--ablate", nargs="+", choices=["vision", "proprio", "contact"], default=[],
                    help="fusion only: replace these modalities by a dead sensor at TEST time (reliance diagnostic)")
    ap.add_argument("--only", nargs="+", default=None, help="evaluate only these scenario names (default: all)")
    ap.add_argument("--out", default="experiments/results/learned_val.json")
    a = ap.parse_args()
    paths = sorted(p for g in a.ckpt for p in glob.glob(str(resolve(g))))
    assert paths, "no checkpoint matched"
    sim, deg = load_config("configs/simulation.yaml"), load_config("configs/degradation.yaml")
    device = pick_device("auto")
    models = [load_checkpoint(p, device) for p in paths]
    kind = models[0][1]["kind"]
    assert all(ck["kind"] == kind for _, ck in models)
    te = load_split(a.data, a.split, load_images=kind in ("vision", "fusion"), max_episodes=a.max_episodes)
    hk = HandKinematics(sim) if kind in ("proprio", "fusion") else None
    dg = Degrader(deg, sim)
    target = te["target"].astype(float)
    results = {}
    gated = models[0][1]["cfg"].get("proprio", {}).get("gate_inputs", False)
    mixed = models[0][1]["cfg"].get("mixing", {}).get("enabled", False)
    name = (("F-" + models[0][1]["cfg"]["fusion"]["mode"]) if kind == "fusion" else ("B2" if kind == "vision" else ("B4g" if gated else "B4"))) + ("m" if mixed else "")
    rel = set(relevant_families(kind))                     # a pair is only an 'unseen combination' if this model sees both families
    unseen = {tuple(sorted(p)) for p in models[0][1]["cfg"].get("mixing", {}).get("forbidden_pairs", []) if set(p) <= rel} if mixed else set()
    print(f"{name} ({kind}): {len(models)} seed(s) on {a.split}")
    if a.ablate:
        assert kind == "fusion", "--ablate only applies to fusion models"
        name += "-no:" + "+".join(a.ablate)
    for sname, levels in SCENARIOS.items():
        if a.only and sname not in a.only:
            continue
        sc = Scenario.make(**levels)
        d = dg.apply(te, sc, int(deg["seed"])) if levels else te
        fams = sorted(f for f, _ in sc.levels)
        tag = "  [UNSEEN COMBINATION]" if any(tuple(sorted(p)) in unseen for p in itertools.combinations(fams, 2)) else ""
        print(f"\n=== {sname}  ({sc.name}){tag}\n{'estimator':<14}" + "".join(f"{s:>20}" for s in STAGE_NAMES))
        per_seed = []
        for model, ck in models:
            inp = make_inputs(kind, d, ck["cfg"], hk)
            est = predict(model, ablate_inputs(inp, a.ablate) if a.ablate else inp, device)
            per_seed.append({sn: summarize(est, target, te["stage"] == si) for si, sn in enumerate(STAGE_NAMES)})
        agg = {sn: {k: {"mean": float(np.mean([r[sn][k] for r in per_seed])), "std": float(np.std([r[sn][k] for r in per_seed]))}
                    for k in per_seed[0][sn]} for sn in STAGE_NAMES}
        results[sname] = {"per_seed": per_seed, "agg": agg}
        row = "".join(f"{agg[s]['rmse_xy_mm']['mean']:6.1f}|{agg[s]['yaw_mae_deg']['mean']:4.1f}|{agg[s]['mean_d2']['mean']:6.1f}".rjust(20)
                      for s in STAGE_NAMES)
        print(f"{name:<14}{row}")
    p = resolve(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"kind": kind, "split": a.split, "ckpts": paths, "results": results}, indent=1))
    print("\n(cells: xy RMSE mm | yaw MAE deg | mean Mahalanobis d2, 4 = calibrated)  saved", p)


if __name__ == "__main__":
    main()

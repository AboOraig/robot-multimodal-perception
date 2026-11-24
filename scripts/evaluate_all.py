"""Final evaluation of every trained model and the classical baselines on one split, saving PER-EPISODE sufficient statistics.

    python scripts/evaluate_all.py --runs-root F:/runs/protocol --split test     --out F:/results/final
    python scripts/evaluate_all.py --runs-root F:/runs/protocol --split test_ood --out F:/results/final

* Each scenario is degraded ONCE (deterministic, seed from configs/degradation.yaml) and every model sees the identical data.
* Output: <out>/<split>/s<NN>.npz per scenario (resumable: existing files are skipped) + <out>/<split>/meta.json.
  Per model/seed: episode stats (n_ep, 6 groups, K sums), reliability counts, error-vs-sigma bins and (UA fusion) mean expert weights.
* Models are the run folders of --runs-root that contain DONE; the model name is the run folder name.
* The test split must only be used for the final numbers: run it once, after docs/preregistration.md is frozen.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from mmperc.baselines.kinematics import HandKinematics
from mmperc.common.config import load_config
from mmperc.data.dataset import load_split, resolve
from mmperc.evaluation.classical import build_classical, oracle_kwargs
from mmperc.evaluation.scenarios import SCENARIOS
from mmperc.evaluation.suffstats import FIELDS, GROUPS, calibration_stats, episode_stats
from mmperc.sensors.degradation import Degrader, Scenario
from mmperc.training.engine import load_checkpoint, pick_device, predict
from mmperc.training.features import make_inputs

SEP = "__"


def discover(runs_root, only=None, seeds=None):
    """{model name: {seed: best.pt path}} for finished runs."""
    out: dict[str, dict[int, str]] = {}
    root = resolve(runs_root)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if only and d.name not in only:
            continue
        for s in sorted(d.glob("seed*")):
            seed = int(s.name[4:])
            if (s / "DONE").exists() and (s / "best.pt").exists() and (seeds is None or seed in seeds):
                out.setdefault(d.name, {})[seed] = str(s / "best.pt")
    return out


def pack(prefix: str, est, target, te, n_ep, aux=None) -> dict[str, np.ndarray]:
    cal = calibration_stats(est, target, te["stage"])
    r = {f"{prefix}{SEP}ep": episode_stats(est, target, te["episode"], te["stage"], n_ep).astype(np.float64),
         f"{prefix}{SEP}reliab": cal["reliab"], f"{prefix}{SEP}counts": cal["counts"], f"{prefix}{SEP}sigma_bins": cal["sigma_bins"]}
    if aux is not None:                                   # mean fusion weight / availability per expert and group
        from mmperc.evaluation.suffstats import _groups
        g = _groups(te["stage"]).astype(float)
        r[f"{prefix}{SEP}w_sum"] = g.T @ aux["w"]                          # (6, 3)
        r[f"{prefix}{SEP}avail_sum"] = g.T @ aux["mask"].astype(float)     # (6, 3)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--split", choices=["test", "test_ood", "val"], default="test")
    ap.add_argument("--out", default="experiments/results/final")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None, help="restrict to these run folders")
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these scenario names")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--no-classical", action="store_true")
    ap.add_argument("--train-episodes", type=int, default=None, help="classical calibration subset (debug only)")
    a = ap.parse_args()

    sim, deg = load_config("configs/simulation.yaml"), load_config("configs/degradation.yaml")
    device = pick_device("auto")
    runs = discover(a.runs_root, a.models, a.seeds)
    assert runs or not a.no_classical, f"no finished run (DONE) under {a.runs_root}"
    out = resolve(a.out) / a.split
    out.mkdir(parents=True, exist_ok=True)
    te = load_split(a.data, a.split, load_images=True, max_episodes=a.max_episodes)
    n_ep = int(te["episode"].max()) + 1
    target = te["target"].astype(float)
    hk, dg = HandKinematics(sim), Degrader(deg, sim)
    est_classical = {}
    if not a.no_classical:
        tr = load_split(a.data, "train", load_images=True, max_episodes=a.train_episodes)
        est_classical, _ = build_classical(sim, tr)
        del tr
    cks = {name: {s: load_checkpoint(p, device) for s, p in seeds.items()} for name, seeds in runs.items()}
    names = list(SCENARIOS)
    meta = {"split": a.split, "data": str(a.data), "scenarios": names, "fields": FIELDS, "groups": GROUPS, "n_episodes": n_ep,
            "ep_id": te["ep_id"][np.r_[0, np.flatnonzero(np.diff(te["episode"])) + 1]].tolist(),
            "models": {n: sorted(s) for n, s in runs.items()}, "classical": list(est_classical) + [f"{k} [oracle]" for k in est_classical if k.startswith(("B3", "B6"))],
            "deg_seed": int(deg["seed"])}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{a.split}: {n_ep} episodes, {len(te['target'])} frames | learned runs: " + ", ".join(f"{n}x{len(s)}" for n, s in runs.items()))
    for si, sname in enumerate(names):
        f = out / f"s{si:02d}.npz"
        if (a.only and sname not in a.only) or f.exists():
            continue
        t0 = time.time()
        levels = SCENARIOS[sname]
        sc = Scenario.make(**levels)
        d = dg.apply(te, sc, int(deg["seed"])) if levels else te
        res: dict[str, np.ndarray] = {}
        cache: dict[str, dict] = {}
        for name, per_seed in cks.items():
            for seed, (model, ck) in per_seed.items():
                cfg = ck["cfg"]
                key = json.dumps([ck["kind"], cfg["proprio"], cfg.get("fusion", {}).get("use_contact", True)], sort_keys=True)
                if key not in cache:
                    cache[key] = make_inputs(ck["kind"], d, cfg, hk if ck["kind"] != "vision" else None)
                est, aux = predict(model, cache[key], device, return_aux=True)
                res.update(pack(f"{name}{SEP}{seed}", est, target, te, n_ep, aux))
            cache.clear()
        for ename, fn in est_classical.items():
            res.update(pack(f"{ename}{SEP}0", fn(d, {}), target, te, n_ep))
            if ename.startswith(("B3", "B6")):
                res.update(pack(f"{ename} [oracle]{SEP}0", fn(d, oracle_kwargs(dg, sc, deg)), target, te, n_ep))
        np.savez_compressed(f, **res)
        print(f"[{si + 1}/{len(names)}] {sname:<42} {time.time() - t0:5.0f}s  -> {f.name}", flush=True)
    print("done", out)


if __name__ == "__main__":
    main()

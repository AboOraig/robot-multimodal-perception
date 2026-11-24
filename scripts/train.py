"""Train a unimodal learned baseline with a heteroscedastic Gaussian head.

    python scripts/train.py --kind vision  --seeds 0 1 2
    python scripts/train.py --kind proprio --seeds 0 1 2 --out-root F:/runs
Trains on CLEAN train data, selects on clean val NLL (after the MSE warm-up). One run per seed.
Smoke test:  python scripts/train.py --kind vision --seeds 0 --train-episodes 20 --val-episodes 10 --epochs 2
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from mmperc.baselines.kinematics import HandKinematics
from mmperc.common.config import load_config
from mmperc.data.dataset import load_split, resolve
from mmperc.training.engine import train
from mmperc.sensors.degradation import Degrader
from mmperc.training.features import fit_stats, make_inputs
from mmperc.training.mixing import Mixer
from mmperc.training.naming import run_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["vision", "proprio", "fusion"], required=True)
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--data", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-episodes", type=int, default=None)
    ap.add_argument("--val-episodes", type=int, default=None)
    ap.add_argument("--mode", choices=["concat", "learned", "gated", "poe", "tpoe", "ci"], default="learned", help="fusion only")
    ap.add_argument("--gate-inputs", action="store_true", help="proprio only: inputs are masked unless the holding gate is on (B4g)")
    ap.add_argument("--mixed", action="store_true", help="train under random mixed degradations (unseen combos held out)")
    ap.add_argument("--no-contact", action="store_true", help="fusion ablation: the contact expert never receives data")
    ap.add_argument("--allow-combos", action="store_true", help="mixed training WITHOUT held-out family pairs (control for H3)")
    ap.add_argument("--tag", default="", help="suffix of the run folder name (ablations)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    if a.gate_inputs:
        assert a.kind == "proprio"
        cfg["proprio"]["gate_inputs"] = True
    cfg["mixing"]["enabled"] = bool(a.mixed)
    if a.allow_combos:
        assert a.mixed, "--allow-combos only makes sense with --mixed"
        cfg["mixing"]["forbidden_pairs"] = []
    if a.no_contact:
        assert a.kind == "fusion"
        cfg["fusion"]["use_contact"] = False
    if a.kind == "fusion":
        cfg["fusion"]["mode"] = a.mode
    name = run_name(a.kind, a.mode, a.gate_inputs, a.mixed, a.tag)
    data = a.data or cfg["data_dir"]
    out_root = resolve(a.out_root or cfg["out_root"])
    sim = load_config("configs/simulation.yaml")
    hk = HandKinematics(sim) if a.kind in ("proprio", "fusion") else None
    img = a.kind in ("vision", "fusion")
    tr = load_split(data, "train", load_images=img, max_episodes=a.train_episodes)
    va = load_split(data, "val", load_images=img, max_episodes=a.val_episodes)
    tr_in, va_in = make_inputs(a.kind, tr, cfg, hk), make_inputs(a.kind, va, cfg, hk)
    stats = fit_stats(tr, tr_in)                           # normalisation from CLEAN train data (never from degraded data)
    dg = Degrader(load_config("configs/degradation.yaml"), sim) if a.mixed else None
    for seed in a.seeds:
        out = out_root / name / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({"cfg": cfg, "seed": seed, "data": str(data), "stats": stats}, indent=1))
        kw = {}
        if a.mixed:
            mix = Mixer(a.kind, cfg, dg, lambda k, d, c: make_inputs(k, d, c, hk), seed)
            kw = {"refresh": lambda ep, m=mix: m.train_inputs(tr, ep), "va_clean_in": va_in}
            va_sel = mix.val_inputs(va)                    # fixed mixed-degraded validation set used for model selection
        else:
            va_sel = va_in
        res = train(a.kind, copy.deepcopy(cfg), tr, va, tr_in, va_sel, stats, out, seed, resume=a.resume, **kw)
        print("done", res)


if __name__ == "__main__":
    main()

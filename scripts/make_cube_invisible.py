"""Arm-only control: regenerate episodes with the cube rendered fully TRANSPARENT (physics unchanged).

    python scripts/make_cube_invisible.py --out data/v1_nocube --limit 40

Same seeds and configuration as the main dataset except cube alpha = 0, so every episode has the same cube pose and robot
trajectory as in data/v1 but the image contains no cube. A vision model that still localises the cube on these images is
reading it from the ARM (policy leakage) or the scene, not from cube pixels. Evaluate with:
    python scripts/evaluate.py --ckpt "experiments/runs/vision_mixed/seed*/best.pt" --data data/v1_nocube --split val
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from mmperc.common.config import load_config, repo_root
from mmperc.data.generate import generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v1_nocube")
    ap.add_argument("--limit", type=int, default=40, help="episodes per split to generate")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    sim, ds = load_config("configs/simulation.yaml"), load_config("configs/dataset.yaml")
    sim = copy.deepcopy(sim)
    sim["cube"]["rgba"] = list(sim["cube"]["rgba"][:3]) + [0.0]
    out = Path(a.out) if Path(a.out).is_absolute() else repo_root() / a.out
    m = generate_dataset(sim, ds, out, workers=a.workers, limit=a.limit)
    print(json.dumps({"out_dir": str(out), "episodes_on_disk": m["n_episodes"]}, indent=1))


if __name__ == "__main__":
    main()

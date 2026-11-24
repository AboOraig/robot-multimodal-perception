"""Generate the multimodal dataset (resumable: existing episode files are skipped).

    python scripts/generate_dataset.py --workers 4
    python scripts/generate_dataset.py --limit 5 --out data/smoke     # 5 episodes per split
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mmperc.common.config import load_config, repo_root
from mmperc.data.generate import generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-config", default="configs/simulation.yaml")
    ap.add_argument("--ds-config", default="configs/dataset.yaml")
    ap.add_argument("--out", default=None, help="override dataset out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes of each split")
    ap.add_argument("--no-images", action="store_true", help="skip rendering (fast; for diagnostics)")
    args = ap.parse_args()

    sim_cfg, ds_cfg = load_config(args.sim_config), load_config(args.ds_config)
    if args.no_images:
        ds_cfg["store_images"] = False
    out = Path(args.out or ds_cfg["out_dir"])
    if not out.is_absolute():
        out = repo_root() / out
    m = generate_dataset(sim_cfg, ds_cfg, out, workers=args.workers, limit=args.limit)
    succ = [v["success"] for v in m["episodes"].values()]
    print(json.dumps({"out_dir": str(out), "episodes_on_disk": m["n_episodes"],
                      "grasp_success": f"{sum(succ)}/{len(succ)}",
                      "run_status": m["last_run_status_counts"],
                      "rejected_by_leakage_guard": len(m["rejected_ids"])}, indent=1))
    print("split diagnostics:", json.dumps(m["split_diagnostics"], indent=1))


if __name__ == "__main__":
    main()

"""Run the full training protocol: every model in RUNS x every seed, sequentially, resumable.

    python scripts/run_protocol.py --seeds 0 1 2 --out-root F:/runs/protocol
    python scripts/run_protocol.py --dry-run                          # print the plan and the config fingerprint
    python scripts/run_protocol.py --seeds 0 --only fusion_ci_mixed   # a single run

Order is SEED-MAJOR: all runs of seed 0, then all runs of seed 1, ... so stopping after any seed leaves a complete
experiment with fewer seeds. Each run is a separate `scripts/train.py` process (isolated memory, a crash loses one run).
A finished run has a DONE file; an interrupted run is resumed from its last checkpoint. All configuration files are
fingerprinted into <out-root>/protocol_manifest.json: changing a config in the middle of the protocol is refused
(--force overrides), so that every model in the comparison was trained under identical settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time

from mmperc.common.config import repo_root
from mmperc.data.dataset import resolve

# name (= run folder written by train.py) -> train.py arguments.  Priority order: the first block answers H1/H2/H3.
RUNS: dict[str, list[str]] = {
    "vision_mixed":                    ["--kind", "vision", "--mixed"],
    "proprio_gated_mixed":             ["--kind", "proprio", "--gate-inputs", "--mixed"],
    "fusion_learned_mixed":            ["--kind", "fusion", "--mode", "learned", "--mixed"],
    "fusion_ci_mixed":                 ["--kind", "fusion", "--mode", "ci", "--mixed"],
    "fusion_gated_mixed":              ["--kind", "fusion", "--mode", "gated", "--mixed"],
    "fusion_concat_mixed":             ["--kind", "fusion", "--mode", "concat", "--mixed"],
    "fusion_poe_mixed":                ["--kind", "fusion", "--mode", "poe", "--mixed"],
    "fusion_tpoe_mixed":               ["--kind", "fusion", "--mode", "tpoe", "--mixed"],
    # H3 and ablations
    "vision":                          ["--kind", "vision"],
    "fusion_ci":                       ["--kind", "fusion", "--mode", "ci"],
    "fusion_ci_mixed_allcombos":       ["--kind", "fusion", "--mode", "ci", "--mixed", "--allow-combos", "--tag", "_allcombos"],
    "fusion_ci_mixed_nocontact":       ["--kind", "fusion", "--mode", "ci", "--mixed", "--no-contact", "--tag", "_nocontact"],
    "proprio_gated":                   ["--kind", "proprio", "--gate-inputs"],
    "fusion_learned":                  ["--kind", "fusion", "--mode", "learned"],
}
CONFIGS = ("configs/training.yaml", "configs/degradation.yaml", "configs/simulation.yaml", "configs/dataset.yaml")


def fingerprint(data_dir: str) -> dict:
    root = repo_root()
    fp = {c: hashlib.sha256((root / c).read_bytes()).hexdigest()[:16] for c in CONFIGS}
    man = resolve(data_dir) / "manifest.json"
    fp["dataset_manifest"] = hashlib.sha256(man.read_bytes()).hexdigest()[:16] if man.exists() else "missing"
    return fp


def plan(seeds, only, skip_done_in=None):
    out = []
    for s in seeds:
        for name, args in RUNS.items():
            if only and name not in only:
                continue
            out.append((s, name, args))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-root", default="experiments/protocol")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--only", nargs="+", default=None, choices=list(RUNS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed run (default: stop)")
    ap.add_argument("--force", action="store_true", help="accept a changed config fingerprint")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args for train.py (e.g. --epochs 1 for a smoke test)")
    a = ap.parse_args()

    out_root = resolve(a.out_root)
    fp = fingerprint(a.data)
    mpath = out_root / "protocol_manifest.json"
    todo = plan(a.seeds, a.only)
    print(f"config fingerprint: {json.dumps(fp)}\nout root: {out_root}\n{len(todo)} runs ({len(a.seeds)} seeds x {len(todo) // max(1, len(a.seeds))} models)")
    for s, name, args in todo:
        done = (out_root / name / f"seed{s}" / "DONE").exists()
        print(f"  seed {s}  {name:<28} {'DONE' if done else 'todo'}   train.py {' '.join(args)}")
    if a.dry_run:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    if mpath.exists():
        old = json.loads(mpath.read_text())["fingerprint"]
        if old != fp and not a.force:
            sys.exit(f"REFUSED: configs/dataset changed since the protocol started.\n old {old}\n new {fp}\n"
                     "Mixing models trained under different settings invalidates the comparison. Use --force only if intended.")
    else:
        mpath.write_text(json.dumps({"fingerprint": fp, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version}, indent=1))
    log = out_root / "protocol_log.jsonl"
    failed = []
    for s, name, args in todo:
        d = out_root / name / f"seed{s}"
        if (d / "DONE").exists():
            continue
        cmd = [sys.executable, str(repo_root() / "scripts" / "train.py"), *args, "--seeds", str(s), "--data", a.data,
               "--out-root", str(out_root), "--resume", *a.extra]
        print(f"\n>>> seed {s} {name}: {' '.join(cmd[2:])}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=repo_root()).returncode
        rec = {"run": name, "seed": s, "rc": rc, "minutes": round((time.time() - t0) / 60, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if rc == 0 and (d / "best.pt").exists():
            (d / "DONE").write_text(json.dumps(rec))
            print(f"<<< finished in {rec['minutes']} min", flush=True)
        else:
            failed.append((s, name))
            print(f"!!! FAILED (rc={rc}): seed {s} {name}", flush=True)
            if not a.keep_going:
                sys.exit(1)
    print("\nprotocol complete" if not failed else f"\nfailed runs: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

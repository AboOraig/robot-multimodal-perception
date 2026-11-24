"""Run one scripted episode and save diagnostic figures (frame montage + signal plots)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmperc.common.config import load_config, repo_root
from mmperc.simulation.episode import run_episode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--config", default="configs/simulation.yaml")
    ap.add_argument("--out", default="experiments/figures")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ep = run_episode(cfg, args.episode)
    out = Path(args.out)
    if not out.is_absolute():
        out = repo_root() / out
    out.mkdir(parents=True, exist_ok=True)
    meta = ep["meta"]
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in meta.items()
           if k in ("episode_id", "success", "lifted_m", "max_tilt_hold_deg", "tcp_cube_gap_m",
                    "grasp_offset_x", "grasp_offset_z", "plan_attempts")})

    n = len(ep["t"])
    idx = np.linspace(0, n - 1, 8).astype(int)
    fig, ax = plt.subplots(2, 4, figsize=(12, 6.4))
    for a, i in zip(ax.ravel(), idx):
        a.imshow(ep["image"][i])
        a.set_title(f"t={ep['t'][i]:.2f}s  {ep['phase_names'][ep['phase'][i]]}", fontsize=8)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"episode_{args.episode:04d}_frames.png", dpi=110)

    t = ep["t"]
    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(t, ep["q"]); ax[0].set_ylabel("arm joints q [rad]")
    ax[1].plot(t, ep["wrench"][:, :3]); ax[1].set_ylabel("wrist force [N]\n(sensor frame)")
    ax[1].legend(["Fx", "Fy", "Fz"], ncol=3, fontsize=8)
    ax[2].plot(t, ep["wrench"][:, 3:]); ax[2].set_ylabel("wrist torque [N m]\n(sensor frame)")
    ax[2].legend(["Tx", "Ty", "Tz"], ncol=3, fontsize=8)
    ax[3].plot(t, ep["cube_pos"][:, 2], label="cube z [m]")
    ax[3].plot(t, ep["n_finger_contacts"] * 0.02, label="fingers touching x 0.02")
    ax[3].set_ylabel("ground truth"); ax[3].set_xlabel("time [s]"); ax[3].legend(fontsize=8)
    for a in ax:
        for p in range(len(ep["phase_names"])):
            m = ep["phase"] == p
            if m.any():
                a.axvspan(t[m][0], t[m][-1], color=f"C{p}", alpha=0.07)
    ax[0].set_title("Phases: " + ", ".join(f"C{i}={n}" for i, n in enumerate(ep["phase_names"])), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"episode_{args.episode:04d}_signals.png", dpi=110)
    print("saved figures to", out)


if __name__ == "__main__":
    main()

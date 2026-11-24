"""Calibrate severity levels against measured signal scales and save example images.

Reports, per family and level:
  vision  : PSNR vs clean [dB]; fraction of the cube's red pixels still detectable (colour threshold)
  proprio : induced TCP position error [mm] through forward kinematics; realised dropout statistics
  contact : noise / bias as a fraction of the measured hold-signal (cube weight, in-hand torque)
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmperc.common.config import load_config, repo_root
from mmperc.data.dataset import load_split, resolve
from mmperc.sensors.degradation import FAMILIES, Degrader, Scenario
from mmperc.simulation.kinematics import ArmIK
from mmperc.simulation.scene import build_model


def red_count(img):
    i = img.astype(np.int16)
    return ((i[..., 0] > 150) & (i[..., 1] < 90) & (i[..., 2] < 90)).sum(axis=(1, 2))


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-episodes", type=int, default=30)
    ap.add_argument("--out", default="experiments/results/degradation_calibration.json")
    ap.add_argument("--fig", default="experiments/figures/degradation_examples.png")
    a = ap.parse_args()

    sim, deg = load_config("configs/simulation.yaml"), load_config("configs/degradation.yaml")
    D = load_split(a.data, a.split, load_images=True, max_episodes=a.max_episodes)
    dg = Degrader(deg, sim)
    ik = ArmIK(build_model(sim))
    seed = int(deg["seed"])
    clean_red = red_count(D["image"])
    ok = clean_red >= 15
    tcp_of = lambda Q: np.array([ik.fk(q)[0] for q in Q])
    tcp_clean = tcp_of(D["q"])
    coupled = D["stage"] == 3
    # contact signal scales actually present in the data (hold minus free hand, per episode)
    dF, dT = [], []
    for e in np.unique(D["episode"]):
        m = D["episode"] == e
        w = D["wrench"][m]
        c = coupled[m]
        dF.append(np.linalg.norm(w[c][-5:, :3].mean(0) - w[0, :3]))
        dT.append(np.linalg.norm(w[c][-5:, 3:5].mean(0) - w[0, 3:5]))
    sigF, sigT = float(np.median(dF)), float(np.median(dT))
    print(f"contact signal scales (median over episodes): dF = {sigF:.2f} N, dT_xy = {sigT:.3f} N m")
    res = {"split": a.split, "n_episodes": int(len(np.unique(D['episode']))), "signal_dF_N": sigF,
           "signal_dT_Nm": sigT, "families": {}}
    cw = 0.5 * D["cube_pos"][:, 2]                                  # unused, placeholder for clarity

    for fam in FAMILIES:
        rows = []
        for lvl in range(5):
            X = dg.apply(D, Scenario.make(**{fam: lvl}), seed)
            row = {"level": lvl, "value": deg["families"][fam]["levels"][lvl]}
            if fam.startswith("vision"):
                row["psnr_db"] = psnr(X["image"], D["image"])
                row["cube_px_visible_frac"] = float(np.mean(red_count(X["image"])[ok] / clean_red[ok]))
            elif fam.startswith("proprio"):
                m = X["proprio_mask"]
                if fam == "proprio_dropout":
                    row["dropout_rate"] = float(1 - m.mean())
                else:
                    err = np.linalg.norm(tcp_of(X["q"]) - tcp_clean, axis=1) * 1e3
                    row["tcp_err_mm_rms"] = float(np.sqrt((err ** 2).mean()))
            else:
                m = X["contact_mask"]
                if fam == "contact_dropout":
                    row["dropout_rate"] = float(1 - m.mean())
                else:
                    d = X["wrench"] - D["wrench"]
                    row["force_err_rms_N"] = float(np.sqrt((d[:, :3] ** 2).sum(1).mean()))
                    row["torque_err_rms_Nm"] = float(np.sqrt((d[:, 3:5] ** 2).sum(1).mean()))
                    row["force_err_over_hold_signal"] = row["force_err_rms_N"] / sigF
                    row["torque_err_over_hold_signal"] = row["torque_err_rms_Nm"] / sigT
            rows.append(row)
        res["families"][fam] = rows
        print(f"\n{fam}  [{deg['families'][fam]['unit']}]")
        for r in rows:
            print("  L%d  %-8s " % (r["level"], r["value"]) + "  ".join(f"{k}={v:.3g}" for k, v in r.items() if k not in ("level", "value")))

    p = resolve(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))

    # example grid: one 'grasp'-stage frame, vision families x levels
    k = int(np.flatnonzero(D["stage"] == 2)[3])
    vf = [f for f in FAMILIES if f.startswith("vision")]
    fig, ax = plt.subplots(len(vf), 5, figsize=(10, 2.1 * len(vf)))
    for i, fam in enumerate(vf):
        for lvl in range(5):
            sl = np.flatnonzero(D["episode"] == D["episode"][k])
            X = dg.apply({kk: vv[sl] for kk, vv in D.items() if isinstance(vv, np.ndarray) and len(vv) == len(D["t"])},
                         Scenario.make(**{fam: lvl}), seed)
            ax[i, lvl].imshow(X["image"][k - sl[0]]); ax[i, lvl].axis("off")
            ax[i, lvl].set_title(f"{fam.replace('vision_', '')} L{lvl}", fontsize=7)
    fig.tight_layout()
    fp = resolve(a.fig)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=130)
    print("\nsaved", p, "and", fp)


if __name__ == "__main__":
    main()

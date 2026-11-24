"""Policy-leakage diagnostic: how well do joint angles alone predict the cube pose, per segment?

A small MLP is trained on proprioception only (joint positions + finger positions, no time, no
phase) to regress the cube xy and yaw. Evaluation is on validation episodes (and optionally the
OOD test split), reported per scripted segment. Skill is R^2 w.r.t. predicting the train mean:
  R^2 = 1 - MSE / Var(target in that segment)   (0 = no information, 1 = perfect)
Yaw is reported as mean absolute error of yaw modulo 90 deg (chance for uniform yaw = 22.5 deg).

High skill in segments where the hand is NOT yet physically coupled to the cube (approach) means
the scripted policy leaks the cube pose into the joint angles. This is a property of the benchmark,
not of any model, and must be reported next to every proprioception result.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from mmperc.common.config import repo_root
from mmperc.common.seeding import set_global_seed
from mmperc.data.dataset import SEGMENT_NAMES, load_split, resolve


def features(d, with_velocity):
    x = [d["q"], d["finger_q"]]
    if with_velocity:
        x.append(d["qd"])
    return np.concatenate(x, axis=1).astype(np.float32)


def targets(d):
    return np.column_stack([d["cube_pos"][:, :2], np.sin(d["cube_phi"]), np.cos(d["cube_phi"])]).astype(np.float32)


def fit(Xtr, Ytr, epochs, seed, device):
    set_global_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-6
    X = torch.tensor((Xtr - mu) / sd, device=device)
    Y = torch.tensor((Ytr - ymu) / ysd, device=device)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                        nn.Linear(128, 4)).to(device)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g).to(device)
        for i in range(0, len(X), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        sched.step()

    def predict(Xte):
        with torch.no_grad():
            return net(torch.tensor((Xte - mu) / sd, device=device)).cpu().numpy() * ysd + ymu
    return predict


def report(name, d, pred, tr_mean_xy):
    Y = targets(d)
    rows = {}
    for si, sn in enumerate(SEGMENT_NAMES):
        m = d["segment"] == si
        if not m.any():
            continue
        err_xy = pred[m, :2] - Y[m, :2]
        mse = (err_xy ** 2).sum(1).mean()
        var = ((Y[m, :2] - tr_mean_xy) ** 2).sum(1).mean()          # mean-predictor MSE
        phi_hat = np.arctan2(pred[m, 2], pred[m, 3])
        dphi = np.angle(np.exp(1j * (phi_hat - d["cube_phi"][m])))
        rows[sn] = {"n": int(m.sum()), "xy_rmse_mm": float(np.sqrt(mse) * 1e3),
                    "xy_rmse_mean_predictor_mm": float(np.sqrt(var) * 1e3),
                    "xy_R2": float(1 - mse / var), "yaw_mae_deg": float(np.degrees(np.abs(dphi)).mean() / 4)}
    print(f"\n[{name}]  segment            n   xy RMSE   mean-pred  R2(xy)  yaw MAE (chance 22.5)")
    for sn, r in rows.items():
        print(f"  {sn:<14} {r['n']:>7}  {r['xy_rmse_mm']:7.1f}mm  {r['xy_rmse_mean_predictor_mm']:7.1f}mm  "
              f"{r['xy_R2']:6.2f}  {r['yaw_mae_deg']:6.1f} deg")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-velocity", action="store_true")
    ap.add_argument("--out", default="experiments/results/leakage.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = load_split(a.data, "train", load_images=False)
    Xtr, Ytr = features(tr, a.with_velocity), targets(tr)
    predict = fit(Xtr, Ytr, a.epochs, a.seed, dev)
    out = {"with_velocity": a.with_velocity, "epochs": a.epochs, "seed": a.seed}
    for split in ("val", "test_ood"):
        d = load_split(a.data, split, load_images=False)
        out[split] = report(split, d, predict(features(d, a.with_velocity)), Ytr[:, :2].mean(0))
    p = resolve(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("\nsaved", p)


if __name__ == "__main__":
    main()

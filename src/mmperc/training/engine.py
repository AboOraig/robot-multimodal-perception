"""Training / prediction engine: batching, AMP, warm-up -> NLL, checkpointing, JSONL logging."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from mmperc.baselines.common import Estimate
from mmperc.common.seeding import set_global_seed
from mmperc.evaluation.metrics import summarize
from mmperc.models.gaussian import covariance, gaussian_nll, mse_warmup
from mmperc.models.unimodal import build_model


def pick_device(name: str = "auto") -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _to(batch: dict, device) -> dict:
    return {k: torch.as_tensor(v).to(device, non_blocking=True) for k, v in batch.items()}


def iterate(inputs: dict, bs: int, order: np.ndarray):
    for i in range(0, len(order), bs):
        idx = np.sort(order[i:i + bs])                    # sorted index -> faster gather from big arrays
        yield idx, {k: v[idx] for k, v in inputs.items()}


@torch.no_grad()
def predict(model, inputs: dict, device, bs: int = 512, return_aux: bool = False):
    """Estimate for every row. With return_aux=True also returns {'w': (N,E), 'mask': (N,E)} for uncertainty-aware fusion
    models (fusion weights and expert availability), else None for the aux part."""
    model.eval()
    n = len(next(iter(inputs.values())))
    means, covs, ws, ms = [], [], [], []
    for _, b in iterate(inputs, bs, np.arange(n)):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            m, L = model(_to(b, device))
        means.append(m.float().cpu().numpy()); covs.append(covariance(L).double().cpu().numpy())
        last = getattr(model, "last", None)
        if return_aux and last and "w" in last:
            ws.append(last["w"].float().cpu().numpy()); ms.append(last["mask"].cpu().numpy())
    est = Estimate(np.concatenate(means).astype(np.float64), np.concatenate(covs), np.ones(n, bool))
    if not return_aux:
        return est
    return est, ({"w": np.concatenate(ws), "mask": np.concatenate(ms)} if ws else None)


def save_checkpoint(path, model, stats, cfg, kind, seed, epoch, opt=None, sched=None, scaler=None, extra=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "stats": stats, "cfg": cfg, "kind": kind, "seed": seed, "epoch": epoch,
                "opt": opt.state_dict() if opt else None, "sched": sched.state_dict() if sched else None,
                "scaler": scaler.state_dict() if scaler else None, "extra": extra or {}}, tmp)
    tmp.replace(path)                                      # atomic


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["kind"], ck["stats"], ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def train(kind: str, cfg: dict, train_d: dict, val_d: dict, tr_in: dict, va_in: dict, stats: dict, out_dir, seed: int,
          resume: bool = False, log=print, stop_after: int | None = None, refresh=None, va_clean_in: dict | None = None) -> dict:
    t = cfg["train"]
    set_global_seed(seed)
    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    device = pick_device(cfg.get("device", "auto"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model = build_model(kind, stats, cfg).to(device)
    lr = cfg.get("fusion", {}).get("lr", t["lr"]) if kind == "fusion" else t["lr"]       # fusion models: lower peak LR (warm-up spikes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=t["weight_decay"])
    n = len(train_d["target"]); spe = math.ceil(n / t["batch_size"]); total = spe * t["epochs"]
    warm = spe                                             # 1 epoch linear LR warm-up, then cosine
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    target = torch.as_tensor(train_d["target"], dtype=torch.float32)
    scale = torch.cat([torch.as_tensor(stats["pos_std"], dtype=torch.float32), torch.ones(1)]).to(device)
    start, best = 0, float("inf")
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start = ck["epoch"] + 1; best = ck["extra"].get("best", best)
        log(f"resumed from epoch {start}")
    rng = np.random.default_rng([seed, 12345])
    for _ in range(start):                                 # keep the shuffling stream identical after a resume
        rng.permutation(n)
    val_t = val_d["target"].astype(np.float64)
    bad = 0
    for ep in range(start, t["epochs"] if stop_after is None else min(t["epochs"], stop_after)):
        model.train(); t0 = time.time(); losses = []
        use_nll = ep >= t["warmup_epochs"]
        cur_in = refresh(ep) if refresh is not None else tr_in      # degraded copy of the train inputs (mixed training)
        for idx, b in iterate(cur_in, t["batch_size"], rng.permutation(n)):
            y = target[idx].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                m, L = model(_to(b, device))
            if use_nll:
                loss = gaussian_nll(m, L, y, t["beta_nll"])
            else:                                          # warm-up: mean by MSE; covariance by NLL on a DETACHED mean
                loss = mse_warmup(m, y, scale) + t.get("warmup_cov_weight", 0.1) * gaussian_nll(m.detach(), L, y)
            if hasattr(model, "aux_loss"):                 # per-expert supervision (uncertainty-aware fusion)
                loss = loss + model.aux_weight * model.aux_loss(y, use_nll, scale, t.get("warmup_cov_weight", 0.1))
            if not torch.isfinite(loss):
                bad += 1
                if bad > 20:
                    raise RuntimeError("too many non-finite losses")
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        est = predict(model, va_in, device)
        r = summarize(est, val_t)
        rec = {"epoch": ep, "phase": "nll" if use_nll else "mse", "train_loss": float(np.mean(losses)), "val": r,
               "val_clean": summarize(predict(model, va_clean_in, device), val_t) if va_clean_in is not None else None,
               "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"[{kind} s{seed}] ep {ep:3d} {rec['phase']} loss {rec['train_loss']:8.3f} | val rmse3 {r['rmse3_mm']:6.1f} mm "
            f"yaw {r['yaw_mae_deg']:5.1f} deg nll {r['nll']:7.3f} d2 {r['mean_d2']:6.2f} | {rec['sec']}s")
        score = r["nll"] if use_nll else float("inf")      # select on validation NLL, only after the warm-up
        if score < best:
            best = score
            save_checkpoint(out / "best.pt", model, stats, cfg, kind, seed, ep, extra={"val": r})
        save_checkpoint(out / "last.pt", model, stats, cfg, kind, seed, ep, opt, sched, scaler, {"best": best})
    return {"best_val_nll": best, "out": str(out)}

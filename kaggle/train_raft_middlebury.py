"""
Fine-tune RAFT-Stereo (realtime config) on Middlebury 2014 — Kaggle version.

Self-contained: only needs the RAFT-Stereo repo (cloned in the notebook) and
the Middlebury dataset. Reproduces the thesis training protocol:
  - downsample=2 (deployment resolution), variant=both (perfect+imperfect)
  - seed-42 random split, val_split=0.15  → same val samples as AANet runs
  - D1/EPE metrics identical to the thesis ablation table

Usage (inside Kaggle notebook, after cloning RAFT-Stereo):
    python train_raft_middlebury.py \
        --data-root /kaggle/input/middlebury2014 \
        --raft-dir  /kaggle/working/RAFT-Stereo \
        --ckpt      /kaggle/working/RAFT-Stereo/models/raftstereo-realtime.pth \
        --out       /kaggle/working/ckpt \
        --steps 4000 --batch-size 4 --crop 320 640 --lr 2e-5
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)


# ─────────────────────────── PFM reader ─────────────────────────────────────
def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode().rstrip()
        if header not in ("Pf", "PF"):
            raise ValueError("Not a PFM file")
        dims = f.readline().decode()
        while dims.startswith("#"):
            dims = f.readline().decode()
        w, h = map(int, dims.split())
        scale = float(f.readline().decode().rstrip())
        data = np.fromfile(f, "<f4" if scale < 0 else ">f4", w * h * (3 if header == "PF" else 1))
        data = data.reshape(h, w, -1).squeeze()
        return np.flipud(data).copy()


# ───────────────────────── Middlebury dataset ───────────────────────────────
class Middlebury2014(Dataset):
    """Matches the thesis loader: downsample=2 via decimation, variant=both."""

    def __init__(self, root, crop=None, training=True, downsample=2):
        self.root = Path(root)
        self.crop = crop
        self.training = training
        self.ds = downsample
        self.scenes = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and (d / "im0.png").exists() and (d / "disp0.pfm").exists()
        )
        if not self.scenes:
            raise SystemExit(f"No scenes found under {root}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        s = self.scenes[i]
        L = cv2.cvtColor(cv2.imread(str(s / "im0.png")), cv2.COLOR_BGR2RGB)
        R = cv2.cvtColor(cv2.imread(str(s / "im1.png")), cv2.COLOR_BGR2RGB)
        D = read_pfm(s / "disp0.pfm").astype(np.float32)
        D[~np.isfinite(D)] = 0.0

        # decimate (same as thesis loader — no interpolation of disparity)
        ds = self.ds
        L, R, D = L[::ds, ::ds], R[::ds, ::ds], D[::ds, ::ds] / ds

        if self.training:
            # photometric jitter, asymmetric between views
            for img in (L, R):
                b = random.uniform(0.8, 1.2); c = random.uniform(0.8, 1.2)
                img[:] = np.clip((img.astype(np.float32) - 127.5) * c + 127.5 * b, 0, 255)
            if self.crop:
                ch, cw = self.crop
                H, W = D.shape
                if H > ch and W > cw:
                    y = random.randint(0, H - ch); x = random.randint(0, W - cw)
                    L, R, D = L[y:y+ch, x:x+cw], R[y:y+ch, x:x+cw], D[y:y+ch, x:x+cw]
                else:  # scene smaller than crop — pad
                    L = np.pad(L, ((0, max(0, ch-H)), (0, max(0, cw-W)), (0, 0)))
                    R = np.pad(R, ((0, max(0, ch-H)), (0, max(0, cw-W)), (0, 0)))
                    D = np.pad(D, ((0, max(0, ch-H)), (0, max(0, cw-W))))
                    L, R, D = L[:ch, :cw], R[:ch, :cw], D[:ch, :cw]

        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float()
        return {"left": to_t(L), "right": to_t(R),
                "disp": torch.from_numpy(np.ascontiguousarray(D)).float(),
                "valid": torch.from_numpy((D > 0).astype(np.float32))}


# ───────────────────────────── metrics ──────────────────────────────────────
def epe_d1(pred, gt, valid):
    mask = valid > 0.5
    if mask.sum() == 0:
        return 0.0, 0.0
    err = (pred[mask] - gt[mask]).abs()
    epe = err.mean().item()
    d1 = ((err > 3.0) & (err / gt[mask].clamp_min(1e-3) > 0.05)).float().mean().item() * 100
    return epe, d1


# ─────────────────────────── RAFT loss ──────────────────────────────────────
def sequence_loss(flow_preds, disp_gt, valid, gamma=0.9, max_disp=700):
    """L1 over all iterative predictions, exponentially weighted (RAFT-Stereo)."""
    flow_gt = -disp_gt.unsqueeze(1)                        # RAFT predicts -disp
    mask = (valid > 0.5) & (disp_gt < max_disp)
    mask = mask.unsqueeze(1)
    n = len(flow_preds)
    loss = 0.0
    for i, fp in enumerate(flow_preds):
        w = gamma ** (n - i - 1)
        loss = loss + w * ((fp - flow_gt).abs())[mask].mean()
    return loss


# ─────────────────────────────── main ───────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--raft-dir",  required=True)
    p.add_argument("--ckpt",      required=True, help="Pretrained RAFT weights to start from")
    p.add_argument("--out",       default="./ckpt")
    p.add_argument("--steps",       type=int, default=4000)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--crop", nargs=2, type=int, default=[320, 640])
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--train-iters", type=int, default=16)
    p.add_argument("--valid-iters", type=int, default=16)
    p.add_argument("--val-split",   type=float, default=0.15)
    p.add_argument("--val-every",   type=int, default=500)
    # AMP on by default (fine on Kaggle T4/P100). Use --no-mixed-precision on
    # GPUs with broken fp16 (GTX 16xx).
    p.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    sys.path.insert(0, args.raft_dir)
    sys.path.insert(0, str(Path(args.raft_dir) / "core"))
    from raft_stereo import RAFTStereo
    from utils.utils import InputPadder

    device = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ── data: same split protocol as the thesis (seed 42) ────────────────────
    full_train = Middlebury2014(args.data_root, crop=tuple(args.crop), training=True)
    full_val   = Middlebury2014(args.data_root, crop=None, training=False)
    n = len(full_train)
    n_val = max(1, int(n * args.val_split))
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[n - n_val:], perm[:n - n_val]
    # NOTE: torch random_split assigns the LAST n_val of the permutation to val
    train_set = torch.utils.data.Subset(full_train, train_idx)
    val_set   = torch.utils.data.Subset(full_val, val_idx)
    print(f"train {len(train_set)}  val {len(val_set)}  scenes")

    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    # ── model (realtime config) ───────────────────────────────────────────────
    raft_args = argparse.Namespace(
        hidden_dims=[128, 128, 128], corr_implementation="reg",
        corr_levels=4, corr_radius=4, context_norm="batch",
        mixed_precision=args.mixed_precision,
        shared_backbone=True, n_downsample=3, n_gru_layers=2, slow_fast_gru=True)
    model = RAFTStereo(raft_args)
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(device)
    model.train()
    model.freeze_bn()      # standard for RAFT fine-tuning

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, eps=1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=args.steps + 100, pct_start=0.01,
        cycle_momentum=False, anneal_strategy="linear")
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    def validate():
        model.eval()
        epes, d1s = [], []
        for i in range(len(val_set)):
            s = val_set[i]
            L = s["left"][None].to(device); R = s["right"][None].to(device)
            padder = InputPadder(L.shape, divis_by=32)
            Lp, Rp = padder.pad(L, R)
            with torch.no_grad():
                _, flow = model(Lp, Rp, iters=args.valid_iters, test_mode=True)
            pred = -padder.unpad(flow).squeeze()
            e, d = epe_d1(pred.cpu(), s["disp"], s["valid"])
            epes.append(e); d1s.append(d)
        model.train()
        model.freeze_bn()
        return float(np.mean(epes)), float(np.mean(d1s))

    epe0, d10 = validate()
    print(f"[step 0 / zero-shot]  EPE {epe0:.3f}  D1 {d10:.2f}%")
    best_d1 = d10
    torch.save(model.state_dict(), out / "best.pth")

    step, t0 = 0, time.time()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            L = batch["left"].to(device); R = batch["right"].to(device)
            gt = batch["disp"].to(device); valid = batch["valid"].to(device)

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                preds = model(L, R, iters=args.train_iters)
            loss = sequence_loss(preds, gt, valid)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            sched.step()
            scaler.update()
            step += 1

            if step % 50 == 0:
                print(f"step {step:5d}  loss {loss.item():.3f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"{(time.time()-t0)/50:.2f}s/step")
                t0 = time.time()

            if step % args.val_every == 0:
                epe, d1 = validate()
                mark = ""
                if d1 < best_d1:
                    best_d1 = d1
                    torch.save(model.state_dict(), out / "best.pth")
                    mark = "  ← saved best"
                print(f"[val @ {step}]  EPE {epe:.3f}  D1 {d1:.2f}%{mark}")

    epe, d1 = validate()
    if d1 < best_d1:
        best_d1 = d1
        torch.save(model.state_dict(), out / "best.pth")
    torch.save(model.state_dict(), out / "last.pth")
    print(f"\nDone. best D1 = {best_d1:.2f}%  (zero-shot was {d10:.2f}%)")
    print(f"Checkpoints: {out}/best.pth  {out}/last.pth")


if __name__ == "__main__":
    main()

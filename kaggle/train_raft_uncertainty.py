"""
Phase-2 uncertainty training for RAFT-Stereo — Kaggle version.

Starts from the fine-tuned disparity model (raft_middlebury_ft/best.pth),
adds a Laplace uncertainty head on the finest GRU hidden state, FREEZES the
base network, and trains only the head with the NLL loss
    |d - d*| / b  +  log(2b)
— the same recipe as the thesis AANet uncertainty phase. Disparity output
(and hence D1) is unchanged by construction; the head learns to predict
where the frozen model errs.

Validation reports NLL and a sparsification check: D1 over the most
confident 100% / 50% / 20% of pixels. If the uncertainty is informative,
D1 must drop monotonically as low-confidence pixels are removed.

Usage (Kaggle, after cloning RAFT-Stereo — see README.md):
    python train_raft_uncertainty.py \
        --data-root /kaggle/input/middlebury2014 \
        --raft-dir  /kaggle/working/RAFT-Stereo \
        --ckpt      /kaggle/input/YOUR-CKPT-DATASET/best.pth \
        --out       /kaggle/working/ckpt_unc \
        --steps 2000 --batch-size 4 --crop 320 640 --lr 1e-4
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
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
        return np.flipud(data.reshape(h, w, -1).squeeze()).copy()


# ───────────────────────── Middlebury dataset ───────────────────────────────
class Middlebury2014(Dataset):
    def __init__(self, root, crop=None, training=True, downsample=2):
        self.root = Path(root)
        self.crop = crop
        self.training = training
        self.ds = downsample
        self.scenes = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and (d / "im0.png").exists() and (d / "disp0.pfm").exists())
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
        ds = self.ds
        L, R, D = L[::ds, ::ds], R[::ds, ::ds], D[::ds, ::ds] / ds

        if self.training:
            for img in (L, R):
                b = random.uniform(0.8, 1.2); c = random.uniform(0.8, 1.2)
                img[:] = np.clip((img.astype(np.float32) - 127.5) * c + 127.5 * b, 0, 255)
            if self.crop:
                ch, cw = self.crop
                H, W = D.shape
                if H > ch and W > cw:
                    y = random.randint(0, H - ch); x = random.randint(0, W - cw)
                    L, R, D = L[y:y+ch, x:x+cw], R[y:y+ch, x:x+cw], D[y:y+ch, x:x+cw]
                else:
                    L = np.pad(L, ((0, max(0, ch-H)), (0, max(0, cw-W)), (0, 0)))
                    R = np.pad(R, ((0, max(0, ch-H)), (0, max(0, cw-W)), (0, 0)))
                    D = np.pad(D, ((0, max(0, ch-H)), (0, max(0, cw-W))))
                    L, R, D = L[:ch, :cw], R[:ch, :cw], D[:ch, :cw]

        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float()
        return {"left": to_t(L), "right": to_t(R),
                "disp": torch.from_numpy(np.ascontiguousarray(D)).float(),
                "valid": torch.from_numpy((D > 0).astype(np.float32))}


# ───────────────────── uncertainty model (subclass) ─────────────────────────
def build_model(raft_dir, mixed_precision):
    sys.path.insert(0, str(raft_dir))
    sys.path.insert(0, str(Path(raft_dir) / "core"))
    from raft_stereo import RAFTStereo, autocast
    from core.corr import CorrBlock1D

    args = argparse.Namespace(
        hidden_dims=[128, 128, 128], corr_implementation="reg",
        corr_levels=4, corr_radius=4, context_norm="batch",
        mixed_precision=mixed_precision,
        shared_backbone=True, n_downsample=3, n_gru_layers=2, slow_fast_gru=True)

    class RAFTStereoUncertainty(RAFTStereo):
        def __init__(self, a):
            super().__init__(a)
            hd = a.hidden_dims[0]
            self.unc_head = nn.Sequential(
                nn.Conv2d(hd, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 3, padding=1))

        def freeze_base(self):
            for p in self.parameters():
                p.requires_grad = False
            for p in self.unc_head.parameters():
                p.requires_grad = True

        def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
            image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
            image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

            with autocast(enabled=self.args.mixed_precision):
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0),
                                          dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
                net_list = [torch.tanh(y[0]) for y in cnet_list]
                inp_list = [torch.relu(y[1]) for y in cnet_list]
                inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1))
                            for i, conv in zip(inp_list, self.context_zqr_convs)]

            fmap1, fmap2 = fmap1.float(), fmap2.float()
            corr_fn = CorrBlock1D(fmap1, fmap2, radius=self.args.corr_radius,
                                  num_levels=self.args.corr_levels)
            coords0, coords1 = self.initialize_flow(net_list[0])
            if flow_init is not None:
                coords1 = coords1 + flow_init

            flow_predictions, flow_up = [], None
            for itr in range(iters):
                coords1 = coords1.detach()
                corr = corr_fn(coords1)
                flow = coords1 - coords0
                with autocast(enabled=self.args.mixed_precision):
                    if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:
                        net_list = self.update_block(net_list, inp_list,
                                                     iter32=self.args.n_gru_layers == 3,
                                                     iter16=True, iter08=False, update=False)
                    net_list, up_mask, delta_flow = self.update_block(
                        net_list, inp_list, corr, flow,
                        iter32=self.args.n_gru_layers == 3,
                        iter16=self.args.n_gru_layers >= 2)
                delta_flow[:, 1] = 0.0
                coords1 = coords1 + delta_flow
                if test_mode and itr < iters - 1:
                    continue
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)[:, :1]
                flow_predictions.append(flow_up)

            log_b_low = self.unc_head(net_list[0].float())
            log_b = F.interpolate(log_b_low, scale_factor=2 ** self.args.n_downsample,
                                  mode="bilinear", align_corners=False).clamp(-5.0, 5.0)
            if test_mode:
                return coords1 - coords0, flow_up, log_b
            return flow_predictions, log_b

    return RAFTStereoUncertainty(args)


# ───────────────────────────── metrics ──────────────────────────────────────
def d1_at_confidence(pred, gt, valid, b, keep_frac):
    """D1 over the most confident keep_frac of valid pixels (smallest b)."""
    mask = valid > 0.5
    if mask.sum() == 0:
        return 0.0
    err = (pred[mask] - gt[mask]).abs()
    gtv = gt[mask].clamp_min(1e-3)
    bv  = b[mask]
    if keep_frac < 1.0:
        k = max(1, int(len(bv) * keep_frac))
        idx = bv.argsort()[:k]
        err, gtv = err[idx], gtv[idx]
    return ((err > 3.0) & (err / gtv > 0.05)).float().mean().item() * 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--raft-dir",  required=True)
    p.add_argument("--ckpt",      required=True,
                   help="Fine-tuned disparity checkpoint (base is frozen).")
    p.add_argument("--out",       default="./ckpt_unc")
    p.add_argument("--steps",       type=int, default=2000)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--crop", nargs=2, type=int, default=[320, 640])
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--train-iters", type=int, default=16)
    p.add_argument("--valid-iters", type=int, default=16)
    p.add_argument("--val-split",   type=float, default=0.15)
    p.add_argument("--val-every",   type=int, default=250)
    p.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    device = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    full_train = Middlebury2014(args.data_root, crop=tuple(args.crop), training=True)
    full_val   = Middlebury2014(args.data_root, crop=None, training=False)
    n = len(full_train)
    n_val = max(1, int(n * args.val_split))
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[n - n_val:], perm[:n - n_val]
    train_set = torch.utils.data.Subset(full_train, train_idx)
    val_set   = torch.utils.data.Subset(full_val, val_idx)
    print(f"train {len(train_set)}  val {len(val_set)}  scenes")

    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    model = build_model(args.raft_dir, args.mixed_precision)
    from utils.utils import InputPadder   # raft-dir put on path by build_model
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing (should be unc_head only):", missing)
    model = model.to(device)
    model.train()
    model.freeze_bn()
    model.freeze_base()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_tr:,}")

    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=args.steps + 50, pct_start=0.05,
        cycle_momentum=False, anneal_strategy="linear")

    def validate():
        model.eval()
        nlls, d100, d50, d20 = [], [], [], []
        for i in range(len(val_set)):
            s = val_set[i]
            L = s["left"][None].to(device); R = s["right"][None].to(device)
            padder = InputPadder(L.shape, divis_by=32)
            Lp, Rp = padder.pad(L, R)
            with torch.no_grad():
                _, flow, log_b = model(Lp, Rp, iters=args.valid_iters, test_mode=True)
            pred  = -padder.unpad(flow).squeeze().cpu()
            log_b = padder.unpad(log_b).squeeze().cpu()
            gt, valid = s["disp"], s["valid"]
            b = log_b.exp()
            m = valid > 0.5
            nll = ((pred[m] - gt[m]).abs() / b[m] + (2 * b[m]).log()).mean().item()
            nlls.append(nll)
            d100.append(d1_at_confidence(pred, gt, valid, b, 1.0))
            d50.append(d1_at_confidence(pred, gt, valid, b, 0.5))
            d20.append(d1_at_confidence(pred, gt, valid, b, 0.2))
        model.train()
        model.freeze_bn()
        return (float(np.mean(nlls)), float(np.mean(d100)),
                float(np.mean(d50)), float(np.mean(d20)))

    nll0, a, b_, c = validate()
    print(f"[step 0]  NLL {nll0:.3f}  D1@100% {a:.2f}  D1@50% {b_:.2f}  D1@20% {c:.2f}")
    best_nll = nll0
    torch.save(model.state_dict(), out / "best.pth")

    step, t0 = 0, time.time()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            L = batch["left"].to(device); R = batch["right"].to(device)
            gt = batch["disp"].to(device); valid = batch["valid"].to(device)

            opt.zero_grad()
            preds, log_b = model(L, R, iters=args.train_iters)
            pred = -preds[-1].squeeze(1)          # final disparity (frozen)
            b = log_b.squeeze(1).exp()
            m = (valid > 0.5) & (gt < 700)
            loss = ((pred.detach()[m] - gt[m]).abs() / b[m] + (2 * b[m]).log()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % 50 == 0:
                print(f"step {step:5d}  NLL {loss.item():.3f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"{(time.time()-t0)/50:.2f}s/step")
                t0 = time.time()

            if step % args.val_every == 0:
                nll, a, b_, c = validate()
                mark = ""
                if nll < best_nll:
                    best_nll = nll
                    torch.save(model.state_dict(), out / "best.pth")
                    mark = "  ← saved best"
                print(f"[val @ {step}]  NLL {nll:.3f}  "
                      f"D1@100% {a:.2f}  D1@50% {b_:.2f}  D1@20% {c:.2f}{mark}")

    nll, a, b_, c = validate()
    if nll < best_nll:
        best_nll = nll
        torch.save(model.state_dict(), out / "best.pth")
    torch.save(model.state_dict(), out / "last.pth")
    print(f"\nDone. best NLL = {best_nll:.3f}")
    print(f"Sparsification (final): D1@100% {a:.2f} → D1@50% {b_:.2f} → D1@20% {c:.2f}")
    print("(monotonic decrease = uncertainty is informative)")
    print(f"Checkpoints: {out}/best.pth  {out}/last.pth")


if __name__ == "__main__":
    main()

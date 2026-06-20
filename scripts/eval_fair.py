"""Fair Middlebury eval in a COMMON metric space (downsample=2), so models
trained at different resolutions are directly comparable.

A model trained at downsample=D outputs disparity in /D units. We upsample its
prediction to the /2 GT resolution and scale the values by (D/2), then compute
masked KITTI metrics (valid AND gt < eval_max_disp) against the /2 ground truth.

Evaluated on the same seed-42 val split train.py uses, so numbers line up with
the values printed during training.

Usage:
  python scripts/eval_fair.py --ckpt checkpoints/middlebury_aanet_ft/best.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets import Middlebury2014Stereo
from src.models.aanet import AANetWrapper
from src.utils.metrics import compute_kitti_metrics

VAL_SPLIT = 0.15
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="datasets/middlebury2014")
    p.add_argument("--variant", default="both")
    p.add_argument("--eval-max-disp", type=int, default=192,
                   help="Mask GT pixels >= this (in /2 units), matching deployment.")
    p.add_argument("--eval-downsample", type=int, default=None,
                   help="Override model train downsample (else read from ckpt).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def val_indices(n):
    n_val = max(1, int(n * VAL_SPLIT))
    n_train = n - n_val
    _, val = random_split(range(n), [n_train, n_val],
                          generator=torch.Generator().manual_seed(SEED))
    return list(val)


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location=device)
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    max_disp = cfg.get("max_disp", 192)
    down = cfg.get("downsample", 2)
    model = AANetWrapper(max_disp=max_disp, predict_uncertainty=False).to(device).eval()
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    if not any(k.startswith("backbone.") for k in sd):
        sd = {"backbone." + k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model, max_disp, down


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model, max_disp, down = load_model(args.ckpt, device)
    if args.eval_downsample is not None:
        down = args.eval_downsample
    print(f"ckpt={args.ckpt}  train_downsample={down}  max_disp={max_disp}")

    # Two raw views of the same scenes: /2 for GT, /down for model input.
    ds_gt = Middlebury2014Stereo(args.data_root, transform=None, downsample=2, variant=args.variant)
    ds_in = Middlebury2014Stereo(args.data_root, transform=None, downsample=down, variant=args.variant)
    idx = val_indices(len(ds_gt))
    scale = down / 2.0  # /down disparity -> /2 disparity

    agg = {k: 0.0 for k in ["EPE", "D1-all", "bad-1", "bad-2", "bad-3"]}
    n = 0
    for i in idx:
        gt = ds_gt[i]["disparity"].to(device)              # (H2, W2) in /2 px
        H2, W2 = gt.shape
        si = ds_in[i]  # transform=None -> left/right already CHW float tensors
        L = si["left"].unsqueeze(0).to(device)
        R = si["right"].unsqueeze(0).to(device)
        pred = model(L, R)["disparity"]                    # (1, Hd, Wd) in /down px
        pred2 = F.interpolate(pred.unsqueeze(1), size=(H2, W2),
                              mode="bilinear", align_corners=False).squeeze(1) * scale
        V = ((gt > 0) & (gt < float(args.eval_max_disp))).float().unsqueeze(0)
        m = compute_kitti_metrics(pred2, gt.unsqueeze(0), V)
        for k in agg:
            agg[k] += m[k]
        n += 1

    print(f"\nval scenes = {n}  (common /2 metric space, masked gt<{args.eval_max_disp})")
    for k in agg:
        print(f"  {k:8s} {agg[k]/max(n,1):.3f}")


if __name__ == "__main__":
    main()

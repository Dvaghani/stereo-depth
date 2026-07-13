"""
Evaluate RAFT-Stereo on the SAME validation splits used for AANet training,
so the numbers are directly comparable to the thesis ablation table.

Reproduces train.py's split exactly: same dataset construction, same
random_split with seed 42, same metrics (compute_kitti_metrics).

Usage:
    # Middlebury val (46 pairs * 0.15 ≈ 6 samples), realtime checkpoint
    python scripts/eval_raft.py --dataset middlebury \
        --data-root /run/media/dvaghani/Expansion/Dataset/middlebury2014 \
        --model realtime

    # KITTI val (200 * 0.10 = 20 samples)
    python scripts/eval_raft.py --dataset kitti \
        --data-root /run/media/dvaghani/Expansion/Dataset/data_scene_flow \
        --model realtime
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
_RAFT = _HERE.parent / "third_party" / "RAFT-Stereo"
sys.path.insert(0, str(_RAFT))
sys.path.insert(0, str(_RAFT / "core"))

from raft_stereo import RAFTStereo            # noqa: E402
from utils.utils import InputPadder           # noqa: E402

from src.datasets import (KITTI2015Stereo, Middlebury2014Stereo,   # noqa: E402
                          StereoTransform)
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD    # noqa: E402
from src.utils.metrics import compute_kitti_metrics                # noqa: E402

PRESETS = {
    "realtime": dict(
        ckpt="raftstereo-realtime.pth",
        shared_backbone=True, n_downsample=3, n_gru_layers=2,
        slow_fast_gru=True, valid_iters=7,
    ),
    "middlebury": dict(
        ckpt="raftstereo-middlebury.pth",
        shared_backbone=False, n_downsample=2, n_gru_layers=3,
        slow_fast_gru=False, valid_iters=32,
    ),
    "sceneflow": dict(
        ckpt="raftstereo-sceneflow.pth",
        shared_backbone=False, n_downsample=2, n_gru_layers=3,
        slow_fast_gru=False, valid_iters=32,
    ),
}

# Match the training configs (middlebury_aanet_v2.yaml / kitti_aanet.yaml)
VAL_SPLIT = {"middlebury": 0.15, "kitti": 0.10}


def build_val_set(dataset: str, data_root: str):
    """Replicates train.py's dataset construction + seed-42 random split."""
    transform = StereoTransform(crop_size=None, color_jitter=0.0, training=False)
    if dataset == "kitti":
        full = KITTI2015Stereo(data_root, split="training", transform=transform)
    else:
        full = Middlebury2014Stereo(data_root, transform=transform,
                                    downsample=2, variant="both")
    n_val = max(1, int(len(full) * VAL_SPLIT[dataset]))
    n_train = len(full) - n_val
    _, val_set = torch.utils.data.random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    return val_set


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["middlebury", "kitti"], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", choices=list(PRESETS), default="realtime")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--ckpt", default=None,
                   help="Override checkpoint path (e.g. a fine-tuned best.pth). "
                        "Architecture still follows --model preset.")
    args = p.parse_args()

    preset = PRESETS[args.model]
    iters  = args.iters if args.iters is not None else preset["valid_iters"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading RAFT-Stereo ({args.model}, {iters} iters)...")
    raft_args = argparse.Namespace(
        hidden_dims=[128, 128, 128], corr_implementation="reg",
        corr_levels=4, corr_radius=4, context_norm="batch",
        mixed_precision=False,
        shared_backbone=preset["shared_backbone"],
        n_downsample=preset["n_downsample"],
        n_gru_layers=preset["n_gru_layers"],
        slow_fast_gru=preset["slow_fast_gru"],
    )
    model = RAFTStereo(raft_args)
    ckpt_path = Path(args.ckpt) if args.ckpt else _RAFT / "models" / preset["ckpt"]
    print(f"Checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(device).eval()

    val_set = build_val_set(args.dataset, args.data_root)
    print(f"{args.dataset} val set: {len(val_set)} samples (seed-42 split, "
          f"val_split={VAL_SPLIT[args.dataset]})")

    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    agg = {}
    t_total = 0.0
    for i in range(len(val_set)):
        sample = val_set[i]
        # dataset outputs ImageNet-normalized tensors; RAFT wants raw 0-255
        L = ((sample["left"]  * std_t + mean_t) * 255.0).clamp(0, 255)[None].to(device)
        R = ((sample["right"] * std_t + mean_t) * 255.0).clamp(0, 255)[None].to(device)
        gt    = sample["disparity"][None].to(device)
        valid = sample["valid"][None].to(device)

        padder = InputPadder(L.shape, divis_by=32)
        Lp, Rp = padder.pad(L, R)
        t0 = time.time()
        with torch.no_grad():
            _, flow = model(Lp, Rp, iters=iters, test_mode=True)
        t_total += time.time() - t0
        pred = -padder.unpad(flow).squeeze(1)   # (1, H, W)

        m = compute_kitti_metrics(pred, gt, valid)
        for k, v in m.items():
            agg.setdefault(k, []).append(v)
        print(f"  [{i+1}/{len(val_set)}] EPE={m['EPE']:.3f}  D1={m['D1-all']:.2f}%")

    print(f"\n=== RAFT-Stereo ({args.model}, {iters} iters) on {args.dataset} val ===")
    for k, vals in agg.items():
        print(f"  {k:8s}: {np.mean(vals):.3f}")
    print(f"  avg inference: {t_total/len(val_set)*1000:.0f} ms/frame")


if __name__ == "__main__":
    main()

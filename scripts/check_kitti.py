"""
Run from the project root:
    python scripts/check_kitti.py --data-root datasets/data_scene_flow
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.datasets import KITTI2015Stereo, StereoTransform
from src.models import StereoUNet
from src.models.stereo_unet import StereoUNetConfig
from src.utils.metrics import compute_kitti_metrics
from src.utils.losses import multi_scale_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--max-disp", type=int, default=192)
    p.add_argument("--crop", type=int, nargs=2, default=[256, 512])
    args = p.parse_args()

    # 1. Dataset
    tf = StereoTransform(crop_size=tuple(args.crop), training=False)
    ds = KITTI2015Stereo(args.data_root, split="training", transform=tf, return_path=True)
    print(f"[check] dataset size = {len(ds)}")

    sample = ds[0]
    L, R, D, V = sample["left"], sample["right"], sample["disparity"], sample["valid"]
    print(f"[check] sample: L={tuple(L.shape)} R={tuple(R.shape)} "
          f"D={tuple(D.shape)} valid_frac={V.mean().item():.3f}")
    print(f"[check] disparity range (valid only): "
          f"{D[V > 0].min().item():.2f} - {D[V > 0].max().item():.2f} px")
    print(f"[check] first path: {sample['path']}")

    # 2. Model (small config so this runs fast on CPU)
    cfg = StereoUNetConfig(
        max_disp=args.max_disp, feat_channels=32, unet_base_channels=16,
        use_cuda_extension=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StereoUNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[check] model params = {n_params:,}, device = {device}")

    # 3. Forward + loss + metrics on one real batch
    L = L.unsqueeze(0).to(device)
    R = R.unsqueeze(0).to(device)
    D = D.unsqueeze(0).to(device)
    V = V.unsqueeze(0).to(device)
    out = model(L, R)
    loss = multi_scale_loss(out["disparity"], out["disparity_low"], D, V)
    metrics = compute_kitti_metrics(out["disparity"], D, V)
    print(f"[check] forward OK   loss = {loss.item():.4f} "
          f"(random init, so this should be large)")
    print(f"[check] init metrics (random init, expected to be bad): {metrics}")
    print("[check] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

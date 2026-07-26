"""Compare a Jetson TensorRT disparity map against the desktop PyTorch reference.

Generates the FP32 PyTorch disparity for the same stereo pair and resolution,
then reports the error introduced by ONNX export + FP16 quantisation. Run on
the DESKTOP after copying the .npy back from the Jetson.

Usage:
    python scripts/compare_disparity.py \
        --ckpt checkpoints/raft_middlebury_ft/best.pth \
        --left  outputs/capture_160mm_20260608_131911/left.png \
        --right outputs/capture_160mm_20260608_131911/right.png \
        --trt   disp_trt_fp16.npy \
        --iters 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from export_raft_onnx import RAFTStereo, RaftDisparity, realtime_args  # noqa: E402


def preprocess(path: str, width: int, height: int) -> torch.Tensor:
    """Must match scripts/jetson_trt_infer.py exactly: raw 0-255 float RGB, CHW."""
    img = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr[None])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--trt", required=True, help=".npy produced on the Jetson")
    p.add_argument("--iters", type=int, default=7)
    p.add_argument("--save-vis", default=None, help="optional side-by-side PNG")
    args = p.parse_args()

    trt_disp = np.load(args.trt)
    height, width = trt_disp.shape
    print(f"TRT disparity: {trt_disp.shape}, "
          f"range [{trt_disp.min():.3f}, {trt_disp.max():.3f}]")

    model = RAFTStereo(realtime_args())
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    wrapper = RaftDisparity(model, args.iters).eval()

    left = preprocess(args.left, width, height)
    right = preprocess(args.right, width, height)
    with torch.no_grad():
        ref = wrapper(left, right)[0, 0].numpy()
    print(f"PyTorch FP32:  {ref.shape}, range [{ref.min():.3f}, {ref.max():.3f}]")

    diff = np.abs(ref - trt_disp)
    denom = np.maximum(np.abs(ref), 1e-6)
    rel = diff / denom

    print("\n=== FP16 TensorRT vs FP32 PyTorch ===")
    print(f"  mean abs error   {diff.mean():.4f} px")
    print(f"  median abs error {np.median(diff):.4f} px")
    print(f"  p95 abs error    {np.percentile(diff, 95):.4f} px")
    print(f"  max abs error    {diff.max():.4f} px")
    print(f"  mean rel error   {100 * rel.mean():.3f} %")
    for t in (0.5, 1.0, 3.0):
        print(f"  pixels off by >{t} px: {100 * (diff > t).mean():.2f} %")

    # >3px is the standard D1 outlier threshold in stereo benchmarks
    d1 = 100 * (diff > 3.0).mean()
    verdict = ("FP16 is faithful" if d1 < 1.0 else
               "FP16 degrades disparity — consider an FP32 engine")
    print(f"\n  D1-style outlier rate (>3px): {d1:.2f} %  ->  {verdict}")

    if args.save_vis:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        vmin, vmax = ref.min(), ref.max()
        ax[0].imshow(ref, cmap="magma", vmin=vmin, vmax=vmax); ax[0].set_title("PyTorch FP32")
        ax[1].imshow(trt_disp, cmap="magma", vmin=vmin, vmax=vmax); ax[1].set_title("TensorRT FP16")
        im = ax[2].imshow(diff, cmap="inferno"); ax[2].set_title("abs difference")
        fig.colorbar(im, ax=ax[2])
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(args.save_vis, dpi=110)
        print(f"  visualisation -> {args.save_vis}")


if __name__ == "__main__":
    main()

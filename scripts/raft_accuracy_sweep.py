"""Accuracy cost of reducing RAFT-Stereo iterations / input resolution.

Runs entirely on the DESKTOP: how much disparity you lose by shrinking the
model is a property of the network, not the device, so only the latency half of
the speed-vs-accuracy table needs the Jetson.

Each configuration is compared against a high-iteration, full-resolution FP32
reference treated as pseudo-ground-truth.

Resolution needs care: disparity is measured in pixels, so halving the input
width halves the disparity values too. Lower-resolution outputs are therefore
upsampled AND rescaled by (ref_width / cfg_width) before comparison —
without that the errors would be dominated by a trivial scale factor.

Usage:
    python scripts/raft_accuracy_sweep.py \
        --ckpt checkpoints/raft_middlebury_ft/best.pth \
        --left  outputs/capture_160mm_20260608_131911/left.png \
        --right outputs/capture_160mm_20260608_131911/right.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from export_raft_onnx import RAFTStereo, RaftDisparity, realtime_args  # noqa: E402

# (iters, height, width) — the configurations exported for the Jetson sweep
CONFIGS = [
    (7, 480, 640),
    (4, 480, 640),
    (2, 480, 640),
    (7, 320, 480),
    (4, 320, 480),
]


def preprocess(path: str, width: int, height: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    return torch.from_numpy(np.transpose(arr, (2, 0, 1))[None])


def run(model_wrapper, left_path, right_path, height, width) -> np.ndarray:
    left = preprocess(left_path, width, height)
    right = preprocess(right_path, width, height)
    with torch.no_grad():
        return model_wrapper(left, right)[0, 0].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--ref-iters", type=int, default=32,
                   help="iterations for the pseudo-ground-truth reference")
    p.add_argument("--ref-height", type=int, default=480)
    p.add_argument("--ref-width", type=int, default=640)
    args = p.parse_args()

    model = RAFTStereo(realtime_args())
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    print("computing reference: iters=%d at %dx%d ..."
          % (args.ref_iters, args.ref_height, args.ref_width))
    ref = run(RaftDisparity(model, args.ref_iters).eval(),
              args.left, args.right, args.ref_height, args.ref_width)
    print("reference range [%.2f, %.2f] px\n" % (ref.min(), ref.max()))

    rows = []
    for iters, height, width in CONFIGS:
        disp = run(RaftDisparity(model, iters).eval(),
                   args.left, args.right, height, width)

        if (height, width) != (args.ref_height, args.ref_width):
            # upsample to reference grid, then rescale: disparity is in pixels
            # and scales with image width
            t = torch.from_numpy(disp)[None, None]
            t = F.interpolate(t, size=(args.ref_height, args.ref_width),
                              mode="bilinear", align_corners=True)
            disp = t[0, 0].numpy() * (float(args.ref_width) / width)

        err = np.abs(ref - disp)
        rows.append({
            "cfg": "i%-2d %dx%d" % (iters, height, width),
            "mean": err.mean(),
            "p95": np.percentile(err, 95),
            "max": err.max(),
            "d1": 100.0 * (err > 3.0).mean(),
            "rel": 100.0 * (err / np.maximum(np.abs(ref), 1e-6)).mean(),
        })
        print("  %s   mean %6.3f px   p95 %6.3f   >3px %5.2f %%"
              % (rows[-1]["cfg"], rows[-1]["mean"], rows[-1]["p95"], rows[-1]["d1"]))

    print("\n=== accuracy vs iters=%d @ %dx%d reference ==="
          % (args.ref_iters, args.ref_height, args.ref_width))
    print("%-14s %10s %10s %10s %9s %9s"
          % ("config", "mean px", "p95 px", "max px", "rel %", ">3px %"))
    for r in rows:
        print("%-14s %10.3f %10.3f %10.3f %9.2f %9.2f"
              % (r["cfg"], r["mean"], r["p95"], r["max"], r["rel"], r["d1"]))
    print("\nPair latency on the Jetson comes from jetson_sustained_bench.py;"
          "\njoin on the config label to complete the speed-vs-accuracy table.")


if __name__ == "__main__":
    main()

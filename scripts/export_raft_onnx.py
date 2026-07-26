"""Export RAFT-Stereo (realtime config) to ONNX for Jetson / TensorRT.

Two things make this work where a naive export fails:

1. `RAFTStereo.forward(..., iters=N)` takes a plain Python int, so
   `for itr in range(iters)` is unrolled by the legacy TorchScript tracer.
   Pass `dynamo=False`; the graph grows linearly with --iters.

2. The correlation lookup calls `F.grid_sample`, which ONNX only supports from
   opset 16 while TensorRT only parses GridSample from 8.5 (JetPack 4.6.1 ships
   8.2.1). In RAFT-*Stereo* the correlation volume is (N, C, 1, W) and the
   sampled y is always zero, so that reduces to 1D interpolation — see
   onnx_sampler1d.py, monkeypatched in below.

Usage:
    python scripts/export_raft_onnx.py \
        --ckpt checkpoints/raft_middlebury_ft/best.pth \
        --out  raft_realtime_i7_480x640.onnx \
        --iters 7 --height 480 --width 640
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_RAFT = _REPO / "third_party" / "RAFT-Stereo"
sys.path.insert(0, str(_RAFT))
sys.path.insert(0, str(_RAFT / "core"))
sys.path.insert(0, str(_HERE))

from raft_stereo import RAFTStereo                    # noqa: E402
import core.corr as _corr                             # noqa: E402
from onnx_sampler1d import bilinear_sampler_1d        # noqa: E402

# corr.py does `from core.utils.utils import bilinear_sampler`, so the name has
# to be replaced in core.corr — patching core.utils.utils would be too late.
_corr.bilinear_sampler = bilinear_sampler_1d


def realtime_args():
    """Matches the config in scripts/live_detect.py. corr_implementation must
    stay "reg" — the *_cuda variants are custom extensions and cannot export."""
    return argparse.Namespace(
        hidden_dims=[128, 128, 128],
        corr_implementation="reg",
        corr_levels=4, corr_radius=4,
        context_norm="batch",
        mixed_precision=False,
        shared_backbone=True, n_downsample=3,
        n_gru_layers=2, slow_fast_gru=True,
    )


class RaftDisparity(nn.Module):
    """Pins iters/test_mode so the exported graph has a single input pair and a
    single output, and flips sign — RAFT emits negative x-flow, disparity is
    positive."""

    def __init__(self, model: nn.Module, iters: int):
        super().__init__()
        self.model = model
        self.iters = iters

    def forward(self, left, right):
        _, flow_up = self.model(left, right, iters=self.iters, test_mode=True)
        return -flow_up


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=7,
                   help="unrolled refinement steps; fewer = smaller graph, less accurate")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()

    model = RAFTStereo(realtime_args())
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    # strict=False: the uncertainty checkpoint carries head weights this model lacks
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:4]}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:4]}")

    wrapper = RaftDisparity(model, args.iters).eval()

    left = torch.randn(1, 3, args.height, args.width)
    right = torch.randn(1, 3, args.height, args.width)
    with torch.no_grad():
        out = wrapper(left, right)
    print(f"torch forward OK -> disparity {tuple(out.shape)}, "
          f"range [{out.min():.2f}, {out.max():.2f}]")

    torch.onnx.export(
        wrapper, (left, right), args.out,
        input_names=["left", "right"], output_names=["disparity"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"ONNX export OK -> {args.out} "
          f"({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

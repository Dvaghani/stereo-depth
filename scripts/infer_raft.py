"""
RAFT-Stereo inference on a saved stereo pair — with optional uncertainty.

Auto-detects the checkpoint type: a plain RAFT checkpoint produces disparity
and depth; a checkpoint containing the trained uncertainty head additionally
produces the per-pixel confidence map and a confidence-masked disparity.

Usage:
    python scripts/infer_raft.py \
        --ckpt  checkpoints/raft_middleburry_uncertainty/best.pth \
        --left  outputs/capture_160mm_20260608_131946/left.png \
        --right outputs/capture_160mm_20260608_131946/right.png \
        --baseline 0.1606 --focal 1247.2 --scale 0.5 \
        --out   outputs/raft_infer/test1

Outputs (in --out):
    disparity.png       turbo-colorized disparity (contrast stretched)
    disparity.npy       raw disparity in px (at inference resolution)
    depth_mm.png        16-bit depth in millimetres
    confidence.png      turbo confidence map (red = confident)   [uncertainty ckpt]
    disparity_conf.png  disparity with b >= threshold masked out [uncertainty ckpt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
_RAFT = _HERE.parent / "third_party" / "RAFT-Stereo"
sys.path.insert(0, str(_RAFT))
sys.path.insert(0, str(_RAFT / "core"))


def turbo(d, valid=None):
    v = d[valid] if valid is not None else d[d > 0]
    if v.size < 100:
        v = d.flatten()
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    n = np.clip((d - lo) / max(hi - lo, 1e-3), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  required=True)
    p.add_argument("--left",  required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--out",   default="outputs/raft_infer")
    p.add_argument("--baseline", type=float, default=0.1606, help="metres")
    p.add_argument("--focal",    type=float, default=1247.2, help="px, full resolution")
    p.add_argument("--scale",    type=float, default=0.5)
    p.add_argument("--iters",    type=int, default=16)
    p.add_argument("--b-thresh", type=float, default=2.0,
                   help="confidence threshold in px for the masked output")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ── load checkpoint, auto-detect uncertainty head ─────────────────────────
    sd = torch.load(args.ckpt, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    has_unc = any(k.startswith("unc_head") for k in sd)

    if has_unc:
        from src.models.raft_uncertainty import build_raft_uncertainty, realtime_args
        print("Uncertainty checkpoint detected.")
        model = build_raft_uncertainty(realtime_args(mixed_precision=False))
    else:
        from raft_stereo import RAFTStereo
        model = RAFTStereo(argparse.Namespace(
            hidden_dims=[128, 128, 128], corr_implementation="reg",
            corr_levels=4, corr_radius=4, context_norm="batch",
            mixed_precision=False, shared_backbone=True, n_downsample=3,
            n_gru_layers=2, slow_fast_gru=True))
    model.load_state_dict(sd)
    model = model.to(device).eval()
    from utils.utils import InputPadder

    # ── load + downsample images ──────────────────────────────────────────────
    L = cv2.imread(args.left); R = cv2.imread(args.right)
    if L is None or R is None:
        raise SystemExit("Could not read input images.")
    if args.scale != 1.0:
        L = cv2.resize(L, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        R = cv2.resize(R, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    focal_eff = args.focal * args.scale

    def to_t(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(device)

    t1, t2 = to_t(L), to_t(R)
    padder = InputPadder(t1.shape, divis_by=32)
    t1, t2 = padder.pad(t1, t2)

    # ── inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        if has_unc:
            _, flow, log_b = model(t1, t2, iters=args.iters, test_mode=True)
            b = padder.unpad(log_b).squeeze().exp().cpu().numpy()
        else:
            _, flow = model(t1, t2, iters=args.iters, test_mode=True)
            b = None
    disp = -padder.unpad(flow).squeeze().cpu().numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disp > 0, args.baseline * focal_eff / disp, 0.0)

    # ── save outputs ──────────────────────────────────────────────────────────
    np.save(out / "disparity.npy", disp)
    cv2.imwrite(str(out / "disparity.png"), turbo(disp))
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(out / "depth_mm.png"), depth_mm)
    saved = ["disparity.png", "disparity.npy", "depth_mm.png"]

    if b is not None:
        bn = np.clip(b / 8.0, 0, 1)
        conf_vis = cv2.applyColorMap(((1 - bn) * 255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO)
        cv2.imwrite(str(out / "confidence.png"), conf_vis)

        conf_mask = b < args.b_thresh
        disp_masked = turbo(disp, valid=conf_mask)
        disp_masked[~conf_mask] = (40, 40, 40)
        cv2.imwrite(str(out / "disparity_conf.png"), disp_masked)
        saved += ["confidence.png", "disparity_conf.png"]

    print(f"\ndisparity: {disp.min():.1f} .. {disp.max():.1f} px")
    v = depth[depth > 0]
    print(f"depth:     {v.min():.2f} .. {v.max():.2f} m")
    if b is not None:
        print(f"b:         {b.min():.2f} .. {b.max():.2f} px   "
              f"confident (b<{args.b_thresh:g}px): {100 * (b < args.b_thresh).mean():.0f}%")
    print(f"\nSaved {', '.join(saved)} → {out}/")


if __name__ == "__main__":
    main()

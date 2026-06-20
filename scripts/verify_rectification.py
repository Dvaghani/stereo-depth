"""
Verify a stereo calibration by measuring the residual VERTICAL disparity on a
real (raw, unrectified) stereo pair.

This is the number that actually predicts stereo-matching quality. A horizontal
stereo matcher (AANet, SGBM, ...) searches only along scanlines, so any vertical
offset between true correspondences is unrecoverable error. Targets:

    < 0.5 px  : excellent — calibration is not your bottleneck
    0.5-1 px  : acceptable
    > 1 px    : rectification is hurting you; recalibrate
    > 2 px    : catastrophic; matcher output will look like noise

It rectifies the raw pair with the calibration's stored maps, finds SIFT
correspondences, keeps the ones with a plausible positive horizontal disparity,
and reports the distribution of their vertical offset.

Usage:
    python scripts/verify_rectification.py \\
        --calib outputs/calibration_160mm/stereo_calib.npz \\
        --left  outputs/capture_159mm_20260528_144805/left_raw.png \\
        --right outputs/capture_159mm_20260528_144805/right_raw.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", type=Path, required=True,
                   help="stereo_calib.npz from calibrate_stereo / recalibrate_filtered.")
    p.add_argument("--left", type=Path, required=True,
                   help="RAW (unrectified) left image.")
    p.add_argument("--right", type=Path, required=True,
                   help="RAW (unrectified) right image.")
    p.add_argument("--max-disp", type=float, default=300.0,
                   help="Upper bound on plausible horizontal disparity (px).")
    p.add_argument("--save", type=Path, default=None,
                   help="Optional path to save a rectified side-by-side preview "
                        "with horizontal reference lines.")
    return p.parse_args()


def main():
    args = parse_args()
    cal = np.load(args.calib)
    for k in ("map1L", "map2L", "map1R", "map2R"):
        if k not in cal:
            raise SystemExit(f"{args.calib} is missing '{k}' — not a full calibration.")

    Lr = cv2.imread(str(args.left))
    Rr = cv2.imread(str(args.right))
    if Lr is None or Rr is None:
        raise SystemExit("Could not read input images.")

    L = cv2.remap(Lr, cal["map1L"], cal["map2L"], cv2.INTER_LINEAR)
    R = cv2.remap(Rr, cal["map1R"], cal["map2R"], cv2.INTER_LINEAR)
    Lg = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(Lg, None)
    k2, d2 = sift.detectAndCompute(Rg, None)
    if d1 is None or d2 is None:
        raise SystemExit("No SIFT features found — textureless scene?")
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    dy, dx = [], []
    for m in good:
        x1, y1 = k1[m.queryIdx].pt
        x2, y2 = k2[m.trainIdx].pt
        if 0 < (x1 - x2) < args.max_disp:
            dy.append(y1 - y2)
            dx.append(x1 - x2)
    dy = np.array(dy)
    dx = np.array(dx)
    if len(dy) < 30:
        raise SystemExit(f"Only {len(dy)} usable matches — inconclusive. "
                         f"Try a more textured scene.")

    med = np.median(dy)
    mad = np.median(np.abs(dy - med))
    keep = np.abs(dy - med) < 5 * mad + 1e-6
    dyk = dy[keep]

    mean, std, amed = dyk.mean(), dyk.std(), abs(med)
    worst = float(np.percentile(np.abs(dyk - 0.0), 95))
    if amed < 0.5:
        verdict = "EXCELLENT — calibration is not your bottleneck"
    elif amed < 1.0:
        verdict = "ACCEPTABLE"
    elif amed < 2.0:
        verdict = "POOR — recalibrate"
    else:
        verdict = "CATASTROPHIC — matcher output will be noise; recalibrate"

    print(f"Calibration:  {args.calib}")
    if "rms_stereo" in cal:
        print(f"Stored stereo RMS: {float(cal['rms_stereo'][0]):.3f} px")
    print(f"Matches used: {len(dyk)}   disparity range {dx.min():.0f}-{dx.max():.0f} px")
    print(f"Vertical residual:  mean={mean:+.2f}px  |median|={amed:.2f}px  "
          f"std={std:.2f}px  95th-pct|dy|={worst:.2f}px")
    print(f"VERDICT: {verdict}")

    if args.save:
        side = np.hstack([L, R])
        for y in range(0, side.shape[0], 50):
            cv2.line(side, (0, y), (side.shape[1], y), (0, 255, 0), 1)
        cv2.imwrite(str(args.save), side)
        print(f"Saved preview: {args.save}")


if __name__ == "__main__":
    main()

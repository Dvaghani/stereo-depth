"""
Measure — and optionally bake out — a residual CONSTANT vertical offset in a
stereo rectification.

Why this exists
---------------
A well-shaped calibration (full frame coverage, low RMS) can still leave a small
*uniform* vertical disparity between the rectified left/right images — typically
from a sub-degree residual pitch in the extrinsics, or the rig pitching slightly
between calibration and use. A horizontal stereo matcher (AANet, SGBM) assumes
correspondences lie on the same scanline, so even a constant ~2 px vertical
offset biases every match.

Unlike a *varying* residual (which means the geometry is wrong and you must
recalibrate), a constant offset is trivially correctable: shift one camera's
rectification map vertically so the median vertical residual goes to ~0.

This script:
  1. Rectifies each given raw pair with the calibration's stored maps.
  2. Finds SIFT correspondences and measures the per-pair vertical residual
     (robust median) AND whether it depends on x / y / disparity.
  3. Reports stability across pairs. If the offset is stable and position/depth
     independent, it is safe to bake.
  4. With --apply: bakes the median shift into map2L (vertical), backs up the
     original npz, rewrites stereo_calib.npz, and re-measures to confirm.

Usage:
    # Diagnose across several captures (no changes written):
    python scripts/refine_vertical_shift.py \\
        --calib outputs/calibration_160mm/stereo_calib.npz \\
        --pairs outputs/capture_160mm_*

    # Bake the correction once you've confirmed it's stable:
    python scripts/refine_vertical_shift.py \\
        --calib outputs/calibration_160mm/stereo_calib.npz \\
        --pairs outputs/capture_160mm_* --apply
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--pairs", nargs="+", required=True,
                   help="Capture dirs (each containing left_raw.png/right_raw.png) "
                        "or glob patterns thereof.")
    p.add_argument("--max-disp", type=float, default=400.0)
    p.add_argument("--apply", action="store_true",
                   help="Bake the measured median vertical shift into the maps.")
    p.add_argument("--max-spread", type=float, default=0.6,
                   help="Refuse to --apply if the per-pair offsets disagree by "
                        "more than this std (px) — that means a loose rig, not a "
                        "constant offset.")
    return p.parse_args()


def measure_pair(cal, left_png, right_png, max_disp):
    """Return (dy_array, x, y, dx) for robust-kept SIFT matches, or None."""
    Lr, Rr = cv2.imread(str(left_png)), cv2.imread(str(right_png))
    if Lr is None or Rr is None:
        return None
    L = cv2.remap(Lr, cal["map1L"], cal["map2L"], cv2.INTER_LINEAR)
    R = cv2.remap(Rr, cal["map1R"], cal["map2R"], cv2.INTER_LINEAR)
    Lg, Rg = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(8000)
    k1, d1 = sift.detectAndCompute(Lg, None)
    k2, d2 = sift.detectAndCompute(Rg, None)
    if d1 is None or d2 is None:
        return None
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < 0.8 * n.distance]
    X, Y, DX, DY = [], [], [], []
    for m in good:
        x1, y1 = k1[m.queryIdx].pt
        x2, y2 = k2[m.trainIdx].pt
        if 0 < (x1 - x2) < max_disp:
            X.append(x1); Y.append(y1); DX.append(x1 - x2); DY.append(y1 - y2)
    if len(DY) < 30:
        return None
    X, Y, DX, DY = map(np.array, (X, Y, DX, DY))
    med = np.median(DY); mad = np.median(np.abs(DY - med))
    keep = np.abs(DY - med) < 4 * mad + 1e-6
    return DY[keep], X[keep], Y[keep], DX[keep]


def expand_pairs(patterns):
    dirs = []
    for pat in patterns:
        hits = sorted(glob.glob(pat)) or [pat]
        for h in hits:
            d = Path(h)
            if (d / "left_raw.png").exists() and (d / "right_raw.png").exists():
                dirs.append(d)
    return dirs


def main():
    args = parse_args()
    cal = dict(np.load(args.calib))
    dirs = expand_pairs(args.pairs)
    if not dirs:
        raise SystemExit("No capture dirs with left_raw.png/right_raw.png found.")

    print(f"Calibration: {args.calib}")
    if "rms_stereo" in cal:
        print(f"Stored stereo RMS: {float(cal['rms_stereo'][0]):.3f} px\n")

    per_pair_med = []
    print(f"{'pair':<40} {'n':>4} {'med_dy':>7} {'std':>5} "
          f"{'c(x)':>5} {'c(y)':>5} {'c(disp)':>7}")
    for d in dirs:
        res = measure_pair(cal, d / "left_raw.png", d / "right_raw.png", args.max_disp)
        if res is None:
            print(f"{d.name:<40} {'--':>4}  (too few matches / unreadable)")
            continue
        DY, X, Y, DX = res
        cx = np.corrcoef(DY, X)[0, 1] if len(DY) > 2 else 0
        cy = np.corrcoef(DY, Y)[0, 1] if len(DY) > 2 else 0
        cd = np.corrcoef(DY, DX)[0, 1] if len(DY) > 2 else 0
        med = float(np.median(DY))
        per_pair_med.append(med)
        print(f"{d.name:<40} {len(DY):>4} {med:>+7.2f} {DY.std():>5.2f} "
              f"{cx:>+5.2f} {cy:>+5.2f} {cd:>+7.2f}")

    if not per_pair_med:
        raise SystemExit("No usable pairs.")

    arr = np.array(per_pair_med)
    overall = float(np.median(arr))
    spread = float(arr.std())
    print(f"\nPer-pair median dy: {np.round(arr, 2).tolist()}")
    print(f"Overall median offset: {overall:+.2f} px   spread(std): {spread:.2f} px")

    if spread > args.max_spread:
        print(f"\n  UNSTABLE: offsets disagree by {spread:.2f} px > {args.max_spread} px.")
        print("  This is NOT a constant offset — likely a loose/flexing rig or a")
        print("  geometry error. Baking a fixed shift will not help. Fix the mount")
        print("  (rigid bar, no play between cameras) and recapture before applying.")
        return
    print(f"  STABLE (spread {spread:.2f} <= {args.max_spread}): a constant offset, "
          f"safe to bake.")

    if not args.apply:
        print("\n--apply not set: no changes written. Re-run with --apply to bake "
              f"the {overall:+.2f}px shift into the rectification maps.")
        return

    # ── Bake the shift ───────────────────────────────────────────────────────
    # measured dy = yL - yR = overall. To zero it, move LEFT rectified content
    # down by `overall` px so its features land on the right's scanline. In a
    # remap, content shifts down by δ when we sample from input row (y - δ):
    #   map2L_new = map2L - overall
    backup = args.calib.with_suffix(".npz.prebake")
    if not backup.exists():
        shutil.copy(args.calib, backup)
        print(f"\nBacked up original to {backup}")
    cal["map2L"] = (cal["map2L"] - overall).astype(cal["map2L"].dtype)
    cal["vshift_baked"] = np.array([overall], dtype=np.float32)
    np.savez(args.calib, **cal)
    print(f"Baked {overall:+.2f}px vertical shift into map2L; rewrote {args.calib}")

    # Re-measure to confirm
    cal2 = dict(np.load(args.calib))
    after = []
    for d in dirs:
        res = measure_pair(cal2, d / "left_raw.png", d / "right_raw.png", args.max_disp)
        if res is not None:
            after.append(float(np.median(res[0])))
    if after:
        a = np.array(after)
        print(f"\nAfter baking — per-pair median dy: {np.round(a, 2).tolist()}")
        print(f"  overall |median| now: {abs(np.median(a)):.2f} px "
              f"(was {abs(overall):.2f})")
        if abs(np.median(a)) < 0.5:
            print("  VERDICT: EXCELLENT — vertical residual is now sub-0.5px.")
        else:
            print("  VERDICT: improved but still >0.5px; check for depth-dependence.")


if __name__ == "__main__":
    main()

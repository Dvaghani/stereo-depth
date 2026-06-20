"""
Compute stereo calibration from captured checkerboard pairs.

Reads pair_*.png files produced by scripts/capture_calibration.py and writes:
    <input>/stereo_calib.npz   — full calibration (loadable by capture_stereo.py)
    <input>/stereo_calib.txt   — human-readable summary

The .npz contains intrinsics (K1, D1, K2, D2), extrinsics (R, T, E, F),
rectification (R1, R2, P1, P2, Q), undistort/rectify maps for both cameras,
the image size, the estimated baseline (mm), and the rectified focal length (px).

Usage:
    python scripts/calibrate_stereo.py \\
        --input outputs/calibration_160mm \\
        --baseline-mm 160 \\
        --pattern 9 6 --square-mm 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_PATTERN = (9, 6)
DEFAULT_SQUARE_MM = 25.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Directory containing left/ and right/ pair_*.png subfolders.")
    p.add_argument("--baseline-mm", type=float, required=True,
                   help="Nominal baseline (mm) — used as a sanity check against "
                        "the value recovered from calibration.")
    p.add_argument("--pattern", type=int, nargs=2, default=list(DEFAULT_PATTERN),
                   metavar=("COLS", "ROWS"))
    p.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM)
    p.add_argument("--alpha", type=float, default=0.0,
                   help="Rectification cropping. 0 = tight crop to common valid "
                        "area (no black borders), 1 = keep all pixels (black "
                        "borders around). Default 0.")
    return p.parse_args()


def main():
    args = parse_args()
    pattern = tuple(args.pattern)
    in_dir = args.input
    left_dir = in_dir / "left"
    right_dir = in_dir / "right"

    left_files = sorted(left_dir.glob("pair_*.png"))
    right_files = sorted(right_dir.glob("pair_*.png"))

    if not left_files:
        raise SystemExit(f"No pair_*.png files in {left_dir}")
    if len(left_files) != len(right_files):
        raise SystemExit(f"Pair count mismatch: {len(left_files)} left "
                         f"vs {len(right_files)} right")
    print(f"Found {len(left_files)} stereo pair(s) in {in_dir}")

    # Object points: 3D coords of checkerboard corners in board coords (z=0).
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:pattern[0], 0:pattern[1]]
                   .T.reshape(-1, 2)
                   .astype(np.float32) * args.square_mm)

    refine_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    chess_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    objpoints = []
    imgpointsL = []
    imgpointsR = []
    accepted = []
    rejected = []
    img_size = None

    for lf, rf in zip(left_files, right_files):
        imgL = cv2.imread(str(lf))
        imgR = cv2.imread(str(rf))
        if img_size is None:
            img_size = (imgL.shape[1], imgL.shape[0])  # (W, H)

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = cv2.findChessboardCorners(grayL, pattern, flags=chess_flags)
        foundR, cornersR = cv2.findChessboardCorners(grayR, pattern, flags=chess_flags)
        if not (foundL and foundR):
            rejected.append(lf.name)
            continue

        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), refine_crit)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), refine_crit)

        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
        accepted.append(lf.name)

    print(f"Accepted: {len(accepted)} pair(s); rejected: {len(rejected)}")
    if len(accepted) < 10:
        print(f"WARNING: only {len(accepted)} pairs accepted — calibration may "
              f"be unreliable. Aim for >=15.")
    if rejected:
        print(f"  Rejected (checkerboard not found in BOTH): "
              f"{rejected[:5]}{' ...' if len(rejected) > 5 else ''}")
    if not objpoints:
        raise SystemExit("No usable pairs. Recapture with a clearer checkerboard.")

    # Stage 1: monocular calibration for each camera
    print("\nCalibrating left camera...")
    rms_L, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, img_size, None, None)
    print(f"  RMS reprojection error: {rms_L:.3f} px")
    print("Calibrating right camera...")
    rms_R, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, img_size, None, None)
    print(f"  RMS reprojection error: {rms_R:.3f} px")

    # Stage 2: stereo calibration (fix intrinsics, solve for R, T)
    print("Stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC
    rms_S, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        K1, D1, K2, D2, img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags,
    )
    print(f"  RMS stereo reprojection error: {rms_S:.3f} px")
    if rms_S > 1.5:
        print("  WARNING: RMS > 1.5 px is high — check for motion blur or "
              "insufficient pose variety in your captures.")

    baseline_estimated = float(np.linalg.norm(T))
    print(f"  Estimated baseline: {baseline_estimated:.1f} mm "
          f"(nominal: {args.baseline_mm} mm)")
    if abs(baseline_estimated - args.baseline_mm) > 20:
        print(f"  WARNING: estimated baseline differs from nominal by "
              f">20 mm — verify rig measurement and square size.")

    # Stage 3: rectification
    print("Computing rectification maps...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=args.alpha,
    )
    map1L, map2L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

    focal_px = float(P1[0, 0])  # rectified left camera focal length
    print(f"  Rectified focal length: {focal_px:.1f} px")

    # Save .npz for programmatic loading
    out_npz = in_dir / "stereo_calib.npz"
    np.savez(
        out_npz,
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1L=map1L, map2L=map2L, map1R=map1R, map2R=map2R,
        image_size=np.array(img_size, dtype=np.int32),
        baseline_mm=np.array([baseline_estimated]),
        focal_px=np.array([focal_px]),
        rms_stereo=np.array([rms_S]),
    )
    print(f"\nSaved calibration to: {out_npz}")

    # Save .txt for human inspection
    out_txt = in_dir / "stereo_calib.txt"
    with open(out_txt, "w") as f:
        f.write(f"# Stereo calibration\n")
        f.write(f"# Source pairs: {len(accepted)} accepted, {len(rejected)} rejected\n")
        f.write(f"# Image size:   {img_size[0]} x {img_size[1]}\n")
        f.write(f"# RMS reproj:   left={rms_L:.4f}, right={rms_R:.4f}, stereo={rms_S:.4f} px\n\n")
        f.write(f"baseline_mm = {baseline_estimated:.2f}    # nominal: {args.baseline_mm}\n")
        f.write(f"focal_px    = {focal_px:.2f}    # rectified, P1[0,0]\n\n")
        f.write(f"K1 (left intrinsics):\n{K1}\n\n")
        f.write(f"D1 (left distortion k1,k2,p1,p2,k3):\n{D1.flatten()}\n\n")
        f.write(f"K2 (right intrinsics):\n{K2}\n\n")
        f.write(f"D2 (right distortion):\n{D2.flatten()}\n\n")
        f.write(f"R (rotation, left to right):\n{R}\n\n")
        f.write(f"T (translation, left to right, mm):\n{T.flatten()}\n\n")
        f.write(f"P1 (rectified left projection):\n{P1}\n\n")
        f.write(f"P2 (rectified right projection):\n{P2}\n\n")
    print(f"Saved human-readable summary to: {out_txt}")

    if accepted:
        first = accepted[0]
        imgL = cv2.imread(str(left_dir / first))
        imgR = cv2.imread(str(right_dir / first))
        rectL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)
        side = np.hstack([rectL, rectR])
        # Draw horizontal lines so you can visually verify alignment
        for y in range(0, side.shape[0], 50):
            cv2.line(side, (0, y), (side.shape[1], y), (0, 255, 0), 1)
        preview_path = in_dir / "rectification_sanity.png"
        cv2.imwrite(str(preview_path), side)
        print(f"\nSanity check: rectified pair with horizontal lines saved to "
              f"{preview_path}")
        print("Open it: corresponding points should fall on the same horizontal "
              "line. If they don't, calibration is bad.")

    print(f"\nReady to use. Next: run scripts/capture_stereo.py to grab "
          f"rectified pairs for the model.")


if __name__ == "__main__":
    main()

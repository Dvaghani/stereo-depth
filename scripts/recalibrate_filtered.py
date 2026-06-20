"""
Re-calibrate a stereo rig from existing checkerboard pairs, automatically
dropping the worst per-view outliers to lower the RMS reprojection error.

Why this exists
---------------
A high stereo RMS (e.g. > 1 px) almost always means a handful of bad capture
pairs (motion blur, near-degenerate pose, mis-detected corners) are poisoning
the global fit. OpenCV's stereoCalibrate minimises a single global objective,
so even a few bad views inflate everyone's residual and — critically — leave a
systematic vertical offset in the rectified output that destroys horizontal
stereo matching.

This script:
  1. Detects corners in all pairs (same as calibrate_stereo.py).
  2. Runs an initial calibration.
  3. Computes each pair's individual stereo reprojection error.
  4. Iteratively drops the worst pairs (above a percentile / absolute floor)
     and re-calibrates, until the RMS target is met or the minimum pair count
     is hit.
  5. Writes the same stereo_calib.npz / .txt as calibrate_stereo.py, plus a
     rectification_sanity.png, so it is a drop-in replacement.

Usage:
    python scripts/recalibrate_filtered.py \\
        --input outputs/calibration_160mm \\
        --baseline-mm 160 --pattern 9 6 --square-mm 25 \\
        --target-rms 0.5 --min-pairs 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--baseline-mm", type=float, required=True)
    p.add_argument("--pattern", type=int, nargs=2, default=[9, 6],
                   metavar=("COLS", "ROWS"))
    p.add_argument("--square-mm", type=float, default=25.0)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--target-rms", type=float, default=0.5,
                   help="Stop dropping pairs once stereo RMS <= this (px).")
    p.add_argument("--min-pairs", type=int, default=12,
                   help="Never drop below this many pairs.")
    p.add_argument("--drop-per-round", type=int, default=1,
                   help="How many worst pairs to drop each iteration.")
    p.add_argument("--no-write", action="store_true",
                   help="Diagnose only; do not overwrite stereo_calib.*")
    return p.parse_args()


def per_view_stereo_error(objp, ptsL, ptsR, K1, D1, K2, D2, R, T):
    """Mean reprojection error (px) for one stereo view.

    Recovers the board pose from the left camera, then reprojects the 3D
    points into BOTH cameras and averages the pixel error over both.
    """
    ok, rvec, tvec = cv2.solvePnP(objp, ptsL, K1, D1)
    if not ok:
        return 1e9
    projL, _ = cv2.projectPoints(objp, rvec, tvec, K1, D1)
    # Compose left->board with left->right extrinsics to get right pose
    Rm, _ = cv2.Rodrigues(rvec)
    R_r = R @ Rm
    t_r = R @ tvec + T.reshape(3, 1)
    rvec_r, _ = cv2.Rodrigues(R_r)
    projR, _ = cv2.projectPoints(objp, rvec_r, t_r, K2, D2)
    eL = np.linalg.norm(projL.reshape(-1, 2) - ptsL.reshape(-1, 2), axis=1)
    eR = np.linalg.norm(projR.reshape(-1, 2) - ptsR.reshape(-1, 2), axis=1)
    return float(np.concatenate([eL, eR]).mean())


def calibrate(objpoints, imgpointsL, imgpointsR, img_size):
    rms_L, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, img_size, None, None)
    rms_R, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, img_size, None, None)
    rms_S, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, K1, D1, K2, D2, img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    return dict(rms_L=rms_L, rms_R=rms_R, rms_S=rms_S,
                K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F)


def main():
    args = parse_args()
    pattern = tuple(args.pattern)
    in_dir = args.input
    left_dir, right_dir = in_dir / "left", in_dir / "right"

    left_files = sorted(left_dir.glob("pair_*.png"))
    right_files = sorted(right_dir.glob("pair_*.png"))
    if not left_files or len(left_files) != len(right_files):
        raise SystemExit(f"Bad/empty pair set in {in_dir}")

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
                   .astype(np.float32) * args.square_mm)
    refine = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    names, objpoints, ptsL, ptsR = [], [], [], []
    img_size = None
    for lf, rf in zip(left_files, right_files):
        imgL, imgR = cv2.imread(str(lf)), cv2.imread(str(rf))
        if img_size is None:
            img_size = (imgL.shape[1], imgL.shape[0])
        gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        fL, cL = cv2.findChessboardCorners(gL, pattern, flags=flags)
        fR, cR = cv2.findChessboardCorners(gR, pattern, flags=flags)
        if not (fL and fR):
            continue
        cL = cv2.cornerSubPix(gL, cL, (11, 11), (-1, -1), refine)
        cR = cv2.cornerSubPix(gR, cR, (11, 11), (-1, -1), refine)
        names.append(lf.name); objpoints.append(objp)
        ptsL.append(cL); ptsR.append(cR)

    print(f"Detected checkerboard in {len(names)}/{len(left_files)} pairs")

    keep = list(range(len(names)))
    history = []
    while True:
        op = [objpoints[i] for i in keep]
        pl = [ptsL[i] for i in keep]
        pr = [ptsR[i] for i in keep]
        cal = calibrate(op, pl, pr, img_size)
        errs = np.array([
            per_view_stereo_error(objpoints[i], ptsL[i], ptsR[i],
                                  cal["K1"], cal["D1"], cal["K2"], cal["D2"],
                                  cal["R"], cal["T"])
            for i in keep
        ])
        history.append((len(keep), cal["rms_S"]))
        worst_local = int(np.argmax(errs))
        print(f"  pairs={len(keep):2d}  stereo_RMS={cal['rms_S']:.3f}px  "
              f"worst='{names[keep[worst_local]]}'({errs[worst_local]:.2f}px)  "
              f"mean_view_err={errs.mean():.2f}px")

        if cal["rms_S"] <= args.target_rms:
            print(f"  -> target RMS {args.target_rms} reached.")
            break
        if len(keep) - args.drop_per_round < args.min_pairs:
            print(f"  -> min pairs ({args.min_pairs}) reached; stopping.")
            break
        # drop the worst pair(s)
        order = np.argsort(errs)[::-1][:args.drop_per_round]
        drop = sorted([keep[j] for j in order], reverse=True)
        for idx in drop:
            keep.remove(idx)

    # Final calibration on the kept set
    op = [objpoints[i] for i in keep]; pl = [ptsL[i] for i in keep]; pr = [ptsR[i] for i in keep]
    cal = calibrate(op, pl, pr, img_size)
    kept_names = [names[i] for i in keep]
    dropped = [n for n in names if n not in kept_names]
    print(f"\nFINAL: {len(keep)} pairs kept, stereo RMS = {cal['rms_S']:.3f} px")
    print(f"  dropped: {dropped}")

    baseline = float(np.linalg.norm(cal["T"]))
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cal["K1"], cal["D1"], cal["K2"], cal["D2"], img_size,
        cal["R"], cal["T"], flags=cv2.CALIB_ZERO_DISPARITY, alpha=args.alpha)
    map1L, map2L = cv2.initUndistortRectifyMap(cal["K1"], cal["D1"], R1, P1, img_size, cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(cal["K2"], cal["D2"], R2, P2, img_size, cv2.CV_32FC1)
    focal_px = float(P1[0, 0])
    print(f"  baseline={baseline:.2f}mm (nominal {args.baseline_mm})  focal={focal_px:.2f}px")

    if args.no_write:
        print("\n--no-write set: not overwriting calibration files.")
        return

    out_npz = in_dir / "stereo_calib.npz"
    np.savez(out_npz, K1=cal["K1"], D1=cal["D1"], K2=cal["K2"], D2=cal["D2"],
             R=cal["R"], T=cal["T"], E=cal["E"], F=cal["F"],
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             map1L=map1L, map2L=map2L, map1R=map1R, map2R=map2R,
             image_size=np.array(img_size, dtype=np.int32),
             baseline_mm=np.array([baseline]), focal_px=np.array([focal_px]),
             rms_stereo=np.array([cal["rms_S"]]))
    out_txt = in_dir / "stereo_calib.txt"
    with open(out_txt, "w") as f:
        f.write("# Stereo calibration (outlier-filtered)\n")
        f.write(f"# Kept {len(keep)} pairs; dropped {len(dropped)}: {dropped}\n")
        f.write(f"# Image size:   {img_size[0]} x {img_size[1]}\n")
        f.write(f"# RMS reproj:   left={cal['rms_L']:.4f}, right={cal['rms_R']:.4f}, "
                f"stereo={cal['rms_S']:.4f} px\n\n")
        f.write(f"baseline_mm = {baseline:.2f}    # nominal: {args.baseline_mm}\n")
        f.write(f"focal_px    = {focal_px:.2f}    # rectified, P1[0,0]\n\n")
        f.write(f"K1 (left intrinsics):\n{cal['K1']}\n\n")
        f.write(f"D1 (left distortion k1,k2,p1,p2,k3):\n{cal['D1'].flatten()}\n\n")
        f.write(f"K2 (right intrinsics):\n{cal['K2']}\n\n")
        f.write(f"D2 (right distortion):\n{cal['D2'].flatten()}\n\n")
        f.write(f"R (rotation, left to right):\n{cal['R']}\n\n")
        f.write(f"T (translation, left to right, mm):\n{cal['T'].flatten()}\n\n")
        f.write(f"P1 (rectified left projection):\n{P1}\n\n")
        f.write(f"P2 (rectified right projection):\n{P2}\n\n")

    # Sanity preview on the first kept pair
    first = kept_names[0]
    imgL = cv2.imread(str(left_dir / first)); imgR = cv2.imread(str(right_dir / first))
    rectL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)
    side = np.hstack([rectL, rectR])
    for y in range(0, side.shape[0], 50):
        cv2.line(side, (0, y), (side.shape[1], y), (0, 255, 0), 1)
    cv2.imwrite(str(in_dir / "rectification_sanity.png"), side)
    print(f"\nWrote {out_npz}, {out_txt}, rectification_sanity.png")
    print("Next: re-run scripts/capture_stereo.py (or re-rectify existing raws) "
          "and re-check the vertical residual.")


if __name__ == "__main__":
    main()

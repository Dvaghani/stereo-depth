"""
Re-solve stereo extrinsics from the station captures themselves.

Why this exists
----------------
A calibration fixes ONE relative pose. If the rig drifts between the
calibration capture and the measurement captures -- or, worse, DURING the
calibration capture, which shows up as an excellent per-camera RMS next to a
bad stereo RMS -- then the stored extrinsics do not describe the geometry that
actually produced the measurements.

The station captures are themselves checkerboard views at known-ish distances,
so they can re-solve the extrinsics for the exact geometry they were taken
under. Intrinsics are held FIXED: they come from the per-camera solves, which
are unaffected by rig drift (each camera's own optics did not change), and
re-fitting them from frontoparallel boards would be badly conditioned.

This is the same idea Chapter 7 reports for the 160mm session, where
re-solving from the station captures moved the 6m error from +208mm to +26mm
and the overall scale from +2.17% to +0.03%.

What it CANNOT do
------------------
Frontoparallel boards at varying distance constrain the baseline well but
constrain rotation about the optical axis weakly. If the reported rotation
change is large, treat the result with suspicion rather than relief. The
residual vertical disparity printed at the end is the honest check: it is
measured on the same corners, so a low value means the new pose actually
explains these images.

Usage:
    python scripts/solve_extrinsics_from_stations.py \\
        --root  outputs/accuracy_160mm_laser \\
        --calib outputs/calibration_160mm/stereo_calib.npz \\
        --out   outputs/calibration_160mm/stereo_calib_atcapture.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CHESS_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
REFINE_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def subpix_window(corners, pattern):
    """Half-window sized from apparent square spacing, so it stays inside one
    square -- a fixed 11x11 straddles neighbouring corners at long range."""
    c = corners.reshape(pattern[1], pattern[0], 2)
    spacing = np.median(np.linalg.norm(np.diff(c, axis=1), axis=2))
    h = int(max(2, min(11, spacing / 3.0)))
    return (h, h)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pattern", type=int, nargs=2, default=[9, 6])
    p.add_argument("--square-mm", type=float, default=25.0)
    args = p.parse_args()
    pattern = tuple(args.pattern)

    cal = np.load(args.calib)
    K1, D1, K2, D2 = cal["K1"], cal["D1"], cal["K2"], cal["D2"]
    img_size = tuple(int(v) for v in cal["image_size"])
    R_old, T_old = cal["R"], cal["T"].reshape(3)

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= args.square_mm

    objpoints, ptsL, ptsR, used = [], [], [], []
    for st in sorted(args.root.glob("z*m")):
        for cap in sorted(st.glob("capture_*")):
            lp, rp = cap / "left_raw.png", cap / "right_raw.png"
            if not (lp.exists() and rp.exists()):
                continue
            gl = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
            gr = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
            if gl is None or gr is None:
                continue
            okl, cl = cv2.findChessboardCorners(gl, pattern, flags=CHESS_FLAGS)
            okr, cr = cv2.findChessboardCorners(gr, pattern, flags=CHESS_FLAGS)
            if not (okl and okr):
                continue
            cl = cv2.cornerSubPix(gl, cl, subpix_window(cl, pattern), (-1, -1), REFINE_CRIT)
            cr = cv2.cornerSubPix(gr, cr, subpix_window(cr, pattern), (-1, -1), REFINE_CRIT)
            objpoints.append(objp)
            ptsL.append(cl)
            ptsR.append(cr)
            used.append(f"{st.name}/{cap.name}")

    print(f"usable station pairs: {len(objpoints)}")
    for u in used:
        print(f"  {u}")
    if len(objpoints) < 6:
        raise SystemExit(f"only {len(objpoints)} usable pairs -- too few to solve extrinsics")

    print("\nsolving extrinsics with intrinsics held FIXED ...")
    rms, *_ , R, T, E, F = cv2.stereoCalibrate(
        objpoints, ptsL, ptsR, K1, D1, K2, D2, img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5))

    B_old = float(np.linalg.norm(T_old))
    B_new = float(np.linalg.norm(T))
    dR, _ = cv2.Rodrigues(R @ R_old.T)
    ang = float(np.degrees(np.linalg.norm(dR)))
    # decompose the rotation change -- only yaw displaces horizontal disparity
    rx, ry, rz = np.degrees(dR.ravel())

    print(f"\n  stereo RMS      : {rms:.4f} px   (was {float(cal['rms_stereo'][0]):.4f} px)"
          if "rms_stereo" in cal else f"\n  stereo RMS      : {rms:.4f} px")
    print(f"  baseline        : {B_new:.2f} mm  (was {B_old:.2f} mm, "
          f"change {B_new-B_old:+.2f} mm)")
    print(f"  rotation change : {ang:.4f} deg   (pitch {rx:+.4f}, yaw {ry:+.4f}, roll {rz:+.4f})")
    print(f"                    yaw is the axis that displaces disparity")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    map1L, map2L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)
    focal_px = float(P1[0, 0])
    print(f"  rectified focal : {focal_px:.2f} px  (was {float(cal['focal_px'][0]):.2f} px)")

    # Honest check: residual vertical disparity of the SAME corners after
    # rectifying with the new pose. Low value => the pose explains these images.
    dys = []
    for cl, cr in zip(ptsL, ptsR):
        a = cv2.undistortPoints(cl, K1, D1, R=R1, P=P1).reshape(-1, 2)
        b = cv2.undistortPoints(cr, K2, D2, R=R2, P=P2).reshape(-1, 2)
        dys.append(np.abs(a[:, 1] - b[:, 1]))
    dys = np.concatenate(dys)
    print(f"\n  residual |dy| on these corners: median {np.median(dys):.3f} px, "
          f"p90 {np.percentile(dys,90):.3f} px  (target < 1 px)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T.reshape(3, 1), E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             map1L=map1L, map2L=map2L, map1R=map1R, map2R=map2R,
             image_size=np.array(img_size, dtype=np.int32),
             baseline_mm=np.array([B_new]), focal_px=np.array([focal_px]),
             rms_stereo=np.array([rms]))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

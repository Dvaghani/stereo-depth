"""
Quick re-calibration after rig drift — NO checkerboard needed.

The intrinsics (focal, distortion) from the full calibration stay valid as
long as focus is locked. Only the relative rotation between the cameras
drifts (mount flex, cable pull). This script re-estimates that rotation from
SIFT feature matches on any texture-rich scene and writes an updated
stereo_calib.npz with corrected rectification maps.

Point the rig at a texture-rich scene (bookshelf, cluttered desk — NOT a
blank wall) and run:

    python scripts/quick_recalib.py \
        --calib outputs/calibration_160mm/stereo_calib.npz \
        --out   outputs/calibration_160mm/stereo_calib_refreshed.npz

It captures a frame from both cameras, matches features, computes the new
relative pose with the KNOWN intrinsics (5-point essential matrix), and
rebuilds the rectification maps. Baseline length cannot be recovered from
features alone — it is kept from the original calibration (the physical
distance between cameras doesn't drift, only the angle).

Verify: run capture_stereo.py with the refreshed calib and check that the
green lines hit the same features in both images.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


def _set_ctrl(dev, name, val):
    try:
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", f"{name}={val}"],
                       check=True, capture_output=True)
    except Exception:
        pass


def lock_focus(index, focus):
    dev = f"/dev/video{index}"
    _set_ctrl(dev, "focus_automatic_continuous", 0)
    _set_ctrl(dev, "focus_absolute", focus)


def read_calib_focus(calib_path: Path):
    try:
        return int((calib_path.parent / "focus.txt").read_text().strip())
    except Exception:
        return None


def capture_pair(left_idx, right_idx, w, h, focus):
    def open_cam(idx):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {idx}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(5): cap.read()
        return cap

    capL, capR = open_cam(left_idx), open_cam(right_idx)
    if focus is not None:
        lock_focus(left_idx, focus)
        lock_focus(right_idx, focus)
    # let exposure/focus settle
    for _ in range(15):
        capL.read(); capR.read()
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    capL.release(); capR.release()
    if not (retL and retR):
        raise SystemExit("Camera read failed")
    return frameL, frameR


def match_features(grayL, grayR, max_feats=4000):
    sift = cv2.SIFT_create(nfeatures=max_feats)
    kL, dL = sift.detectAndCompute(grayL, None)
    kR, dR = sift.detectAndCompute(grayR, None)
    if dL is None or dR is None:
        raise SystemExit("No features found — point at a texture-rich scene.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(dL, dR, k=2)
    ptsL, ptsR = [], []
    for m, n in raw:
        if m.distance < 0.75 * n.distance:      # Lowe ratio test
            ptsL.append(kL[m.queryIdx].pt)
            ptsR.append(kR[m.trainIdx].pt)
    return np.float32(ptsL), np.float32(ptsR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", type=Path, required=True,
                   help="Existing stereo_calib.npz (intrinsics are reused).")
    p.add_argument("--out",   type=Path, default=None,
                   help="Output npz. Default: <calib_dir>/stereo_calib_refreshed.npz")
    p.add_argument("--left",  type=Path, default=None,
                   help="Optional: use saved RAW left image instead of capturing.")
    p.add_argument("--right", type=Path, default=None,
                   help="Optional: use saved RAW right image instead of capturing.")
    p.add_argument("--left-index",  type=int, default=0)
    p.add_argument("--right-index", type=int, default=4)
    p.add_argument("--focus", type=int, default=None)
    args = p.parse_args()

    calib = np.load(args.calib)
    K1, D1 = calib["K1"], calib["D1"]
    K2, D2 = calib["K2"], calib["D2"]
    T_orig = calib["T"].reshape(3)
    img_size = tuple(int(x) for x in calib["image_size"])   # (W, H)
    baseline_mm = float(calib["baseline_mm"][0])
    W, H = img_size

    out_path = args.out or (args.calib.parent / "stereo_calib_refreshed.npz")

    # ── Get a raw (UNRECTIFIED) pair ─────────────────────────────────────────
    if args.left and args.right:
        frameL = cv2.imread(str(args.left))
        frameR = cv2.imread(str(args.right))
        if frameL is None or frameR is None:
            raise SystemExit("Could not read the provided images.")
        print(f"Using saved pair: {args.left}, {args.right}")
        print("NOTE: these must be RAW captures (left_raw.png), not rectified!")
    else:
        focus = args.focus if args.focus is not None else read_calib_focus(args.calib)
        print(f"Capturing from cameras (focus={focus})...")
        frameL, frameR = capture_pair(args.left_index, args.right_index, W, H, focus)

    # ── Undistort matched points with KNOWN intrinsics ───────────────────────
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    ptsL, ptsR = match_features(grayL, grayR)
    print(f"Matches after ratio test: {len(ptsL)}")
    if len(ptsL) < 100:
        raise SystemExit("Too few matches (<100). Use a more textured scene.")

    # Normalize through the known intrinsics → essential matrix geometry
    ptsL_n = cv2.undistortPoints(ptsL.reshape(-1, 1, 2), K1, D1)
    ptsR_n = cv2.undistortPoints(ptsR.reshape(-1, 1, 2), K2, D2)

    E, inl = cv2.findEssentialMat(
        ptsL_n, ptsR_n, np.eye(3),
        method=cv2.RANSAC, prob=0.999, threshold=1.5 / float(K1[0, 0]))
    n_inl = int(inl.sum())
    print(f"Essential-matrix inliers: {n_inl}/{len(ptsL)}")
    if n_inl < 60:
        raise SystemExit("Too few inliers — bad scene or cameras moved during capture.")

    # Only the relative ROTATION drifts (the physical bar keeps its length and,
    # to first order, its direction). Re-estimating translation from features
    # is noisy, so keep T fixed and optimize a small rotation correction that
    # minimizes the vertical disparity of the inlier matches after
    # rectification. Seeded from the original calibration.
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as SciRot

    R_orig = calib["R"]
    T_new = T_orig.copy()
    inl_mask = inl.ravel().astype(bool)
    pL = ptsL[inl_mask].reshape(-1, 1, 2)
    pR = ptsR[inl_mask].reshape(-1, 1, 2)

    # Parameters: 3 for rotation correction, 2 for baseline-direction tilt
    # (baseline LENGTH stays fixed — the physical bar doesn't stretch).
    T_len = np.linalg.norm(T_orig)

    def unpack(params):
        R_test = SciRot.from_rotvec(params[:3]).as_matrix() @ R_orig
        T_test = SciRot.from_rotvec([0.0, params[3], params[4]]).as_matrix() @ T_orig
        T_test = T_test / np.linalg.norm(T_test) * T_len
        return R_test, T_test

    def vertical_residuals(params):
        R_test, T_test = unpack(params)
        R1t, R2t, P1t, P2t, _, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R_test, T_test,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        a = cv2.undistortPoints(pL, K1, D1, R=R1t, P=P1t).reshape(-1, 2)
        b = cv2.undistortPoints(pR, K2, D2, R=R2t, P=P2t).reshape(-1, 2)
        return a[:, 1] - b[:, 1]

    res0 = vertical_residuals(np.zeros(5))
    print(f"Vertical disparity BEFORE: median {np.median(np.abs(res0)):.2f}px")

    sol = least_squares(vertical_residuals, np.zeros(5),
                        loss="soft_l1", f_scale=2.0, diff_step=1e-4)
    R_new, T_new = unpack(sol.x)

    # Report how much the rig drifted
    ang = np.degrees(np.linalg.norm(sol.x[:3]))
    print(f"Rotation drift vs old calibration: {ang:.3f}°")
    print(f"Translation direction old: {T_orig / np.linalg.norm(T_orig)}")
    print(f"Translation direction new: {T_new  / np.linalg.norm(T_new)}")

    # ── Rebuild rectification with old intrinsics + new extrinsics ──────────
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R_new, T_new,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    map1L, map2L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)
    focal_px = float(P1[0, 0])

    # ── Sanity check: residual vertical disparity on the inlier matches ─────
    inl_mask = inl.ravel().astype(bool)
    rectL_pts = cv2.undistortPoints(ptsL[inl_mask].reshape(-1, 1, 2), K1, D1, R=R1, P=P1).reshape(-1, 2)
    rectR_pts = cv2.undistortPoints(ptsR[inl_mask].reshape(-1, 1, 2), K2, D2, R=R2, P=P2).reshape(-1, 2)
    dy = rectL_pts[:, 1] - rectR_pts[:, 1]
    print(f"Residual vertical disparity: median {np.median(np.abs(dy)):.2f}px, "
          f"p90 {np.percentile(np.abs(dy), 90):.2f}px  (target: < 1px)")

    np.savez(out_path,
             K1=K1, D1=D1, K2=K2, D2=D2,
             R=R_new, T=T_new.reshape(3, 1),
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             map1L=map1L, map2L=map2L, map1R=map1R, map2R=map2R,
             image_size=np.array(img_size),
             baseline_mm=np.array([baseline_mm]),
             focal_px=np.array([focal_px]))
    print(f"\nSaved → {out_path}")
    print(f"  focal={focal_px:.1f}px  baseline={baseline_mm:.1f}mm (kept from original)")
    print(f"\nVerify with:")
    print(f"  python scripts/capture_stereo.py --calib {out_path}")


if __name__ == "__main__":
    main()

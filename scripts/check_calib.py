"""
Check whether a stereo calibration is still good for the CURRENT rig state.

Captures a live pair (or takes a saved RAW pair), rectifies with the given
calibration, matches SIFT features, and reports the residual vertical
disparity. On a perfectly rectified pair, matched features lie on the same
row in both images.

    median |dy| < 1 px   → calibration good
    median |dy| 1-3 px   → marginal, depth noisier than it should be
    median |dy| > 3 px   → rig drifted, run quick_recalib.py or recalibrate

Usage (live):
    python scripts/check_calib.py --calib outputs/calibration_160mm/stereo_calib.npz

Usage (saved raw pair):
    python scripts/check_calib.py --calib ... --left left_raw.png --right right_raw.png
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _set_ctrl(dev, name, val):
    try:
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", f"{name}={val}"],
                       check=True, capture_output=True)
    except Exception:
        pass


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
        for idx in (left_idx, right_idx):
            dev = f"/dev/video{idx}"
            _set_ctrl(dev, "focus_automatic_continuous", 0)
            _set_ctrl(dev, "focus_absolute", focus)
    for _ in range(15):
        capL.read(); capR.read()
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    capL.release(); capR.release()
    if not (retL and retR):
        raise SystemExit("Camera read failed")
    return frameL, frameR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--left",  type=Path, default=None, help="Saved RAW left image")
    p.add_argument("--right", type=Path, default=None, help="Saved RAW right image")
    p.add_argument("--left-index",  type=int, default=0)
    p.add_argument("--right-index", type=int, default=4)
    p.add_argument("--focus", type=int, default=None)
    args = p.parse_args()

    calib = np.load(args.calib)
    W, H = (int(x) for x in calib["image_size"])

    if args.left and args.right:
        frameL = cv2.imread(str(args.left))
        frameR = cv2.imread(str(args.right))
        if frameL is None or frameR is None:
            raise SystemExit("Could not read images.")
        print("Using saved RAW pair (must be unrectified!)")
    else:
        focus = args.focus if args.focus is not None else read_calib_focus(args.calib)
        print(f"Capturing live pair (focus={focus})...")
        frameL, frameR = capture_pair(args.left_index, args.right_index, W, H, focus)

    rectL = cv2.remap(frameL, calib["map1L"], calib["map2L"], cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, calib["map1R"], calib["map2R"], cv2.INTER_LINEAR)
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=4000)
    kL, dL = sift.detectAndCompute(grayL, None)
    kR, dR = sift.detectAndCompute(grayR, None)
    if dL is None or dR is None:
        raise SystemExit("No features — aim at a textured scene.")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dL, dR, k=2)

    dys, dxs = [], []
    for a, b in matches:
        if a.distance < 0.75 * b.distance:
            ptL = kL[a.queryIdx].pt
            ptR = kR[a.trainIdx].pt
            dys.append(ptL[1] - ptR[1])
            dxs.append(ptL[0] - ptR[0])   # horizontal disparity (should be >= 0)
    dys = np.array(dys); dxs = np.array(dxs)
    n = len(dys)
    print(f"\nMatches: {n}")
    if n < 50:
        raise SystemExit("Too few matches for a reliable check — use a more textured scene.")

    med  = np.median(np.abs(dys))
    p90  = np.percentile(np.abs(dys), 90)
    neg  = 100.0 * (dxs < -1.0).mean()

    print(f"Residual vertical disparity |dy|:  median {med:.2f}px   p90 {p90:.2f}px")
    print(f"Negative horizontal disparity:     {neg:.0f}%  (should be ~0%)")

    if med < 1.0:
        print("\n→ GOOD: calibration is valid for the current rig state.")
    elif med < 3.0:
        print("\n→ MARGINAL: usable but depth will be noisier than necessary.")
        print("  Consider: python scripts/quick_recalib.py --calib", args.calib)
    else:
        print("\n→ BAD: rig has drifted, rectification is broken.")
        print("  Fix:      python scripts/quick_recalib.py --calib", args.calib)

    # Save a visual check image
    side = np.hstack([rectL, rectR])
    for y in range(0, side.shape[0], 60):
        cv2.line(side, (0, y), (side.shape[1], y), (0, 255, 0), 1)
    out = args.calib.parent / "check_calib_lines.png"
    cv2.imwrite(str(out), side)
    print(f"\nVisual check saved → {out}")
    print("(zoom in: the same object corner must sit on the same green line in both halves)")


if __name__ == "__main__":
    main()

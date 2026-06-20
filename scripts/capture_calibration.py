"""
Live capture of stereo image pairs for camera calibration.
Usage:
    python scripts/capture_calibration.py \\
        --baseline-mm 160 \\
        --pattern 9 6 --square-mm 25
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


LEFT_IDX = 0    # /dev/video0
RIGHT_IDX = 4   # /dev/video4

DEFAULT_PATTERN = (9, 6)   # inner corners (cols, rows)
DEFAULT_SQUARE_MM = 25.0
DEFAULT_W = 1920
DEFAULT_H = 1080
PREVIEW_SCALE = 0.5         # preview is 50% of capture size

# Focus is AUTO-SETTLED then LOCKED by default (focus=None): we let the camera
# autofocus on your board, read the value back, and pin it. Pass --focus to
# override with an explicit value instead.
# Exposure and white balance are intentionally left on AUTO — neither affects
# calibration geometry (corner detection is done on grayscale), and pinning them
# to wrong values caused a dark/blue feed.
DEFAULT_FOCUS = None


def _set_ctrl(dev: str, name: str, val) -> bool:
    try:
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", f"{name}={val}"],
                       check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        print("  WARNING: v4l2-ctl not found — install v4l-utils to control "
              "focus/exposure.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  note: could not set {name} on {dev} "
              f"({(e.stderr or '').strip() or 'unsupported'})")
        return False


def _get_ctrl(dev: str, name: str):
    """Return the current integer value of a v4l2 control, or None."""
    try:
        out = subprocess.run(["v4l2-ctl", "-d", dev, "--get-ctrl", name],
                             check=True, capture_output=True, text=True).stdout
        # format: "focus_absolute: 51"
        return int(out.split(":")[1].strip())
    except Exception:
        return None


def enable_autos(index: int) -> None:
    """Turn AF / auto-exposure / auto-WB ON so the camera can settle on the scene."""
    dev = f"/dev/video{index}"
    _set_ctrl(dev, "focus_automatic_continuous", 1)
    _set_ctrl(dev, "auto_exposure", 3)            # 3 = aperture priority (auto)
    _set_ctrl(dev, "white_balance_automatic", 1)


def lock_focus(index: int, focus: int) -> None:
    """Pin focus only (leaves exposure and white balance on auto)."""
    dev = f"/dev/video{index}"
    _set_ctrl(dev, "focus_automatic_continuous", 0)
    _set_ctrl(dev, "focus_absolute", focus)
    print(f"  {dev}: LOCKED focus={focus} (exposure & WB left auto)")


def make_capture(index: int, w: int, h: int,
                 focus: int | None = DEFAULT_FOCUS,
                 lock: bool = True) -> cv2.VideoCapture:
    if lock:
        enable_autos(index)   # let it focus while we warm up below
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # Warm up AND let autofocus converge on the scene in front of it.
    for _ in range(40):
        cap.read()
    if lock:
        dev = f"/dev/video{index}"
        settled_focus = focus if focus is not None else _get_ctrl(dev, "focus_absolute")
        if settled_focus is None:
            print(f"  WARNING: could not read settled focus on {dev}; "
                  f"leaving camera on auto.")
        else:
            lock_focus(index, settled_focus)
            for _ in range(5):
                cap.read()
    return cap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-mm", type=int, default=160,
                   help="Baseline label (mm) for the output folder. Use the "
                        "actual baseline you've set on the rig: 110, 160, or 280.")
    p.add_argument("--pattern", type=int, nargs=2, default=list(DEFAULT_PATTERN),
                   metavar=("COLS", "ROWS"),
                   help="Checkerboard INNER corners (cols rows). Default: 9 6.")
    p.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM,
                   help="Square size in mm. Default: 25.")
    p.add_argument("--width", type=int, default=DEFAULT_W)
    p.add_argument("--height", type=int, default=DEFAULT_H)
    p.add_argument("--left-index", type=int, default=LEFT_IDX)
    p.add_argument("--right-index", type=int, default=RIGHT_IDX)
    p.add_argument("--focus", type=int, default=DEFAULT_FOCUS,
                   help="Fixed focus value (0-255 on Brio). Default: auto-settle on "
                        "the board, then lock. If you set it, reuse the SAME value "
                        "for live depth capture. Adjustable live with [ and ].")
    p.add_argument("--no-lock", action="store_true",
                   help="Do NOT lock focus/exposure (debug only — produces an "
                        "unstable calibration on autofocus webcams).")
    return p.parse_args()


def main():
    args = parse_args()
    pattern = tuple(args.pattern)

    out_root = (Path(__file__).resolve().parent.parent
                / "outputs" / f"calibration_{args.baseline_mm}mm")
    left_dir = out_root / "left"
    right_dir = out_root / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(left_dir.glob("pair_*.png"))
    next_idx = len(existing) + 1
    print(f"Output: {out_root}  (starting at pair_{next_idx:03d})")
    print(f"Pattern: {pattern[0]}x{pattern[1]} inner corners, {args.square_mm} mm squares")

    lock = not args.no_lock
    print("Auto-settling then locking focus (exposure & WB stay auto)..." if lock
          else "WARNING: --no-lock set; autofocus will drift between shots.")
    capL = make_capture(args.left_index, args.width, args.height,
                        focus=args.focus, lock=lock)
    capR = make_capture(args.right_index, args.width, args.height,
                        focus=args.focus, lock=lock)

    devL, devR = f"/dev/video{args.left_index}", f"/dev/video{args.right_index}"
    # Track the current locked focus (read back from the left camera) so the live
    # [ / ] keys can nudge BOTH cameras together.
    cur_focus = _get_ctrl(devL, "focus_absolute") if lock else None
    if lock and cur_focus is not None:
        # Persist the focus next to the calibration so capture_stereo.py can lock
        # the live depth feed to the EXACT same value automatically (no guessing).
        focus_file = out_root / "focus.txt"
        focus_file.write_text(f"{cur_focus}\n")
        print(f"  -> Settled focus={cur_focus}; wrote {focus_file}.")
        print(f"  -> capture_stereo.py will auto-lock the live feed to this value.\n")
    elif lock:
        print("  -> WARNING: could not read settled focus; live capture must be "
              "locked manually with --focus.\n")

    refine_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    chess_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                   | cv2.CALIB_CB_NORMALIZE_IMAGE
                   | cv2.CALIB_CB_FAST_CHECK)

    cv2.namedWindow("Stereo calibration capture", cv2.WINDOW_NORMAL)
    print()
    print("Controls:")
    print("  SPACE   - capture (only when BOTH cameras detect the checkerboard)")
    print("  [ / ]   - focus  -/+   (both cameras)")
    print("  Q / ESC - quit")
    print()
    print("Tips for a calibration good enough for AANet (target stereo RMS < 0.4 px):")
    print("  - Aim for 35-50 pairs (more than you think you need)")
    print("  - COVER THE WHOLE FRAME: deliberately put the board in all 4 corners")
    print("    and along the edges, not just the centre — distortion is largest there")
    print("  - Vary DISTANCE across your operating range (~0.5 m to ~4 m): fill the")
    print("    frame up close in some, small-and-cornered far away in others")
    print("  - Vary TILT: pitch/yaw/roll the board +-30-45 deg — tilts constrain focal length")
    print("  - Mount the board FLAT on rigid backing (foamboard); a floppy sheet ruins it")
    print("  - Avoid motion blur: hold steady or pause ~0.5 s before SPACE")
    print("  - Keep the rig RIGID and do not touch focus after this session")
    print()

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not (retL and retR):
            continue

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        foundL, cornersL = cv2.findChessboardCorners(grayL, pattern, flags=chess_flags)
        foundR, cornersR = cv2.findChessboardCorners(grayR, pattern, flags=chess_flags)
        both = foundL and foundR

        visL = frameL.copy()
        visR = frameR.copy()
        if foundL:
            cv2.drawChessboardCorners(visL, pattern, cornersL, True)
        if foundR:
            cv2.drawChessboardCorners(visR, pattern, cornersR, True)

        border_color = (0, 220, 0) if both else (0, 80, 220)  # BGR
        for vis in (visL, visR):
            cv2.rectangle(vis, (0, 0), (vis.shape[1]-1, vis.shape[0]-1),
                          border_color, 8)

        combined = np.hstack([visL, visR])
        ph, pw = combined.shape[:2]
        preview = cv2.resize(combined, (int(pw * PREVIEW_SCALE), int(ph * PREVIEW_SCALE)))

        if both:
            detect = "BOTH"
        elif foundL:
            detect = "LEFT only"
        elif foundR:
            detect = "RIGHT only"
        else:
            detect = "NEITHER"
        status = f"Captured: {next_idx - 1}    Detected: {detect}"
        if lock and cur_focus is not None:
            status += f"    focus={cur_focus}"
        # Black outline, then colored fill — readable on any background
        cv2.putText(preview, status, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, status, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 1, cv2.LINE_AA)

        cv2.imshow("Stereo calibration capture", preview)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        # ── Live focus tuning (both cameras together) ─────────────────────────
        if lock and cur_focus is not None and key in (ord('['), ord(']')):
            cur_focus = int(np.clip(cur_focus + (5 if key == ord(']') else -5), 0, 255))
            for d in (devL, devR):
                _set_ctrl(d, "focus_absolute", cur_focus)
            print(f"  focus -> {cur_focus}")
        if key == ord(' ') and both:
            # Save full-resolution images (refinement is done in calibrate_stereo.py)
            stamp = f"pair_{next_idx:03d}"
            cv2.imwrite(str(left_dir / f"{stamp}.png"), frameL)
            cv2.imwrite(str(right_dir / f"{stamp}.png"), frameR)
            print(f"  saved {stamp}.png")
            next_idx += 1
            # White flash for visual feedback
            flash = np.full_like(preview, 255)
            cv2.imshow("Stereo calibration capture", flash)
            cv2.waitKey(80)

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    n = next_idx - 1
    print(f"\nTotal captured: {n} pair(s)")
    if n < 15:
        print("WARNING: fewer than 15 pairs - calibration accuracy will suffer.")
        print("Recommended: rerun and capture more before calibrating.")
    print(f"\nNext step:")
    print(f"  python scripts/calibrate_stereo.py "
          f"--input {out_root} --baseline-mm {args.baseline_mm} "
          f"--pattern {pattern[0]} {pattern[1]} --square-mm {args.square_mm}")


if __name__ == "__main__":
    main()

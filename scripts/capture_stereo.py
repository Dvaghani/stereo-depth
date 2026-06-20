"""
Capture a rectified stereo pair from the Brio rig using a saved calibration.
Output:
    outputs/capture_<baseline>mm_<timestamp>/
        left_raw.png
        right_raw.png
        left.png                 ← rectified, feed this to infer.py
        right.png                ← rectified, feed this to infer.py
        side_by_side_lines.png   ← visual check that rectification aligned
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


LEFT_IDX = 0
RIGHT_IDX = 4
DEFAULT_W = 1920
DEFAULT_H = 1080
PREVIEW_SCALE = 0.5


def _set_ctrl(dev: str, name: str, val) -> bool:
    try:
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", f"{name}={val}"],
                       check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        print("  WARNING: v4l2-ctl not found — install v4l-utils to lock focus.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  note: could not set {name} on {dev} "
              f"({(e.stderr or '').strip() or 'unsupported'})")
        return False


def lock_focus(index: int, focus: int) -> None:
    """Pin focus only (leaves exposure & white balance on auto)."""
    dev = f"/dev/video{index}"
    _set_ctrl(dev, "focus_automatic_continuous", 0)
    _set_ctrl(dev, "focus_absolute", focus)


def read_calib_focus(calib_path: Path) -> int | None:
    """Read the focus value persisted by capture_calibration.py (focus.txt
    sibling of the .npz). Returns None if absent/unreadable."""
    focus_file = calib_path.parent / "focus.txt"
    try:
        return int(focus_file.read_text().strip())
    except Exception:
        return None


def make_capture(index: int, w: int, h: int,
                 focus: int | None = None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(5):
        cap.read()
    if focus is not None:
        # CRITICAL: live depth pairs MUST be shot at the SAME focus as the
        # calibration, or the stored intrinsics/rectification are invalid.
        lock_focus(index, focus)
        for _ in range(5):
            cap.read()
    return cap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", type=Path, required=True,
                   help="Path to stereo_calib.npz from calibrate_stereo.py")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Output parent dir. Default: outputs/")
    p.add_argument("--left-index", type=int, default=LEFT_IDX)
    p.add_argument("--right-index", type=int, default=RIGHT_IDX)
    p.add_argument("--width", type=int, default=DEFAULT_W)
    p.add_argument("--height", type=int, default=DEFAULT_H)
    p.add_argument("--focus", type=int, default=None,
                   help="Fixed focus (0-255) to lock BOTH cameras to. Default: "
                        "read focus.txt next to the calibration (written by "
                        "capture_calibration.py). MUST match the calibration "
                        "focus or rectification is invalid. Adjust live with [ ].")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.calib.exists():
        raise SystemExit(f"Calibration file not found: {args.calib}")
    calib = np.load(args.calib)
    map1L = calib["map1L"]; map2L = calib["map2L"]
    map1R = calib["map1R"]; map2R = calib["map2R"]
    img_size = tuple(int(x) for x in calib["image_size"])  # (W, H)
    baseline_mm = float(calib["baseline_mm"][0])
    focal_px = float(calib["focal_px"][0])

    expected_w, expected_h = img_size
    if (args.width, args.height) != (expected_w, expected_h):
        print(f"NOTE: requested capture {args.width}x{args.height} differs from "
              f"calibration image size {expected_w}x{expected_h}. Using "
              f"calibration size — capture will be set accordingly.")
        args.width, args.height = expected_w, expected_h

    out_root = args.out_root or (Path(__file__).resolve().parent.parent / "outputs")

    # Resolve the focus to lock: explicit --focus wins, else the persisted
    # calibration focus, else leave on whatever the camera defaults to (warn).
    cur_focus = args.focus if args.focus is not None else read_calib_focus(args.calib)
    if cur_focus is not None:
        cur_focus = int(np.clip(cur_focus, 0, 255))
        src = "--focus" if args.focus is not None else "focus.txt"
        print(f"Locking focus={cur_focus} on both cameras (from {src}).")
    else:
        print("WARNING: no focus.txt next to the calibration and no --focus given.\n"
              "         The live feed may focus to a DIFFERENT value than the\n"
              "         calibration, invalidating rectification. Pass --focus <0-255>\n"
              "         matching your calibration, or re-run capture_calibration.py\n"
              "         (it now writes focus.txt automatically).")

    capL = make_capture(args.left_index, args.width, args.height, focus=cur_focus)
    capR = make_capture(args.right_index, args.width, args.height, focus=cur_focus)
    devL = f"/dev/video{args.left_index}"
    devR = f"/dev/video{args.right_index}"

    cv2.namedWindow("Stereo capture (rectified preview)", cv2.WINDOW_NORMAL)
    print(f"Calibration: baseline={baseline_mm:.1f}mm, focal={focal_px:.1f}px, "
          f"size={expected_w}x{expected_h}")
    print()
    print("Controls:")
    print("  SPACE   - capture rectified pair to disk")
    print("  [ / ]   - focus -/+ (both cameras; keep this MATCHING calibration)")
    print("  Q / ESC - quit")
    print()

    # Pick a default model checkpoint to suggest after capture.
    # KITTI is a reasonable default for outdoor / general scenes; user can
    # swap to Middlebury for indoor close-range.
    suggested_ckpt = "checkpoints/kitti_uncertainty/best.pt"

    capture_count = 0
    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not (retL and retR):
            continue

        # Rectify
        rectL = cv2.remap(frameL, map1L, map2L, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, map1R, map2R, cv2.INTER_LINEAR)

        # Side-by-side preview with horizontal reference lines
        combined = np.hstack([rectL, rectR])
        for y in range(0, combined.shape[0], 80):
            cv2.line(combined, (0, y), (combined.shape[1], y), (0, 200, 0), 1)
        ph, pw = combined.shape[:2]
        preview = cv2.resize(combined, (int(pw * PREVIEW_SCALE),
                                          int(ph * PREVIEW_SCALE)))

        status = (f"Captures: {capture_count}    "
                  f"Baseline: {baseline_mm:.0f}mm    Focal: {focal_px:.0f}px")
        if cur_focus is not None:
            status += f"    focus={cur_focus}"
        cv2.putText(preview, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Stereo capture (rectified preview)", preview)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        # Live focus tuning — keep BOTH cameras matched to the calibration focus.
        if cur_focus is not None and key in (ord('['), ord(']')):
            cur_focus = int(np.clip(cur_focus + (5 if key == ord(']') else -5), 0, 255))
            for d in (devL, devR):
                _set_ctrl(d, "focus_absolute", cur_focus)
            print(f"  focus -> {cur_focus}")
        if key == ord(' '):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_dir = out_root / f"capture_{int(baseline_mm)}mm_{timestamp}"
            out_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(out_dir / "left_raw.png"), frameL)
            cv2.imwrite(str(out_dir / "right_raw.png"), frameR)
            cv2.imwrite(str(out_dir / "left.png"), rectL)
            cv2.imwrite(str(out_dir / "right.png"), rectR)

            # Visual rectification check artifact
            side = np.hstack([rectL, rectR])
            for y in range(0, side.shape[0], 50):
                cv2.line(side, (0, y), (side.shape[1], y), (0, 255, 0), 1)
            cv2.imwrite(str(out_dir / "side_by_side_lines.png"), side)

            print(f"\nSaved capture to: {out_dir}")
            # Print the infer command — baseline in METERS for infer.py.
            # Use relative paths from the project root (the working directory
            # the user typically runs scripts from) to avoid shell quoting
            # issues with project paths that contain spaces.
            baseline_m = baseline_mm / 1000.0
            try:
                rel = out_dir.relative_to(Path(__file__).resolve().parent.parent)
            except ValueError:
                rel = out_dir
            print(f"  python scripts/infer.py \\")
            print(f"      --ckpt {suggested_ckpt} \\")
            print(f"      --left {rel}/left.png \\")
            print(f"      --right {rel}/right.png \\")
            print(f"      --out {rel}/depth \\")
            print(f"      --baseline {baseline_m:.4f} \\")
            print(f"      --focal {focal_px:.1f}")
            print(f"  (swap to checkpoints/middlebury_uncertainty/best.pt "
                  f"for indoor close-range)")
            capture_count += 1

            # Brief white flash for feedback
            flash = np.full_like(preview, 255)
            cv2.imshow("Stereo capture (rectified preview)", flash)
            cv2.waitKey(80)

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {capture_count} capture(s) saved.")


if __name__ == "__main__":
    main()

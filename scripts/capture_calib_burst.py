"""
Capture a burst of rectified stereo pairs from the Brio rig, for use as an
INT8 calibration set.

Why not capture_stereo.py
-------------------------
That script captures one pair per invocation, interactively, with a preview and
a keypress. Fine for spot checks; impractical for the ~200 pairs an INT8
calibration set wants. This one runs unattended: it captures on a timer while
you walk the rig around the scene.

Why own-rig pairs matter here
-----------------------------
build_raft_calib_set.py reports how much of a calibration set's disparity
distribution overlaps the rig's actual operating range. Public data scores
badly: a KITTI-dominated mix reached only 13% overlap and Middlebury alone 41%,
because their native aspect ratios and disparity ranges do not survive being
resized to the engine's 640x480 input. Pairs from this rig are by construction
a 100% match, so they are the best calibration source available.

Capture technique matters as much as count
------------------------------------------
The set should span the *deployment* distribution, not one static view:
  - vary distance to the nearest object across the rig's working range
  - include the near end especially, since that is where disparity is largest
    and where the correlation-volume activations peak
  - vary scene content and lighting
  - avoid capturing 200 near-identical frames of one wall -- that calibrates
    for a single operating point

Output layout is picked up directly by build_raft_calib_set.py's 'rig' source:

    outputs/rig_burst_<timestamp>/pair_0000/{left.png,right.png}

Quick calibration check happens automatically, first
-----------------------------------------------------
Every pair this script writes is rectified with the given --calib file. If the
rig has drifted since that calibration was made (mount flex, a bumped cable,
even reseating a camera), every one of the 200 pairs bakes in the same bad
rectification -- and it would be silent, because nothing here would look
obviously wrong on its own. RAFT's correlation search assumes true epipolar
alignment, so drift shows up as calibration scales tuned around a
systematically wrong disparity, not as an obvious visual error. So rather than
relying on a human to remember to run check_calib.py first, this script runs
the same check itself (SIFT feature matches on the first captured frame,
median residual vertical disparity) before starting the burst:

    < 1 px    good, burst starts immediately
    1-3 px    marginal -- warns, then proceeds (usable, just noisier)
    > 3 px    drifted -- ABORTS. Point at a texture-rich scene (bookshelf,
              cluttered desk, not a blank wall) and fix it first, no
              checkerboard needed:

    python scripts/quick_recalib.py \\
        --calib outputs/calibration_110mm/stereo_calib.npz \\
        --out   outputs/calibration_110mm/stereo_calib_refreshed.npz

then re-run this script with the refreshed file. --skip-calib-check bypasses
the gate entirely, for when it was already verified separately.

Then run the burst capture with whichever calib file passed the check:

    python scripts/capture_calib_burst.py \\
        --calib outputs/calibration_110mm/stereo_calib.npz \\
        --left-index 2 --right-index 0 --count 200 --interval 1.0

Then:
    python scripts/build_raft_calib_set.py --sources rig \\
        --rig-glob "outputs/rig_burst_*/pair_*" \\
        --out outputs/int8_calib_raft --count 200
"""

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEFAULT_W = 1920
DEFAULT_H = 1080


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib", type=Path, required=True,
                   help="stereo_calib.npz matching the rig's current baseline")
    p.add_argument("--left-index", type=int, required=True,
                   help="/dev/videoN for the PHYSICAL left camera (must match "
                        "the L/R assignment used during calibration)")
    p.add_argument("--right-index", type=int, required=True)
    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--interval", type=float, default=1.0,
                   help="seconds between captures (default 1.0)")
    p.add_argument("--width", type=int, default=DEFAULT_W)
    p.add_argument("--height", type=int, default=DEFAULT_H)
    p.add_argument("--focus", type=int, default=None,
                   help="lock focus to this value; defaults to the value saved "
                        "beside the calibration (focus.txt) if present. Focus "
                        "must match calibration or rectification degrades.")
    p.add_argument("--no-preview", action="store_true",
                   help="run headless (no window). Recommended when unattended.")
    p.add_argument("--skip-calib-check", action="store_true",
                   help="skip the automatic pre-burst drift check (e.g. if "
                        "check_calib.py was already run separately)")
    p.add_argument("--calib-check-bad-px", type=float, default=3.0,
                   help="median |dy| above this aborts the burst (default 3.0, "
                        "matching check_calib.py's BAD threshold)")
    return p.parse_args()


def quick_calib_check(rect_l, rect_r, bad_px):
    """Same method as check_calib.py: SIFT matches between the two rectified
    frames should lie on the same row (residual vertical disparity ~ 0) if
    rectification is correct. Returns (median_abs_dy, n_matches) so the caller
    can decide whether to proceed."""
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=4000)
    k_l, d_l = sift.detectAndCompute(gray_l, None)
    k_r, d_r = sift.detectAndCompute(gray_r, None)
    if d_l is None or d_r is None:
        print("  WARNING: no SIFT features found -- aim at a textured scene "
              "to run the check. Proceeding without verification.")
        return None, 0

    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d_l, d_r, k=2)
    dys = [k_l[a.queryIdx].pt[1] - k_r[a.trainIdx].pt[1]
           for a, b in matches if a.distance < 0.75 * b.distance]
    if len(dys) < 50:
        print("  WARNING: only %d good matches -- too few for a reliable check. "
              "Aim at a more textured scene. Proceeding without verification."
              % len(dys))
        return None, len(dys)

    med = float(np.median(np.abs(dys)))
    print("  residual vertical disparity: median %.2fpx over %d matches"
          % (med, len(dys)))
    if med < 1.0:
        print("  -> GOOD: calibration matches the current rig state.")
    elif med < bad_px:
        print("  -> MARGINAL: usable, but depth/disparity will be noisier "
              "than necessary. Consider quick_recalib.py before a long burst.")
    else:
        print("  -> BAD: rig has drifted since this calibration was made.")
    return med, len(dys)


def set_ctrl(dev, name, val):
    try:
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", "%s=%s" % (name, val)],
                       check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        print("  WARNING: v4l2-ctl not found -- cannot lock focus.")
        return False
    except subprocess.CalledProcessError:
        return False


def lock_focus(index, focus):
    dev = "/dev/video%d" % index
    set_ctrl(dev, "focus_automatic_continuous", 0)
    set_ctrl(dev, "focus_absolute", focus)


def open_camera(index, w, h):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit("could not open /dev/video%d" % index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(5):          # flush stale buffered frames
        cap.read()
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (w, h):
        print("  NOTE: /dev/video%d negotiated %dx%d, not the requested %dx%d"
              % (index, got[0], got[1], w, h))
    return cap


def capture_pair(cap_l, cap_r):
    """grab() both before retrieve()ing either, so the two frames are as close
    in time as USB allows -- skew shows up directly as disparity error."""
    cap_l.grab()
    cap_r.grab()
    ok_l, left = cap_l.retrieve()
    ok_r, right = cap_r.retrieve()
    if not (ok_l and ok_r):
        return None, None
    return left, right


def main():
    args = parse_args()

    calib = np.load(str(args.calib))
    native = (int(calib["image_size"][0]), int(calib["image_size"][1]))
    if (args.width, args.height) != native:
        raise SystemExit(
            "capture size %dx%d != calibration size %dx%d. The saved "
            "rectification maps only apply at the calibrated size; capture at "
            "the native size here and let build_raft_calib_set.py do the "
            "resizing." % (args.width, args.height, native[0], native[1]))
    map1L, map2L = calib["map1L"], calib["map2L"]
    map1R, map2R = calib["map1R"], calib["map2R"]

    focus = args.focus
    if focus is None:
        f = args.calib.parent / "focus.txt"
        if f.exists():
            try:
                focus = int(f.read_text().strip())
            except ValueError:
                focus = None
    if focus is not None:
        print("locking focus to %d on both cameras" % focus)
        lock_focus(args.left_index, focus)
        lock_focus(args.right_index, focus)
    else:
        print("WARNING: no focus value given or found beside the calibration. "
              "Autofocus drift between now and calibration will degrade "
              "rectification.")

    out_dir = args.out_root / ("rig_burst_%s" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print("writing to %s" % out_dir)

    cap_l = open_camera(args.left_index, args.width, args.height)
    cap_r = open_camera(args.right_index, args.width, args.height)

    first_rect = None
    if not args.skip_calib_check:
        print("\nchecking calibration against the current rig state "
              "(aim at a textured scene -- bookshelf, cluttered desk)...")
        left, right = capture_pair(cap_l, cap_r)
        if left is None:
            raise SystemExit("frame grab failed during the calibration check")
        rect_l = cv2.remap(left, map1L, map2L, cv2.INTER_LINEAR)
        rect_r = cv2.remap(right, map1R, map2R, cv2.INTER_LINEAR)
        med, _ = quick_calib_check(rect_l, rect_r, args.calib_check_bad_px)
        if med is not None and med >= args.calib_check_bad_px:
            cap_l.release(); cap_r.release()
            raise SystemExit(
                "\nABORTED: calibration looks drifted for the current rig "
                "state (median |dy|=%.2fpx >= %.1fpx). A 200-pair burst "
                "through this rectification would bake the drift into every "
                "pair. Fix it first:\n\n"
                "  python scripts/quick_recalib.py --calib %s "
                "--out %s\n\n"
                "or pass --skip-calib-check to proceed anyway."
                % (med, args.calib_check_bad_px, args.calib,
                   args.calib.with_name(args.calib.stem + "_refreshed.npz")))
        first_rect = (rect_l, rect_r)  # reuse as pair_0000, no wasted capture

    print("\ncapturing %d pairs every %.1fs (~%.1f min). Move the rig around: "
          "vary distance, especially close range.\nCtrl-C to stop early.\n"
          % (args.count, args.interval, args.count * args.interval / 60.0))

    n = 0
    try:
        while n < args.count:
            t0 = time.time()
            if first_rect is not None:
                rect_l, rect_r = first_rect
                first_rect = None
            else:
                left, right = capture_pair(cap_l, cap_r)
                if left is None:
                    print("  frame grab failed, retrying")
                    time.sleep(0.1)
                    continue
                rect_l = cv2.remap(left, map1L, map2L, cv2.INTER_LINEAR)
                rect_r = cv2.remap(right, map1R, map2R, cv2.INTER_LINEAR)

            d = out_dir / ("pair_%04d" % n)
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / "left.png"), rect_l)
            cv2.imwrite(str(d / "right.png"), rect_r)
            n += 1
            print("\r  %d/%d" % (n, args.count), end="", flush=True)

            if not args.no_preview:
                prev = cv2.resize(np.hstack([rect_l, rect_r]), None, fx=0.25, fy=0.25)
                cv2.imshow("burst capture (q to stop)", prev)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            dt = args.interval - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cap_l.release()
        cap_r.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    print("\n\ncaptured %d pairs in %s" % (n, out_dir))
    print("\nnext:\n  python scripts/build_raft_calib_set.py --sources rig \\\n"
          "      --rig-glob \"outputs/rig_burst_*/pair_*\" \\\n"
          "      --out outputs/int8_calib_raft --count %d" % n)


if __name__ == "__main__":
    main()

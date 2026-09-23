"""
Re-rectify saved captures with a different calibration.

Why this exists
---------------
capture_stereo.py writes both the raw pair (left_raw.png / right_raw.png) and
the rectified pair (left.png / right.png), the latter baked with whatever
calibration was loaded at capture time. When that calibration later turns out to
be wrong -- a stale one carried through a remount, say -- the measurements are
invalid but the *captures* are not: the raw images are untouched by the mistake.

This regenerates the rectified pair from the raw one using a corrected
calibration, so a station set does not have to be re-shot. Rectified images are
derived data; the raw pair plus a calibration is the real record. That is
exactly why capture_stereo.py saves the raw pair at all.

The rig must not have moved between the captures and the new calibration, or
the new calibration does not describe the geometry that produced these images.
There is no way to check that from the files alone -- it is on the operator.

Usage:
    python scripts/rerectify_captures.py \\
        --root outputs/accuracy_110mm \\
        --calib outputs/calibration_110mm/stereo_calib.npz
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="directory tree containing capture_* folders")
    p.add_argument("--calib", type=Path, required=True,
                   help="calibration to rectify with")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be rewritten, change nothing")
    return p.parse_args()


def main():
    args = parse_args()
    calib = np.load(args.calib)
    maps = (calib["map1L"], calib["map2L"], calib["map1R"], calib["map2R"])
    exp_w, exp_h = (int(v) for v in calib["image_size"])
    print("calibration: %s" % args.calib)
    print("  baseline %.2f mm, focal %.2f px, expects %dx%d\n"
          % (calib["baseline_mm"][0], calib["focal_px"][0], exp_w, exp_h))

    caps = sorted(args.root.rglob("capture_*"))
    caps = [c for c in caps if c.is_dir() and (c / "left_raw.png").exists()]
    if not caps:
        raise SystemExit("no capture_* folders with left_raw.png under %s" % args.root)

    done, skipped = 0, []
    for cap in caps:
        lr, rr = cap / "left_raw.png", cap / "right_raw.png"
        if not rr.exists():
            skipped.append("%s: no right_raw.png" % cap.name)
            continue
        li, ri = cv2.imread(str(lr)), cv2.imread(str(rr))
        if li is None or ri is None:
            skipped.append("%s: unreadable raw pair" % cap.name)
            continue
        h, w = li.shape[:2]
        if (w, h) != (exp_w, exp_h):
            # Silently rectifying a mismatched size would produce a plausible
            # but geometrically wrong image, so refuse instead.
            skipped.append("%s: raw is %dx%d, calibration expects %dx%d"
                           % (cap.name, w, h, exp_w, exp_h))
            continue
        if args.dry_run:
            done += 1
            continue
        cv2.imwrite(str(cap / "left.png"),
                    cv2.remap(li, maps[0], maps[1], cv2.INTER_LINEAR))
        cv2.imwrite(str(cap / "right.png"),
                    cv2.remap(ri, maps[2], maps[3], cv2.INTER_LINEAR))
        done += 1

    verb = "would rewrite" if args.dry_run else "rewrote"
    print("%s left.png/right.png in %d capture folder(s)" % (verb, done))
    for s in skipped:
        print("  SKIPPED %s" % s)
    if not args.dry_run and done:
        print("\nleft_raw.png / right_raw.png are untouched, so this is "
              "repeatable with any other calibration.")


if __name__ == "__main__":
    main()

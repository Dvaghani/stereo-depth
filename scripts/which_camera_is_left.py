"""
Decide which camera index is the physical LEFT one, from two snapshots.

Why this exists
---------------
Getting left/right backwards produces negative disparity, which breaks every
downstream measurement -- and it is easy to get wrong, because judging it by eye
while standing in front of the rig mirrors the sense. It has flipped at every
remount of this rig so far (index 0 was left at 280 mm, index 4 at 110 mm),
sometimes from a physical swap and sometimes from USB re-enumeration, so it must
be re-checked every time rather than assumed.

check_cameras.py prints "LEFT = index 0" unconditionally -- that is a default
label, not a measurement. This script measures it.

The geometry
------------
For a horizontal stereo pair, a scene point at depth Z satisfies

    x_left - x_right = f * B / Z  > 0

so the LEFT camera sees every scene point FURTHER RIGHT in its own image. The
sign of the median horizontal shift between matched features settles it, with no
dependence on which way the operator was facing.

Usage:
    python scripts/check_cameras.py            # writes the snapshots
    python scripts/which_camera_is_left.py     # reads them and decides
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "camera_check"
# Matches whose vertical offset differs wildly from the median are mismatches,
# not correspondences; a horizontal rig keeps true matches near one row.
MAX_DY_SPREAD_PX = 40.0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                   help="directory holding cam_<i>.jpg snapshots")
    p.add_argument("--indices", type=int, nargs=2, default=[0, 4],
                   metavar=("A", "B"), help="the two camera indices to compare")
    return p.parse_args()


def main():
    args = parse_args()
    a_i, b_i = args.indices
    pa, pb = args.dir / ("cam_%d.jpg" % a_i), args.dir / ("cam_%d.jpg" % b_i)
    for p in (pa, pb):
        if not p.exists():
            raise SystemExit("missing %s -- run scripts/check_cameras.py first" % p)

    A = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
    B = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
    print("sizes: %d=%s  %d=%s" % (a_i, A.shape[::-1], b_i, B.shape[::-1]))

    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(A, None)
    kb, db = sift.detectAndCompute(B, None)
    print("features: %d=%d  %d=%d" % (a_i, len(ka), b_i, len(kb)))
    if da is None or db is None:
        raise SystemExit("no features -- point the rig at a texture-rich scene")

    raw = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]   # Lowe ratio
    print("good matches after ratio test: %d" % len(good))
    if len(good) < 20:
        raise SystemExit("too few matches to be confident -- use a cluttered scene")

    dx = np.array([ka[m.queryIdx].pt[0] - kb[m.trainIdx].pt[0] for m in good])
    dy = np.array([ka[m.queryIdx].pt[1] - kb[m.trainIdx].pt[1] for m in good])
    keep = np.abs(dy - np.median(dy)) < MAX_DY_SPREAD_PX
    dx, dy = dx[keep], dy[keep]
    print("kept %d matches with consistent vertical offset" % dx.size)

    med = float(np.median(dx))
    agree = float(np.mean(dx > 0 if med > 0 else dx < 0)) * 100
    print("\nhorizontal shift (x_%d - x_%d):" % (a_i, b_i))
    print("  median  = %+.1f px" % med)
    print("  p25/p75 = %+.1f / %+.1f px" % tuple(np.percentile(dx, [25, 75])))
    print("  %.0f%% of matches agree with the median direction" % agree)
    print("  median vertical offset = %+.1f px" % float(np.median(dy)))

    left, right = (a_i, b_i) if med > 0 else (b_i, a_i)
    print("\n=> LEFT = index %d,  RIGHT = index %d" % (left, right))
    print("   (points sit further right in %d's image, so %d is displaced left)"
          % (left, left))
    if agree < 90:
        print("\n   WARNING: only %.0f%% agreement. Re-shoot the snapshots on a "
              "closer, more textured scene before trusting this." % agree)
    if left != 0:
        print("\n   NOTE: this is NOT the script default. Pass "
              "--left-index %d --right-index %d to capture_calibration.py, "
              "capture_stereo.py and check_calib.py." % (left, right))


if __name__ == "__main__":
    main()

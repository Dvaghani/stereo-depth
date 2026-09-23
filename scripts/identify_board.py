"""
Work out a checkerboard's inner-corner pattern from a photo of it.

Why this exists
----------------
cv2.findChessboardCorners needs the EXACT inner-corner count (cols, rows);
give it the wrong one and it simply returns False, with nothing to say which
count would have worked. Counting squares by eye is error-prone because the
inner-corner count is one less than the square count in each direction, and
because a board's "4 black squares across" can mean 7 or 8 columns depending
on whether the row starts and ends on black.

This brute-forces the plausible range and reports every pattern that detects,
so the board can be identified from one image instead of from a description.

It also measures the mean corner spacing in pixels, which is a sanity check
against the physical square size once you know the distance to the board.

Usage:
    python scripts/identify_board.py --image path/to/board.png
    python scripts/identify_board.py --image board.png --max-dim 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--min-dim", type=int, default=3)
    p.add_argument("--max-dim", type=int, default=12)
    p.add_argument("--save-vis", default=None,
                   help="write an image with the detected corners drawn on")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"image: {args.image}  {img.shape[1]}x{img.shape[0]}\n")

    hits = []
    for cols in range(args.min_dim, args.max_dim + 1):
        for rows in range(args.min_dim, cols + 1):   # cols >= rows by convention
            ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=FLAGS)
            if not ok:
                continue
            c = corners.reshape(rows, cols, 2)
            # spacing along each axis, in px
            dx = np.median(np.linalg.norm(np.diff(c, axis=1), axis=2))
            dy = np.median(np.linalg.norm(np.diff(c, axis=0), axis=2))
            hits.append((cols, rows, dx, dy, corners))

    if not hits:
        print("No pattern detected. Things to check:")
        print("  - the whole board must be visible, with a white margin around it")
        print("  - it must be reasonably flat and in focus")
        print("  - try a wider --max-dim if the board is large")
        return

    print(f"{'pattern':>12}  {'corners':>8}  {'spacing x':>10}  {'spacing y':>10}  {'square?':>9}")
    for cols, rows, dx, dy, _ in hits:
        square = "yes" if abs(dx - dy) / max(dx, dy) < 0.08 else "NO - skewed"
        print(f"{cols:>5} x {rows:<4}  {cols*rows:>8}  {dx:>9.1f}px  {dy:>9.1f}px  {square:>9}")

    cols, rows, dx, dy, corners = hits[0]
    if len(hits) > 1:
        print(f"\nNOTE: {len(hits)} patterns detected. The LARGEST is normally the")
        print("real board -- a smaller pattern can match a sub-region of it.")
        cols, rows, dx, dy, corners = max(hits, key=lambda h: h[0] * h[1])

    print(f"\n=> use --pattern {cols} {rows}")
    print(f"   ({cols}x{rows} inner corners = {cols+1}x{rows+1} squares, "
          f"{cols*rows} corners per view)")
    print(f"   mean corner spacing {(dx+dy)/2:.1f} px in this image")
    print("\nMeasure the SQUARE SIZE physically with a caliper, across several")
    print("squares and divided by the count -- do not infer it from this image.")

    if args.save_vis:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, (cols, rows), corners, True)
        cv2.imwrite(args.save_vis, vis)
        print(f"\nwrote {args.save_vis} -- check every corner is marked")


if __name__ == "__main__":
    main()

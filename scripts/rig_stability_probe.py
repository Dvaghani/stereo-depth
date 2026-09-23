"""
Is the rig mechanically settled enough to calibrate?

Why this exists
----------------
A stereo calibration solves ONE relative pose (R, T) for the whole capture
set. If the rig creeps while those 30-40 pairs are being taken, no single
pose fits them all and the error lands in the stereo RMS -- typically as an
excellent per-camera RMS (each camera's own images fit fine) next to a bad
stereo RMS. That signature is easy to misread as "bad images" when the real
cause is a moving rig.

This probe measures pose stability directly, WITHOUT needing a calibration,
so it can be run before spending 15 minutes on a capture that would be
invalidated anyway.

How it works
-------------
The median VERTICAL offset (dy) of SIFT correspondences between the two raw
views is a function of the relative camera pose alone. Scene content changes
the horizontal offset (that is disparity, i.e. depth), but leaves dy alone.
So sampling dy over time isolates rig motion from everything else: a stable
rig holds dy constant even as the scene changes.

Reading the result
-------------------
Drift is reported in px per 10 minutes, extrapolated from the sampling window:
    < 0.2 px / 10 min   settled -- safe to calibrate
    0.2-0.5 px / 10 min marginal -- calibration will be noisier than necessary
    > 0.5 px / 10 min   still creeping -- wait, a calibration taken now will
                        carry the motion into its stereo RMS

Usage:
    python scripts/rig_stability_probe.py --left-index 4 --right-index 0
    python scripts/rig_stability_probe.py --samples 6 --interval 120
"""
from __future__ import annotations

import argparse
import subprocess
import time

import cv2
import numpy as np


def lock_focus(idx, val):
    dev = f"/dev/video{idx}"
    for ctl in ("focus_automatic_continuous=0", f"focus_absolute={val}"):
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", ctl],
                       capture_output=True)


def grab(idx, w, h):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(12):          # let exposure settle
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"camera {idx} read failed")
    return frame


def pose_signature(left, right, nfeat=4000):
    """Median vertical offset of matches -- a pure function of relative pose."""
    sift = cv2.SIFT_create(nfeat)
    kl, dl = sift.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)
    kr, dr = sift.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
    if dl is None or dr is None:
        raise SystemExit("no features -- point the rig at a textured scene")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dl, dr, k=2)
    dy = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            dy.append(kl[m.queryIdx].pt[1] - kr[m.trainIdx].pt[1])
    if len(dy) < 30:
        raise SystemExit(f"only {len(dy)} matches -- need a more textured scene")
    dy = np.asarray(dy)
    med = np.median(dy)
    keep = np.abs(dy - med) < 20          # reject cross-row mismatches
    return float(np.median(dy[keep])), int(keep.sum())


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--left-index", type=int, required=True)
    p.add_argument("--right-index", type=int, required=True)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--interval", type=float, default=60.0, help="seconds between samples")
    p.add_argument("--focus", type=int, default=0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    args = p.parse_args()

    for idx in (args.left_index, args.right_index):
        lock_focus(idx, args.focus)
    time.sleep(1)

    print(f"probing rig pose stability: {args.samples} samples, "
          f"{args.interval:.0f}s apart "
          f"(~{args.samples * args.interval / 60:.1f} min total)\n")
    print(f"{'t [min]':>8}  {'median dy [px]':>15}  {'change':>9}  {'matches':>8}")

    t0 = time.time()
    first = None
    series = []
    for i in range(args.samples):
        if i:
            time.sleep(args.interval)
        L = grab(args.left_index, args.width, args.height)
        R = grab(args.right_index, args.width, args.height)
        dy, n = pose_signature(L, R)
        t = (time.time() - t0) / 60.0
        if first is None:
            first = dy
            print(f"{t:8.1f}  {dy:15.2f}  {'--':>9}  {n:8d}")
        else:
            print(f"{t:8.1f}  {dy:15.2f}  {dy - first:+9.2f}  {n:8d}")
        series.append((t, dy))

    if len(series) < 2:
        return
    ts = np.array([s[0] for s in series])
    ds = np.array([s[1] for s in series])
    slope, icpt = np.polyfit(ts, ds, 1)       # px per minute
    per10 = abs(slope) * 10.0
    span = ds.max() - ds.min()
    sd = float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0
    pred = slope * ts + icpt
    ss_tot = float(np.sum((ds - ds.mean()) ** 2))
    r2 = 1.0 - float(np.sum((ds - pred) ** 2)) / ss_tot if ss_tot > 0 else 0.0
    monotonic = bool(np.all(np.diff(ds) > 0) or np.all(np.diff(ds) < 0))

    print(f"\ntotal excursion : {span:.2f} px over {ts[-1]:.1f} min")
    print(f"sample std dev  : {sd:.3f} px")
    print(f"linear slope    : {abs(slope):.3f} px/min  ({per10:.2f} px per 10 min)")
    print(f"fit quality     : R^2 = {r2:.3f}, {'monotonic' if monotonic else 'NOT monotonic'}")

    # A fitted slope only means drift if a line actually describes the data.
    # Real mechanical creep is monotonic and clears the ~0.4px matching noise
    # floor; noise produces a slope too, but oscillates and fits poorly.
    NOISE_FLOOR = 0.4
    if span < NOISE_FLOOR and not monotonic:
        verdict = ("SETTLED -- excursion is within measurement noise and the "
                   "series is not monotonic, so the fitted slope is not drift")
    elif per10 < 0.2:
        verdict = "SETTLED -- safe to calibrate"
    elif not monotonic or r2 < 0.7:
        verdict = (f"PROBABLY SETTLED -- slope {per10:.2f}px/10min but the fit is "
                   f"weak (R^2={r2:.2f}"
                   f"{', not monotonic' if not monotonic else ''}), so this is "
                   f"likely noise rather than creep. Re-run to confirm if unsure")
    elif per10 < 0.5:
        verdict = "MARGINAL -- calibration will be noisier than necessary"
    else:
        verdict = ("STILL CREEPING -- wait; a calibration taken now will carry "
                   "this motion into its stereo RMS")
    print(f"verdict         : {verdict}")
    if per10 >= 0.2:
        mins = args.samples * args.interval / 60
        print(f"\nRe-run in 20-30 min. Mechanical creep decays roughly "
              f"logarithmically, so waiting is usually more effective than "
              f"re-tightening (which restarts the settling).")


if __name__ == "__main__":
    main()

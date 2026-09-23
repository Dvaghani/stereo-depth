"""
Measured depth accuracy from checkerboard captures at tape-measured distances.

What this measures -- and what it does not
------------------------------------------
This evaluates the **calibration and the optics**, not a stereo network. It
takes the checkerboard corners directly out of the rectified pair, computes
disparity as `x_left - x_right` per corner, and converts to depth with
`Z = f*B/d`. No matcher is involved, so the result is the accuracy floor the
rig imposes: the best any model could do on this hardware. A learned matcher
adds its own error on top and can only be worse.

That makes it the right number for a *baseline comparison*, because it isolates
the variable under study (baseline length) from the model, which is shared
across all baselines anyway.

Why corners rather than a region of interest
--------------------------------------------
Corners are localised to sub-pixel precision by cornerSubPix, are detected
automatically and identically at every station, and there are 54 of them per
board -- so the median is well determined and the spread is itself informative
(it reveals board tilt and any residual rectification error). A hand-drawn
region would depend on where the operator dragged the box.

Corner ordering
---------------
findChessboardCorners returns corners in a consistent scan order per image, but
the starting corner depends on the board's apparent orientation, so the left and
right lists can come back reversed relative to each other. Rectification gives a
strong check: matched corners must lie on the same row. This script verifies
median |dy| and retries with the right list reversed if it does not, rather than
silently pairing corner 0 with corner 53.

Usage:
    python scripts/eval_depth_accuracy.py \\
        --root outputs/accuracy_280mm \\
        --calib outputs/calibration_280mm/stereo_calib.npz
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

PATTERN = (9, 6)
CHESS_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
REFINE_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
# Matched corners in a rectified pair share a row; more than this means the two
# corner lists were paired in the wrong order.
MAX_ROW_MISMATCH_PX = 3.0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="directory of z<N>m/ station folders")
    p.add_argument("--calib", type=Path, required=True,
                   help="stereo_calib.npz used to rectify these captures")
    p.add_argument("--pattern", type=int, nargs=2, default=list(PATTERN),
                   metavar=("COLS", "ROWS"))
    p.add_argument("--square-mm", type=float, default=25.0,
                   help="checkerboard square size in mm, MEASURED not nominal "
                        "(printer scaling of 1-2%% is common). Used only by the "
                        "PnP cross-check; the stereo depths do not depend on it.")
    p.add_argument("--from-raw", action="store_true", default=True,
                   help="detect corners on left_raw/right_raw and rectify the "
                        "coordinates, instead of using the pre-rectified PNGs. "
                        "Default on: avoids resampling loss, which matters most "
                        "at long range where squares are only a few px.")
    p.add_argument("--from-rectified", dest="from_raw", action="store_false",
                   help="use the pre-rectified left.png/right.png instead")
    p.add_argument("--distances", type=Path, default=None,
                   help="optional text file of '<station> <metres>' lines giving "
                        "the ACTUAL tape readings. Without it the nominal "
                        "distance is parsed from each folder name, which is only "
                        "as good as the rig was positioned.")
    return p.parse_args()


def station_distance(name):
    """Parse 'z0.75m' -> 0.75 (metres)."""
    m = re.search(r"([0-9]*\.?[0-9]+)", name)
    return float(m.group(1)) if m else None


def subpix_window(corners, pattern):
    """Half-window for cornerSubPix, sized from the board's apparent square size.

    cornerSubPix fits a saddle over the window, so the window must lie within a
    single square: too large and it straddles neighbouring corners and drags the
    estimate off. A third of the square spacing keeps it comfortably inside,
    with a floor of 2 px so it stays a usable neighbourhood at long range.
    """
    c = corners.reshape(pattern[1], pattern[0], 2)
    spacing = np.median(np.linalg.norm(np.diff(c, axis=1), axis=2))
    h = int(max(2, min(11, spacing / 3.0)))
    return (h, h)


def corner_disparities(left_png, right_png, pattern, rect=None):
    """Sub-pixel disparity at every checkerboard corner of one stereo pair.

    Two modes:

    rect=None   the images are already rectified; corners are detected and
                differenced directly.

    rect=(K1,D1,R1,P1,K2,D2,R2,P2)
                the images are RAW. Corners are detected on the raw pixels and
                the *coordinates* are then rectified with undistortPoints. This
                is strictly better: remapping an image resamples it, and at long
                range the board is only a few px per square, so that blur costs
                both detection rate and sub-pixel accuracy. Rectifying 54 points
                instead of 2 megapixels avoids the loss entirely, and is immune
                to how the rectification happens to crop.

    Returns ((disparities, median_row_mismatch, left_corners), None)
    or (None, reason).
    """
    gl = cv2.imread(str(left_png), cv2.IMREAD_GRAYSCALE)
    gr = cv2.imread(str(right_png), cv2.IMREAD_GRAYSCALE)
    if gl is None or gr is None:
        return None, "unreadable"

    okl, cl = cv2.findChessboardCorners(gl, pattern, flags=CHESS_FLAGS)
    okr, cr = cv2.findChessboardCorners(gr, pattern, flags=CHESS_FLAGS)
    if not (okl and okr):
        return None, "corners not found in %s" % (
            "both" if not (okl or okr) else ("left" if not okl else "right"))

    # The refinement window must stay INSIDE one square. A fixed 11x11 window is
    # fine at 0.75 m (41 px squares) but spans two whole squares at 6 m (5 px),
    # which corrupts the sub-pixel fit that this whole measurement rests on. So
    # scale it to the observed square spacing.
    win = subpix_window(cl, pattern)
    cl = cv2.cornerSubPix(gl, cl, win, (-1, -1), REFINE_CRIT).reshape(-1, 2)
    cr = cv2.cornerSubPix(gr, cr, win, (-1, -1), REFINE_CRIT).reshape(-1, 2)

    if rect is not None:
        K1, D1, R1, P1, K2, D2, R2, P2 = rect
        cl = cv2.undistortPoints(cl.reshape(-1, 1, 2), K1, D1,
                                 R=R1, P=P1).reshape(-1, 2)
        cr = cv2.undistortPoints(cr.reshape(-1, 1, 2), K2, D2,
                                 R=R2, P=P2).reshape(-1, 2)

    # Rectified correspondences share a row; if they do not, the lists are
    # ordered oppositely (the detector picked a different starting corner).
    best = None
    for candidate in (cr, cr[::-1]):
        dy = float(np.median(np.abs(cl[:, 1] - candidate[:, 1])))
        if best is None or dy < best[0]:
            best = (dy, candidate)
    dy, cr = best
    if dy > MAX_ROW_MISMATCH_PX:
        return None, "corner rows disagree by %.1f px -- pairing unreliable" % dy

    disp = cl[:, 0] - cr[:, 0]
    if np.median(disp) <= 0:
        return None, "non-positive disparity (left/right swapped?)"
    return (disp, dy, cl), None


def pnp_depth(corners, pattern, K, square_mm):
    """Board distance from the LEFT image alone, via solvePnP.

    Uses the rectified intrinsics and the known square size; the baseline and
    the disparity play no part. That independence is the whole point -- it lets
    a stereo/PnP disagreement be attributed to the baseline, and a shared
    disagreement with the tape be attributed to the ground truth.
    """
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:pattern[0], 0:pattern[1]]
                   .T.reshape(-1, 2).astype(np.float32) * square_mm)
    ok, _, tvec = cv2.solvePnP(objp, corners.astype(np.float32), K,
                               np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    return float(np.asarray(tvec).ravel()[2]) if ok else None


def main():
    args = parse_args()
    pattern = tuple(args.pattern)

    calib = np.load(args.calib)
    f = float(calib["focal_px"][0])
    B = float(calib["baseline_mm"][0])
    K_rect = np.ascontiguousarray(calib["P1"][:3, :3])   # rectified intrinsics
    rect = (calib["K1"], calib["D1"], calib["R1"], calib["P1"],
            calib["K2"], calib["D2"], calib["R2"], calib["P2"])
    print("calibration: f=%.2f px, B=%.2f mm  (%s)" % (f, B, args.calib))
    print("Z = f*B/d, so depth is in mm when B is in mm and d in px.\n")

    overrides = {}
    if args.distances and args.distances.exists():
        for line in args.distances.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and not line.strip().startswith("#"):
                overrides[parts[0]] = float(parts[1])
        print("using measured distances from %s\n" % args.distances)

    stations = sorted([d for d in args.root.iterdir() if d.is_dir()],
                      key=lambda d: station_distance(d.name) or 0.0)
    if not stations:
        raise SystemExit("no station folders under %s" % args.root)

    rows = []
    for st in stations:
        z_true_m = overrides.get(st.name, station_distance(st.name))
        if z_true_m is None:
            print("  skipping %s: cannot parse a distance from the name" % st.name)
            continue
        z_true = z_true_m * 1000.0

        all_disp, n_caps, dys, problems, pnps, modes = [], 0, [], [], [], {}
        for cap in sorted(st.glob("capture_*")):
            # Try raw first (no resampling loss), fall back to the rectified
            # pair. Neither dominates: rectification removes lens distortion,
            # which helps findChessboardCorners lock onto a small far-range
            # board, but remapping blurs it, which hurts. Whichever detects is
            # the one that yields a measurement; the disparity is computed in
            # the same rectified frame either way.
            attempts = []
            if args.from_raw and (cap / "left_raw.png").exists():
                attempts.append(("raw", "left_raw.png", "right_raw.png", rect))
            attempts.append(("rect", "left.png", "right.png", None))
            got, err = None, "no images"
            for tag, ln, rn, rc in attempts:
                got, err = corner_disparities(cap / ln, cap / rn, pattern, rect=rc)
                if got is not None:
                    modes[tag] = modes.get(tag, 0) + 1
                    break
            if got is None:
                problems.append("%s: %s" % (cap.name[-15:], err))
                continue
            disp, dy, cl = got
            all_disp.append(disp)
            dys.append(dy)
            n_caps += 1
            zp = pnp_depth(cl, pattern, K_rect, args.square_mm)
            if zp:
                pnps.append(zp)

        if not all_disp:
            print("  %s: no usable captures (%s)" % (st.name, "; ".join(problems)))
            continue

        disp = np.concatenate(all_disp)
        # Median over corners: robust to a corner or two landing on a blurred edge.
        d_med = float(np.median(disp))
        z_meas = f * B / d_med
        # Depth spread implied by the corner-to-corner disparity spread. This is
        # board tilt plus rectification residual, not depth noise, but it bounds
        # how flat the target actually was.
        d_p16, d_p84 = np.percentile(disp, [16, 84])
        z_spread = abs(f * B / d_p84 - f * B / d_p16) / 2.0

        rows.append({
            "name": st.name, "z_true": z_true, "z_meas": z_meas,
            "d_med": d_med, "n_corners": disp.size, "n_caps": n_caps,
            "z_spread": z_spread, "dy": float(np.mean(dys)),
            "z_pnp": float(np.median(pnps)) if pnps else None,
            "modes": modes,
        })
        for p in problems:
            print("  note %s -> %s" % (st.name, p))

    if not rows:
        raise SystemExit("nothing measurable")

    # ── Raw results ───────────────────────────────────────────────────────────
    print("## Measured depth accuracy\n")
    print("| station | tape (m) | captures | corners | median disp (px) | "
          "measured Z (m) | error (mm) | error (%) | plane spread (mm) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        err = r["z_meas"] - r["z_true"]
        print("| %s | %.3f | %d | %d | %.2f | %.3f | %+.0f | %+.2f%% | +/-%.0f |"
              % (r["name"], r["z_true"] / 1000.0, r["n_caps"], r["n_corners"],
                 r["d_med"], r["z_meas"] / 1000.0, err,
                 100.0 * err / r["z_true"], r["z_spread"]))

    # ── Offset fit ────────────────────────────────────────────────────────────
    # The tape is read from the camera body's front face; stereo Z originates at
    # the optical centre, roughly a centimetre further back. That is a constant,
    # so fitting Z_meas = Z_true + c recovers it instead of guessing.
    zt = np.array([r["z_true"] for r in rows])
    zm = np.array([r["z_meas"] for r in rows])
    c = float(np.mean(zm - zt))
    resid = zm - (zt + c)
    rms_raw = float(np.sqrt(np.mean((zm - zt) ** 2)))
    rms_corr = float(np.sqrt(np.mean(resid ** 2)))

    print("\n## After correcting the tape-measure origin\n")
    print("Fitted constant offset c = %+.1f mm — nominally the distance from the "
          "camera body's front face (where the tape was read) back to the "
          "optical centre (where Z originates)." % c)
    # A real origin offset is a constant, so removing it must reduce the spread.
    # If it does not, the errors are not a shared bias and the offset model is
    # simply absorbing noise -- saying so is more useful than reporting a fit.
    if rms_corr < rms_raw * 0.9:
        print("Removing it cuts RMS error from %.1f mm to %.1f mm, so a shared "
              "origin offset does explain much of the raw error."
              % (rms_raw, rms_corr))
    else:
        print("WARNING: removing it changes RMS error only from %.1f mm to "
              "%.1f mm, and the per-station errors do not share a sign. A real "
              "origin offset is a constant bias, so this is NOT one -- the "
              "errors are dominated by something varying per station, most "
              "likely the accuracy of the tape readings and board placement "
              "rather than the stereo. Treat c as unidentified and quote the "
              "raw errors." % (rms_raw, rms_corr))
    print("\n| station | tape+c (m) | measured Z (m) | residual (mm) | residual (%) |")
    print("|---|---:|---:|---:|---:|")
    for r, rr in zip(rows, resid):
        print("| %s | %.3f | %.3f | %+.0f | %+.2f%% |"
              % (r["name"], (r["z_true"] + c) / 1000.0, r["z_meas"] / 1000.0,
                 rr, 100.0 * rr / r["z_true"]))
    print("\nRMS residual after offset correction: %.1f mm over %.2f-%.2f m"
          % (float(np.sqrt(np.mean(resid ** 2))), zt.min() / 1000.0, zt.max() / 1000.0))

    # ── Implied disparity accuracy ────────────────────────────────────────────
    # Invert |dZ| = Z^2*dd/(f*B) to express the residual as a disparity error,
    # which is the scale-free way to compare against the model error figures in
    # docs/jetson_deployment_results.md.
    print("\n## Implied disparity accuracy\n")
    print("| station | residual (mm) | implied disparity error (px) |")
    print("|---|---:|---:|")
    for r, rr in zip(rows, resid):
        dd = abs(rr) * f * B / (r["z_true"] ** 2)
        print("| %s | %+.0f | %.3f |" % (r["name"], rr, dd))
    dd_all = [abs(rr) * f * B / (r["z_true"] ** 2) for r, rr in zip(rows, resid)]
    # Median, not mean: this quantity divides by Z^2, so a fixed placement error
    # at the nearest station explodes into a huge apparent disparity error and
    # would dominate any mean.
    print("\nMedian implied disparity error: %.3f px (mean %.3f px, inflated by "
          "the nearest station -- this metric divides by Z^2, so a few cm of "
          "ground-truth error at close range looks like tens of px)."
          % (float(np.median(dd_all)), float(np.mean(dd_all))))
    print("Compare with the model's measured 1.073 px mean (i4 @ 480x640, "
          "docs/jetson_deployment_results.md section 3).")

    # The corner-to-corner spread is the stereo's own repeatability, measured
    # without reference to the tape at all. Where it is small but the error
    # against tape is large, the tape is the thing in doubt.
    print("\n## Is the error the stereo, or the ground truth?\n")
    print("| station | error vs tape (mm) | internal spread (mm) | verdict |")
    print("|---|---:|---:|---|")
    for r in rows:
        err = abs(r["z_meas"] - r["z_true"])
        v = ("ground truth suspect" if err > 4 * max(r["z_spread"], 1.0)
             else "consistent")
        print("| %s | %+.0f | +/-%.0f | %s |"
              % (r["name"], r["z_meas"] - r["z_true"], r["z_spread"], v))
    print("\nAll %d corners of one board are measured independently, so if the "
          "stereo were wrong they would disagree with each other. Where they "
          "agree tightly (small internal spread) but disagree with the tape, "
          "the reference is the weaker number, not the depth."
          % rows[0]["n_corners"])

    # ── PnP cross-check: is a disagreement the rig, or the tape? ──────────────
    # solvePnP recovers the board distance from the LEFT image alone, using the
    # focal length and the known square size. It never touches the baseline or
    # the disparity. So:
    #   stereo/PnP  differing  -> the baseline is wrong
    #   both agreeing but both differing from tape -> the tape is wrong
    # This is the only way to tell those apart without a second instrument, and
    # it needs no extra capture.
    pnp_rows = [r for r in rows if r.get("z_pnp")]
    if pnp_rows:
        print("\n## PnP cross-check (isolates baseline from ground truth)\n")
        print("| station | tape (m) | PnP Z (m) | stereo Z (m) | stereo-PnP (mm) |")
        print("|---|---:|---:|---:|---:|")
        for r in pnp_rows:
            print("| %s | %.3f | %.3f | %.3f | %+.0f |"
                  % (r["name"], r["z_true"] / 1000.0, r["z_pnp"] / 1000.0,
                     r["z_meas"] / 1000.0, r["z_meas"] - r["z_pnp"]))
        zt_ = np.array([r["z_true"] for r in pnp_rows])
        zp_ = np.array([r["z_pnp"] for r in pnp_rows])
        zs_ = np.array([r["z_meas"] for r in pnp_rows])
        fit = lambda x, y: float(np.sum(x * y) / np.sum(x * x))
        k_pnp, k_ster, k_base = fit(zt_, zp_), fit(zt_, zs_), fit(zp_, zs_)
        print("\nPnP    / tape = %.5f (%+.2f%%)   no baseline involved"
              % (k_pnp, (k_pnp - 1) * 100))
        print("stereo / tape = %.5f (%+.2f%%)   baseline involved"
              % (k_ster, (k_ster - 1) * 100))
        print("stereo / PnP  = %.5f (%+.2f%%)   <-- the baseline error alone"
              % (k_base, (k_base - 1) * 100))
        print("   implied true baseline %.2f mm vs calibrated %.2f mm"
              % (B / k_base, B))
        if abs(k_base - 1) < 0.01 and abs(k_pnp - 1) > 0.004:
            print("\nVERDICT: stereo and PnP agree with each other but both "
                  "disagree with the tape. Two independent methods do not share "
                  "an error mode, so the ground truth (board placement and tape "
                  "readings) is the weaker measurement -- not the rig.")

    print("\nNOTE: this is the geometric floor -- checkerboard corners are the "
          "easiest possible matching target. Real scenes with texture-poor or "
          "occluded regions will do worse. Quote it as the rig's best case.")


if __name__ == "__main__":
    main()

"""
Geometric range and precision statistics for every calibrated baseline.

Why this exists
---------------
The thesis compares three physical baselines (110/160/280 mm). Most of what
distinguishes them is not measured, it is *derived*: given a calibration, the
usable range and the depth precision at every distance follow from geometry
alone. Deriving them here -- from the real stereo_calib.npz files rather than
from nominal numbers -- keeps the table honest and regenerable. The recovered
baseline is never exactly the nominal one (280.52 vs 280), and the rectified
focal differs per calibration session, so nominal figures would be wrong in the
third significant digit.

The geometry
------------
    Z = f * B / d                       depth from disparity
    dZ/dd = -f * B / d^2                differentiate
    |dZ| = (Z^2 / (f * B)) * |dd|       substitute d = f*B/Z

That last line is the whole story: **depth error grows with the square of
distance and shrinks linearly with f*B**. Doubling the baseline halves the
error at every range. It is also why "what is the maximum range" has no single
answer -- error degrades smoothly, so a range limit only exists once you fix an
acceptable error. This script reports the limit at several thresholds instead of
inventing one.

Choosing --disp-err
-------------------
Depth precision is only as good as disparity precision, so the assumed matching
error dominates every number here. The default 1.073 px is not a textbook value;
it is the measured mean error of the recommended on-device configuration
(`i4 @ 480x640`) from docs/jetson_deployment_results.md section 3. Two caveats
worth stating in the thesis:

  - That error is measured against a 32-iteration full-resolution reference,
    not against ground truth, so true error against the world is somewhat
    larger. These figures are therefore a floor, not a guarantee.
  - It is a *mean*. The p95 for the same configuration is 5.22 px, roughly 5x
    worse. Pass --disp-err 5.22 to see the pessimistic tail.

Near-limit caveat
-----------------
Two different things can set the near limit, and which one binds depends on the
model:

  - AANet / StereoUNet search a bounded window, so Z_min = f*B/max_disp. This
    is architectural: max_disp is a cost-volume dimension baked into the
    trained checkpoint, not a runtime flag.
  - RAFT-Stereo has no such bound (correlation pyramid, not a fixed window), so
    its near limit is field-of-view overlap, which is far closer.

Both are reported. Use the one matching the model being described.

Usage:
    python scripts/baseline_stats.py
    python scripts/baseline_stats.py --disp-err 5.22 --latex
"""

import argparse
import math
from pathlib import Path

import numpy as np

# Measured mean disparity error of the recommended on-device configuration
# (i4 @ 480x640) -- docs/jetson_deployment_results.md section 3.
DEFAULT_DISP_ERR = 1.073
DEFAULT_MAX_DISP = 192
DEFAULT_DISTANCES = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0)
# Relative-depth-error thresholds used to define "maximum usable range".
ERROR_THRESHOLDS = (0.01, 0.02, 0.05, 0.10)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs", type=Path, default=Path("outputs"),
                   help="directory holding calibration_<N>mm/ folders")
    p.add_argument("--width", type=int, default=640,
                   help="inference input width in px (default 640, the "
                        "deployment engine size). Disparity scales with width, "
                        "so this changes every number below.")
    p.add_argument("--disp-err", type=float, default=DEFAULT_DISP_ERR,
                   help="assumed disparity matching error in px at the "
                        "inference width (default %.3f, the measured mean of "
                        "i4 @ 480x640)" % DEFAULT_DISP_ERR)
    p.add_argument("--max-disp", type=int, default=DEFAULT_MAX_DISP,
                   help="cost-volume disparity bins for AANet/StereoUNet "
                        "(default 192). Sets the near limit for those models; "
                        "RAFT-Stereo is unbounded and ignores this.")
    p.add_argument("--distances", type=float, nargs="+", default=list(DEFAULT_DISTANCES),
                   metavar="M", help="distances in metres for the precision table")
    p.add_argument("--latex", action="store_true",
                   help="also emit LaTeX tabular bodies for pasting into the thesis")
    return p.parse_args()


def load_rigs(outputs, width):
    """Read every calibration_<N>mm/stereo_calib.npz, scaled to `width`.

    Uses the checkerboard-derived stereo_calib.npz rather than any
    stereo_calib_refreshed*.npz sitting beside it: the refreshed files come from
    quick_recalib.py, which re-estimates rotation from natural-scene features
    and keeps the original baseline, so it cannot improve the numbers used here.
    """
    rigs = []
    for d in sorted(outputs.glob("calibration_*mm")):
        npz = d / "stereo_calib.npz"
        if not npz.exists():
            print("  note: %s has no stereo_calib.npz, skipping" % d.name)
            continue
        c = np.load(npz)
        w_full = int(c["image_size"][0])
        f_full = float(c["focal_px"][0])
        scale = width / float(w_full)
        rigs.append({
            "name": d.name.replace("calibration_", ""),
            "nominal": int("".join(ch for ch in d.name if ch.isdigit())),
            "B": float(c["baseline_mm"][0]),
            "f_full": f_full,
            "w_full": w_full,
            "f": f_full * scale,          # focal at the inference width
            "rms": float(c["rms_stereo"][0]),
            "n_pairs": None,
        })
    rigs.sort(key=lambda r: r["B"])
    return rigs


def hfov_deg(f, width):
    """Horizontal field of view from the rectified focal length."""
    return 2.0 * math.degrees(math.atan(width / (2.0 * f)))


def main():
    args = parse_args()
    rigs = load_rigs(args.outputs, args.width)
    if not rigs:
        raise SystemExit("no calibration_*mm/stereo_calib.npz found under %s"
                         % args.outputs)

    dd = args.disp_err
    W = args.width

    print("=" * 78)
    print("BASELINE COMPARISON  --  inference width %d px, assumed disparity "
          "error %.3f px" % (W, dd))
    print("=" * 78)

    # ── 1. Rig summary ────────────────────────────────────────────────────────
    print("\n## 1. Calibrated rig parameters\n")
    print("| baseline | recovered B (mm) | focal @%d (px) | focal @%d (px) | "
          "HFOV | stereo RMS |" % (rigs[0]["w_full"], W))
    print("|---|---:|---:|---:|---:|---:|")
    for r in rigs:
        print("| %s | %.2f | %.1f | %.1f | %.1f deg | %.3f px |"
              % (r["name"], r["B"], r["f_full"], r["f"],
                 hfov_deg(r["f"], W), r["rms"]))
    print("\nRecovered B is what stereoCalibrate solved for, not the tape "
          "measurement -- it is the value used in every calculation below.")

    # ── 2. Range limits ───────────────────────────────────────────────────────
    print("\n## 2. Range limits\n")
    print("Near limit has two definitions depending on the model (see module "
          "docstring). Far limit is quoted at several relative-error "
          "thresholds, because depth error degrades smoothly with distance.\n")
    hdr = ("| baseline | f*B | near: AANet (d=%d) | near: RAFT (FOV overlap) | "
           % args.max_disp)
    hdr += " | ".join("far: <=%d%% err" % int(t * 100) for t in ERROR_THRESHOLDS) + " |"
    print(hdr)
    print("|---|---:|---:|---:|" + "---:|" * len(ERROR_THRESHOLDS))
    for r in rigs:
        fB = r["f"] * r["B"]
        z_near_disp = fB / args.max_disp / 1000.0
        # Frusta start to overlap once half-width at Z exceeds half the baseline.
        z_near_fov = r["B"] / (2.0 * (W / (2.0 * r["f"]))) / 1000.0
        # |dZ|/Z = Z*dd/(f*B)  ->  Z at threshold t is t*f*B/dd
        fars = [t * fB / dd / 1000.0 for t in ERROR_THRESHOLDS]
        print("| %s | %.0f | %.2f m | %.2f m | %s |"
              % (r["name"], fB, z_near_disp, z_near_fov,
                 " | ".join("%.1f m" % z for z in fars)))

    # ── 3. Depth precision vs distance ────────────────────────────────────────
    print("\n## 3. Depth precision  (+/- mm at %.3f px disparity error)\n" % dd)
    print("| distance | " + " | ".join(r["name"] for r in rigs) + " |")
    print("|---|" + "---:|" * len(rigs))
    for z in args.distances:
        cells = []
        for r in rigs:
            zmm = z * 1000.0
            err = zmm * zmm * dd / (r["f"] * r["B"])
            cells.append("%.0f mm (%.1f%%)" % (err, 100.0 * err / zmm))
        print("| %.1f m | %s |" % (z, " | ".join(cells)))
    ref, best = rigs[0], rigs[-1]
    print("\nDepth error scales as Z^2/(f*B): quadratic in distance, inverse in "
          "f*B. %s vs %s is a %.2fx improvement at every range."
          % (best["name"], ref["name"],
             (best["f"] * best["B"]) / (ref["f"] * ref["B"])))

    # ── 4. Disparity vs distance ──────────────────────────────────────────────
    print("\n## 4. Disparity at the inference width (px @ %d wide)\n" % W)
    print("| distance | " + " | ".join(r["name"] for r in rigs) + " |")
    print("|---|" + "---:|" * len(rigs))
    for z in args.distances:
        cells = []
        for r in rigs:
            d = r["f"] * r["B"] / (z * 1000.0)
            flag = " **>%d**" % args.max_disp if d > args.max_disp else ""
            cells.append("%.1f%s" % (d, flag))
        print("| %.1f m | %s |" % (z, " | ".join(cells)))
    print("\nBold marks disparities beyond the %d-bin cost volume: AANet/"
          "StereoUNet cannot represent these and will return a confident wrong "
          "value rather than flagging failure. RAFT-Stereo is unaffected."
          % args.max_disp)

    if args.latex:
        print("\n" + "=" * 78)
        print("LaTeX tabular bodies")
        print("=" * 78)
        print("\n%% Range limits")
        print(r"\begin{tabular}{lrrrr}")
        print(r"\toprule")
        print(r"Baseline & $fB$ & Near (AANet) & Near (RAFT) & Far ($\leq$10\,\%) \\")
        print(r"\midrule")
        for r in rigs:
            fB = r["f"] * r["B"]
            print(r"%s & %.0f & %.2f\,m & %.2f\,m & %.1f\,m \\"
                  % (r["name"], fB, fB / args.max_disp / 1000.0,
                     r["B"] / (2.0 * (W / (2.0 * r["f"]))) / 1000.0,
                     0.10 * fB / dd / 1000.0))
        print(r"\bottomrule"); print(r"\end{tabular}")

        print("\n%% Depth precision")
        print(r"\begin{tabular}{l" + "r" * len(rigs) + "}")
        print(r"\toprule")
        print(r"Distance & " + " & ".join(r["name"] for r in rigs) + r" \\")
        print(r"\midrule")
        for z in args.distances:
            cells = ["%.0f" % ((z * 1000.0) ** 2 * dd / (r["f"] * r["B"])) for r in rigs]
            print(r"%.1f\,m & " % z + " & ".join(cells) + r" \\")
        print(r"\bottomrule"); print(r"\end{tabular}")

    print("\nNOTE: every figure above is geometric -- derived from calibration, "
          "not measured against ground truth. They bound what the optics can "
          "do; they do not capture matching failures on textureless or "
          "occluded regions. Pair them with a measured error table from real "
          "captures at tape-measured distances.")


if __name__ == "__main__":
    main()

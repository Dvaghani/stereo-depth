"""
Build a portable INT8 calibration set of stereo pairs for RAFT-Stereo.

Why this exists
---------------
Like the YOLO calibration set, INT8 for RAFT needs a fixed set of real stereo
pairs at engine build time — TensorRT records activation ranges over them and
bakes the scales in. Random data gives poor scales, and for a *regression*
network like RAFT the consequences are worse than for a classifier: there is no
argmax to hide small errors, so a bad scale shows up directly as disparity
error.

What "representative" means here
--------------------------------
For YOLO it meant covering all 11 classes. For RAFT the analogous axis is the
**disparity distribution**, because that is what drives correlation-volume
magnitudes. A calibration set whose disparities sit in a different range than
the deployment rig will set the ranges in the wrong place. This script therefore
reports the disparity histogram of the set it builds, so the match (or mismatch)
against the rig's operating range is visible rather than assumed.

Reference: on the current 280 mm Brio rig the working range is roughly 23-192 px
at 640x480, derived from the measured calibration (see --rig-disp-range below).
The earlier 160 mm rig ran at 30-122 px (docs/jetson_deployment_results.md
section 2).

Sources
-------
  kitti       200 pairs, ground-truth disparity. The standard benchmark, but
              1242x375 (3.3:1) squashed into a 1.33:1 engine input distorts
              geometry heavily -- good for GT accuracy scoring, less ideal as a
              calibration source.
  middlebury  23 scenes, ground-truth disparity, ~1.5:1 aspect -- much closer
              to the engine input, so generally the better calibration source
              of the two.
  rig         own captures (outputs/capture_*/left.png + right.png). No ground
              truth, but the exact deployment distribution. Best calibration
              source when enough pairs exist; capture more with capture_stereo.py.

Mixing sources is supported and usually right: rig pairs for domain match,
public data for diversity.

Usage:
    python scripts/build_raft_calib_set.py --sources middlebury,rig \\
        --out outputs/int8_calib_raft --count 200

Then on the Orin, point a TensorRT IInt8EntropyCalibrator2 at the manifest.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import read_pfm, read_kitti_disparity_png  # noqa: E402

KITTI_REL = "data_scene_flow/training"
MIDDLEBURY_REL = "middlebury2014"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default="middlebury,kitti",
                   help="comma-separated: middlebury, kitti, rig")
    p.add_argument("--data-root", type=Path,
                   default=Path("/run/media/dvaghani/Expansion/Dataset"),
                   help="root holding data_scene_flow/ and middlebury2014/")
    p.add_argument("--rig-glob", default="outputs/capture_*",
                   help="glob for own-rig capture dirs (expects left.png/right.png)")
    p.add_argument("--min-sharpness", type=float, default=150.0,
                   help="drop rig frames below this Laplacian-variance "
                        "sharpness (motion blur from timer-driven bursts -- "
                        "see collect_rig). 0 disables filtering.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=200,
                   help="target number of pairs (default 200)")
    p.add_argument("--width", type=int, default=640,
                   help="engine input width (default 640)")
    p.add_argument("--height", type=int, default=480,
                   help="engine input height (default 480)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gt", action="store_true",
                   help="skip saving ground-truth disparity (smaller output)")
    p.add_argument("--rig-disp-range", default="23,192",
                   help="the rig's working disparity range in px at the engine "
                        "input size, as MIN,MAX -- used only to report overlap. "
                        "The 23-192 default is the 280 mm rig (outputs/"
                        "calibration_280mm): rectified focal 1232.6 px at 1920 "
                        "wide is 410.9 px at 640 wide, and with baseline "
                        "280.52 mm, d = 410.9 * 280.52 / Z gives 192 px at "
                        "0.6 m and 23 px at 5 m. Earlier rigs, same formula: "
                        "160 mm gave 30-122 (docs/jetson_deployment_results.md "
                        "section 2), 110 mm roughly 21-84. Set this to match "
                        "the baseline and working distances actually in use -- "
                        "note RAFT itself has no max-disparity bound (it uses a "
                        "correlation pyramid, not a fixed search window), so the "
                        "near end here reflects the deployment distance chosen, "
                        "not an architectural limit.")
    return p.parse_args()


def collect_kitti(root, want_gt):
    """KITTI 2015: image_2=left, image_3=right, disp_occ_0=GT. Only the *_10
    frames have ground truth, so the *_11 frames are skipped."""
    base = root / KITTI_REL
    if not (base / "image_2").is_dir():
        print("  kitti: not found at %s, skipping" % base)
        return []
    out = []
    for left in sorted((base / "image_2").glob("*_10.png")):
        right = base / "image_3" / left.name
        gt = base / "disp_occ_0" / left.name
        if right.exists():
            out.append({"left": left, "right": right,
                        "gt": gt if (want_gt and gt.exists()) else None,
                        "gt_kind": "kitti_png", "source": "kitti"})
    return out


def collect_middlebury(root, want_gt):
    """Middlebury 2014: im0=left, im1=right, disp0.pfm=GT. Uses the -perfect
    variants; -imperfect duplicates the same scene with worse rectification."""
    base = root / MIDDLEBURY_REL
    if not base.is_dir():
        print("  middlebury: not found at %s, skipping" % base)
        return []
    out = []
    for scene in sorted(base.glob("*-perfect")):
        left, right = scene / "im0.png", scene / "im1.png"
        gt = scene / "disp0.pfm"
        if left.exists() and right.exists():
            out.append({"left": left, "right": right,
                        "gt": gt if (want_gt and gt.exists()) else None,
                        "gt_kind": "pfm", "source": "middlebury"})
    return out


def sharpness(path):
    """Variance of Laplacian -- a standard, cheap focus/motion-blur proxy.
    Lower means blurrier (either defocus or motion blur during the exposure)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def collect_rig(pattern, min_sharpness=0.0):
    """Own captures: rectified left.png/right.png, no ground truth.

    Timer-driven bursts (capture_calib_burst.py) can catch the rig mid-motion
    between positions -- that shows up as directional motion blur, not
    defocus, and it is common enough (measured ~44% of one 200-pair burst)
    that filtering it here by default is worth the loss of some frames. A
    blurry frame does not help INT8 calibration; it just adds noise to the
    activation-range estimate.
    """
    out = []
    dropped = 0
    for d in sorted(Path(".").glob(pattern)):
        left, right = d / "left.png", d / "right.png"
        if not (left.exists() and right.exists()):
            continue
        if min_sharpness > 0 and sharpness(left) < min_sharpness:
            dropped += 1
            continue
        out.append({"left": left, "right": right, "gt": None,
                    "gt_kind": None, "source": "rig"})
    if min_sharpness > 0:
        print("  rig: dropped %d/%d frames below sharpness %.0f (motion blur)"
              % (dropped, dropped + len(out), min_sharpness))
    return out


def load_gt(entry):
    if entry["gt"] is None:
        return None
    if entry["gt_kind"] == "kitti_png":
        return read_kitti_disparity_png(entry["gt"])
    disp, _ = read_pfm(entry["gt"])
    disp = np.asarray(disp, dtype=np.float32)
    disp[~np.isfinite(disp)] = 0.0     # Middlebury marks unknown as inf
    return disp


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    want_gt = not args.no_gt

    wanted = [s.strip() for s in args.sources.split(",") if s.strip()]
    entries = []
    print("collecting sources: %s" % ", ".join(wanted))
    for src in wanted:
        if src == "kitti":
            got = collect_kitti(args.data_root, want_gt)
        elif src == "middlebury":
            got = collect_middlebury(args.data_root, want_gt)
        elif src == "rig":
            got = collect_rig(args.rig_glob, args.min_sharpness)
        else:
            raise SystemExit("unknown source: %s" % src)
        print("  %-12s %d pairs" % (src, len(got)))
        entries.extend(got)

    if not entries:
        raise SystemExit("no stereo pairs found -- check --data-root/--rig-glob")

    idx = rng.permutation(len(entries))[: min(args.count, len(entries))]
    picked = [entries[i] for i in idx]
    if len(picked) < args.count:
        print("NOTE: only %d pairs available, using all of them" % len(picked))

    out_l = args.out / "left"
    out_r = args.out / "right"
    out_d = args.out / "disp_gt"
    for d in (out_l, out_r):
        d.mkdir(parents=True, exist_ok=True)
    if want_gt:
        out_d.mkdir(parents=True, exist_ok=True)

    size = (args.width, args.height)
    manifest, disp_samples, n_gt = [], [], 0

    for i, e in enumerate(picked):
        li = cv2.imread(str(e["left"]), cv2.IMREAD_COLOR)
        ri = cv2.imread(str(e["right"]), cv2.IMREAD_COLOR)
        if li is None or ri is None:
            print("  skipping unreadable pair: %s" % e["left"])
            continue
        oh, ow = li.shape[:2]
        # disparity is a horizontal quantity, so it rescales with width only
        sx = args.width / float(ow)

        name = "%04d.png" % i
        cv2.imwrite(str(out_l / name), cv2.resize(li, size, interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(out_r / name), cv2.resize(ri, size, interpolation=cv2.INTER_AREA))

        rec = {"id": i, "source": e["source"], "orig_size": [ow, oh],
               "scale_x": sx, "left": "left/" + name, "right": "right/" + name}

        gt = load_gt(e)
        if gt is not None:
            # resize the map, then scale the values: both are needed
            gt_r = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST) * sx
            np.save(str(out_d / ("%04d.npy" % i)), gt_r.astype(np.float32))
            rec["disp_gt"] = "disp_gt/%04d.npy" % i
            valid = gt_r[np.isfinite(gt_r) & (gt_r > 0)]
            if valid.size:
                disp_samples.append(rng.choice(valid, size=min(valid.size, 20000),
                                               replace=False))
            n_gt += 1
        manifest.append(rec)

    (args.out / "manifest.json").write_text(json.dumps({
        "input_size": [args.width, args.height],
        "note": "left/right are resized to input_size; disp_gt is resized AND "
                "value-scaled by scale_x, so it is in pixels at input_size.",
        "pairs": manifest,
    }, indent=2))

    mb = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print("\nwrote %d pairs (%d with ground truth) to %s (%.1f MB)"
          % (len(manifest), n_gt, args.out, mb))

    by_src = {}
    for r in manifest:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print("  by source: %s" % ", ".join("%s=%d" % kv for kv in sorted(by_src.items())))

    if disp_samples:
        all_disp = np.concatenate(disp_samples)
        pct = np.percentile(all_disp, [1, 25, 50, 75, 99])
        print("\ndisparity distribution at %dx%d (px):" % (args.width, args.height))
        print("  p1=%.1f  p25=%.1f  median=%.1f  p75=%.1f  p99=%.1f"
              % tuple(pct))
        lo, hi = (float(v) for v in args.rig_disp_range.split(","))
        print("  rig operating range for comparison: %.0f-%.0f px" % (lo, hi))
        overlap = np.mean((all_disp >= lo) & (all_disp <= hi)) * 100
        print("  %.0f%% of calibration disparities fall inside the rig's range"
              % overlap)
        if overlap < 50:
            print("  WARNING: weak overlap with the deployment range. INT8 scales "
                  "will be tuned for disparities the rig does not produce. "
                  "Prefer --sources with more 'rig' pairs.")
    print("\nnext: build the INT8 engine on the Orin with a calibrator reading "
          "%s/manifest.json" % args.out)


if __name__ == "__main__":
    main()

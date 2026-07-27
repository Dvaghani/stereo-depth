"""Cable sources -> unified taxonomy, in YOLO SEGMENTATION format.

Cable is the one class trained with real polygon masks rather than boxes, so
this deliberately preserves polygons instead of reducing them to bounding
boxes: a bbox around a thin diagonal cable is almost entirely background.

Two sources:
  own     labelme JSONs from construction_yolo/prelabel_cable_hsv.py, corrected
          by hand — the in-domain orange construction cable
  roboflow  giuseppe-x8ycy/cable-segmentation, already YOLO-seg with nc=1;
          transmission lines, so out of domain, but supplies bulk "thin line"
          signal (measured to transfer only for dark-cable-against-sky)

The own/val split is by CAPTURE SESSION, not random: the images come from two
outings at different locations, and splitting randomly would put near-duplicate
views of the same scene on both sides, making val measure memorisation.

Usage:
    python construction_yolo/datasets/converters/cable_converter.py \
        --own /run/media/dvaghani/Expansion/Yolo/cable_prelabel \
        --roboflow /run/media/dvaghani/Expansion/Yolo/sources_raw/cable_segmentation
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.taxonomy import CLASS_ID  # noqa: E402

OUT_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/cable_seg")
CABLE_ID = CLASS_ID["cable"]

# Images whose filename contains any of these go to val. Defaults to the
# 2026-07-22 outing, holding out a whole session rather than random frames.
DEFAULT_VAL_PATTERNS = ["2026-07-22"]


def polygon_to_yolo_seg(points, width, height, cls_id):
    """YOLO-seg line: '<cls> x1 y1 x2 y2 ...' normalised to 0-1."""
    coords = []
    for x, y in points:
        nx = min(max(x / width, 0.0), 1.0)
        ny = min(max(y / height, 0.0), 1.0)
        coords.extend((nx, ny))
    if len(coords) < 6:          # need >=3 points
        return None
    return "%d %s" % (cls_id, " ".join("%.6f" % c for c in coords))


def convert_own(src: Path, val_patterns, counts: Counter, oversample=1):
    n = {"train": 0, "val": 0}
    for jf in sorted(src.glob("*.json")):
        doc = json.loads(jf.read_text())
        shapes = [s for s in doc.get("shapes", [])
                  if s.get("label", "").lower() == "cable"
                  and s.get("shape_type") == "polygon"]
        if not shapes:
            continue

        img_name = doc.get("imagePath") or (jf.stem + ".jpeg")
        img_path = src / img_name
        if not img_path.exists():
            print("  missing image for %s" % jf.name)
            continue

        W, H = doc["imageWidth"], doc["imageHeight"]
        lines = []
        for s in shapes:
            line = polygon_to_yolo_seg(s["points"], W, H, CABLE_ID)
            if line:
                lines.append(line)
                counts["own"] += 1
        if not lines:
            continue

        split = "val" if any(p in jf.stem for p in val_patterns) else "train"
        base = "owncable_" + jf.stem.replace(" ", "_")
        # Oversample train only — repeating val images would just inflate the
        # metric. Copies are identical on disk, but YOLO's per-epoch mosaic/HSV/
        # flip augmentation means the model does not see the same pixels twice.
        reps = oversample if split == "train" else 1
        for r in range(reps):
            stem = base if r == 0 else "%s_rep%02d" % (base, r)
            shutil.copy(img_path, OUT_ROOT / "images" / split / (stem + img_path.suffix))
            (OUT_ROOT / "labels" / split / (stem + ".txt")).write_text("\n".join(lines))
            n[split] += 1
    return n


def convert_roboflow(src: Path, counts: Counter, limit=None):
    """Already YOLO-seg with a single class 0; remap that id to ours.

    limit caps how many images are taken. Left uncapped this source outnumbers
    the in-domain data ~100:1 and the model simply learns transmission lines,
    which measured near-zero transfer to orange construction cable."""
    n = {"train": 0, "val": 0}
    taken = 0
    for split_in, split_out in (("train", "train"), ("valid", "train"), ("test", "train")):
        lbl_dir, img_dir = src / split_in / "labels", src / split_in / "images"
        if not lbl_dir.exists():
            continue
        for lbl in sorted(lbl_dir.glob("*.txt")):
            if limit is not None and taken >= limit:
                return n
            taken += 1
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = img_dir / (lbl.stem + ext)
                if cand.exists():
                    img = cand
                    break
            if img is None:
                continue
            lines = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 7:      # class + >=3 xy pairs
                    continue
                lines.append("%d %s" % (CABLE_ID, " ".join(parts[1:])))
                counts["roboflow"] += 1
            if not lines:
                continue
            stem = "rfcable_%s_%s" % (split_in, lbl.stem)
            shutil.copy(img, OUT_ROOT / "images" / split_out / (stem + img.suffix))
            (OUT_ROOT / "labels" / split_out / (stem + ".txt")).write_text("\n".join(lines))
            n[split_out] += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--own", required=True, help="corrected labelme dir")
    p.add_argument("--roboflow", default=None, help="Cable-segmentation-3 root")
    p.add_argument("--oversample", type=int, default=20,
                   help="repeat each in-domain TRAIN image N times so the 9 real "
                        "images are not drowned by the out-of-domain bulk")
    p.add_argument("--roboflow-limit", type=int, default=300,
                   help="cap out-of-domain images (0 = uncapped)")
    p.add_argument("--val-pattern", action="append", default=None,
                   help="filename substring routing an image to val "
                        "(default: %s)" % DEFAULT_VAL_PATTERNS)
    args = p.parse_args()

    for split in ("train", "val"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    val_patterns = args.val_pattern or DEFAULT_VAL_PATTERNS
    counts = Counter()

    print("own (in-domain, held out by session: %s)" % ", ".join(val_patterns))
    own = convert_own(Path(args.own), val_patterns, counts, args.oversample)
    print("  train %d images (%dx oversampled), val %d images, %d unique polygons"
          % (own["train"], args.oversample, own["val"], counts["own"]))

    if args.roboflow:
        print("roboflow cable-segmentation (out-of-domain bulk, train only)")
        rf = convert_roboflow(Path(args.roboflow), counts,
                              args.roboflow_limit or None)
        print("  train %d images, %d polygons" % (rf["train"], counts["roboflow"]))

    print("\n-> %s" % OUT_ROOT)
    for split in ("train", "val"):
        n = len(list((OUT_ROOT / "labels" / split).glob("*.txt")))
        print("  %-5s %d images" % (split, n))
    if counts["own"]:
        # Report the EFFECTIVE balance the model actually trains on, i.e. after
        # oversampling — the unique-polygon ratio understates it badly.
        own_eff = counts["own"] * args.oversample
        share = 100.0 * own_eff / (own_eff + counts["roboflow"])
        print("\n%d unique in-domain polygons, %d after %dx oversampling, vs %d "
              "out-of-domain\n-> in-domain is %.1f%% of what the model sees."
              % (counts["own"], own_eff, args.oversample, counts["roboflow"], share))
        print("The remainder is transmission-line imagery, measured to transfer "
              "poorly to\norange construction cable. Expect cable to be the "
              "weakest class, and note that\noversampling reweights the loss but "
              "adds no new information — it cannot\ncompensate for having only "
              "%d distinct training images." % (own["train"] // max(args.oversample, 1)))


if __name__ == "__main__":
    main()

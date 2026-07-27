"""Merge every source into one 11-class YOLO **segmentation** dataset.

Run 3 trains a YOLO11-seg model so that `cable` can carry real masks — a box
around a thin diagonal cable is almost all background, which matters because
the class exists for collision avoidance where recall on thin obstacles is the
whole point.

The other ten classes keep their existing box labels: a box is emitted as its
four corners, a degenerate polygon. That is a pure format conversion, so none
of the ~37k previously labelled images need re-annotating, and downstream code
can still read `.boxes` from the Ultralytics results exactly as before.

Sources:
  own              our DJI frames (boxes)      train + val, split unchanged
  coco/roboflow/
  openimages/mocs  public boxes                train only
  barricade_barrier  boxes                     train only
  cable_seg        REAL polygons               train + val (own val session)

Every image is verified to open and every label to parse before it is copied.

Usage:
    python construction_yolo/datasets/merge_unified_seg.py \
        --out /run/media/dvaghani/Expansion/Yolo/unified_seg_dataset
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.taxonomy import CLASS_NAMES  # noqa: E402

OWN_DATASET = Path("/run/media/dvaghani/Expansion/Yolo/construction_dataset")

# box-labelled sources -> converted to rectangle polygons
BOX_SOURCES = [
    Path("/run/media/dvaghani/Expansion/Yolo/sources/coco_yolo"),
    Path("/run/media/dvaghani/Expansion/Yolo/sources/roboflow_yolo"),
    Path("/run/media/dvaghani/Expansion/Yolo/sources/openimages_yolo"),
    Path("/run/media/dvaghani/Expansion/Yolo/sources/mocs_yolo"),
    Path("/run/media/dvaghani/Expansion/Yolo/sources/barricade_barrier_yolo"),
]

# already-polygon sources, with their own train/val split to respect
SEG_SOURCES = [
    Path("/run/media/dvaghani/Expansion/Yolo/sources/cable_seg"),
]

# suspended-load and barrier appear in no in-domain val image, so without this
# their mAP reads 0.00 however well they are learned. Both sources kept their
# original split in the filename, so route a bounded slice of it to val.
# Bounded deliberately: MOCS alone has 4000 val images and would swamp the 450
# in-domain ones, turning overall mAP into a measure of MOCS performance.
VAL_TOPUP = [
    # (source, filename prefix, class that needs val coverage, max images)
    (Path("/run/media/dvaghani/Expansion/Yolo/sources/mocs_yolo"),
     "mocs_val_", "suspended-load", 150),
    (Path("/run/media/dvaghani/Expansion/Yolo/sources/barricade_barrier_yolo"),
     "barricade_barrier_valid_", "barrier", None),
]


def box_to_polygon(cls_id: int, cx: float, cy: float, w: float, h: float) -> str:
    """A box as its four corners — a degenerate polygon, in YOLO-seg format."""
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    pts = [x0, y0, x1, y0, x1, y1, x0, y1]
    pts = [min(max(v, 0.0), 1.0) for v in pts]
    return "%d %s" % (cls_id, " ".join("%.6f" % v for v in pts))


def parse_label(lbl_path: Path, counts: Counter, already_seg: bool):
    """Return validated YOLO-seg lines, or None if nothing usable."""
    lines = []
    for raw in lbl_path.read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cls_id = int(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if not (0 <= cls_id < len(CLASS_NAMES)):
            continue
        if not all(0.0 <= v <= 1.0 for v in vals):
            continue

        if already_seg or len(vals) > 4:
            if len(vals) < 6 or len(vals) % 2:    # need >=3 xy pairs
                continue
            lines.append("%d %s" % (cls_id, " ".join("%.6f" % v for v in vals)))
        else:
            if len(vals) != 4:
                continue
            lines.append(box_to_polygon(cls_id, *vals))
        counts[CLASS_NAMES[cls_id]] += 1
    return lines or None


def copy_split(img_dir: Path, lbl_dir: Path, out_img: Path, out_lbl: Path,
               counts: Counter, already_seg: bool):
    from PIL import Image
    n_ok = n_bad = 0
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = img_dir / (lbl.stem + ext)
            if cand.exists():
                img = cand
                break
        if img is None:
            n_bad += 1
            continue
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception:
            n_bad += 1
            continue
        lines = parse_label(lbl, counts, already_seg)
        if lines is None:
            n_bad += 1
            continue
        shutil.copy(img, out_img / img.name)
        (out_lbl / (lbl.stem + ".txt")).write_text("\n".join(lines))
        n_ok += 1
    return n_ok, n_bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": Counter(), "val": Counter()}
    total_bad = 0

    # our own frames keep their existing split, so run-3 val stays comparable
    for split in ("train", "val"):
        ok, bad = copy_split(OWN_DATASET / "images" / split,
                             OWN_DATASET / "labels" / split,
                             out / "images" / split, out / "labels" / split,
                             counts[split], already_seg=False)
        total_bad += bad
        print("own/%-5s %6d images  (%d skipped)" % (split, ok, bad))

    for src in BOX_SOURCES:
        if not src.exists():
            print("skip (missing): %s" % src.name)
            continue
        ok, bad = copy_split(src / "images", src / "labels",
                             out / "images" / "train", out / "labels" / "train",
                             counts["train"], already_seg=False)
        total_bad += bad
        print("%-26s %6d images -> train  (%d skipped)" % (src.name, ok, bad))

    for src in SEG_SOURCES:
        if not src.exists():
            print("skip (missing): %s" % src.name)
            continue
        for split in ("train", "val"):
            sub_img, sub_lbl = src / "images" / split, src / "labels" / split
            if not sub_lbl.exists():
                continue
            ok, bad = copy_split(sub_img, sub_lbl,
                                 out / "images" / split, out / "labels" / split,
                                 counts[split], already_seg=True)
            total_bad += bad
            print("%-26s %6d images -> %-5s (%d skipped)"
                  % (src.name, ok, split, bad))

    # ── val top-up so the new classes are measurable at all ────────────────
    from PIL import Image
    for src, prefix, need_class, cap in VAL_TOPUP:
        if not src.exists():
            continue
        cls_id = CLASS_NAMES.index(need_class)
        moved = 0
        for lbl in sorted((src / "labels").glob(prefix + "*.txt")):
            if cap is not None and moved >= cap:
                break
            # only take images that actually contain the class we lack
            if not any(l.split() and l.split()[0] == str(cls_id)
                       for l in lbl.read_text().splitlines()):
                continue
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = src / "images" / (lbl.stem + ext)
                if cand.exists():
                    img = cand
                    break
            if img is None:
                continue
            # it was already copied into train — remove it there so the same
            # image is not in both splits
            for ext in (".jpg", ".jpeg", ".png"):
                stale = out / "images" / "train" / (lbl.stem + ext)
                if stale.exists():
                    stale.unlink()
            stale_lbl = out / "labels" / "train" / (lbl.stem + ".txt")
            if stale_lbl.exists():
                for line in stale_lbl.read_text().splitlines():
                    if line.split():
                        counts["train"][CLASS_NAMES[int(line.split()[0])]] -= 1
                stale_lbl.unlink()

            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception:
                continue
            lines = parse_label(lbl, counts["val"], already_seg=False)
            if lines is None:
                continue
            shutil.copy(img, out / "images" / "val" / img.name)
            (out / "labels" / "val" / (lbl.stem + ".txt")).write_text("\n".join(lines))
            moved += 1
        print("val top-up: %-14s %3d images moved train -> val" % (need_class, moved))

    names_yaml = "\n".join("  %d: %s" % (i, n) for i, n in enumerate(CLASS_NAMES))
    (out / "dataset.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\n"
        "nc: %d\nnames:\n%s\n" % (out, len(CLASS_NAMES), names_yaml))

    n_train = len(list((out / "labels" / "train").glob("*.txt")))
    n_val = len(list((out / "labels" / "val").glob("*.txt")))
    print("\n=== unified 11-class segmentation dataset ===")
    print("train %d images   val %d images   (%d pairs skipped)"
          % (n_train, n_val, total_bad))
    print("\n%-16s %12s %10s" % ("class", "train", "val"))
    for name in CLASS_NAMES:
        print("%-16s %12d %10d" % (name, counts["train"][name], counts["val"][name]))
    missing = [n for n in CLASS_NAMES if counts["val"][n] == 0]
    if missing:
        print("\nNOTE: no val instances for: %s" % ", ".join(missing))
        print("Their mAP will read 0 regardless of how well the model does.")
    print("\ndataset.yaml -> %s/dataset.yaml" % out)


if __name__ == "__main__":
    main()

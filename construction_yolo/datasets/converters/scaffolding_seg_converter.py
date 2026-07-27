"""Auto-labelled scaffolding masks -> unified taxonomy, YOLO-seg polygons.

Consumes the labelme JSON written by prelabel_roboflow_seg.py. Scaffolding was
previously 605 crude YOLO-World boxes over 406 frames; the promptable
segmentation model found 3523 polygons over 1487 frames, so 1225 frames gained
scaffolding they previously lacked entirely. Those frames were not merely
unlabelled — they were teaching the model that visible scaffolding is
background, which is worse than absence.

Polygons are kept rather than reduced to boxes. Scaffolding wraps building
facades in irregular shapes, so a box around one is largely building.

These are foundation-model pseudo-labels, not human annotation. Spot-checking
found the localisation sound and the boundaries loose — they trace the
scaffolded facade rather than individual tubes. Worth stating plainly in the
thesis as distillation from a foundation model rather than ground truth.

Usage:
    python construction_yolo/datasets/converters/scaffolding_seg_converter.py \
        --src /run/media/dvaghani/Expansion/Yolo/prelabel_scaffolding
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

OUT_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/scaffolding_seg")
SCAFFOLD_ID = CLASS_ID["scaffolding"]
MIN_AREA_FRAC = 0.0008     # discard slivers that survived the prelabel filter


def polygon_area(pts) -> float:
    """Shoelace, in pixel units."""
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def to_yolo_seg(pts, width, height) -> str | None:
    coords = []
    for x, y in pts:
        coords.append(min(max(x / width, 0.0), 1.0))
        coords.append(min(max(y / height, 0.0), 1.0))
    if len(coords) < 6:
        return None
    return "%d %s" % (SCAFFOLD_ID, " ".join("%.6f" % c for c in coords))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="labelme dir from prelabel_roboflow_seg")
    p.add_argument("--split", default="train", choices=["train", "val"],
                   help="which split these frames belong to")
    args = p.parse_args()

    src = Path(args.src)
    img_out = OUT_ROOT / "images" / args.split
    lbl_out = OUT_ROOT / "labels" / args.split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    n_img = n_empty = n_dropped = 0
    for jf in sorted(src.glob("*.json")):
        doc = json.loads(jf.read_text())
        W, H = doc.get("imageWidth"), doc.get("imageHeight")
        if not W or not H:
            continue

        lines = []
        for s in doc.get("shapes", []):
            if s.get("shape_type") != "polygon":
                continue
            pts = s.get("points") or []
            if len(pts) < 3:
                continue
            if polygon_area(pts) < MIN_AREA_FRAC * W * H:
                n_dropped += 1
                continue
            line = to_yolo_seg(pts, W, H)
            if line:
                lines.append(line)
                counts["scaffolding"] += 1

        if not lines:
            n_empty += 1
            continue

        img_name = doc.get("imagePath") or (jf.stem + ".jpg")
        img_path = src / img_name
        if not img_path.exists():
            continue
        stem = "scafseg_" + jf.stem
        shutil.copy(img_path, img_out / (stem + img_path.suffix))
        (lbl_out / (stem + ".txt")).write_text("\n".join(lines))
        n_img += 1

    print("scaffolding polygons: %d over %d images" % (counts["scaffolding"], n_img))
    print("  %d images had no polygon (skipped), %d slivers dropped"
          % (n_empty, n_dropped))
    print("-> %s (%s split)" % (OUT_ROOT, args.split))


if __name__ == "__main__":
    main()

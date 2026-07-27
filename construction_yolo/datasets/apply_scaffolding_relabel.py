"""Replace scaffolding labels in our own frames with the auto-labelled masks.

The auto-labelled scaffolding covers the same DJI frames that already carry
labels for person, machinery, vehicle and the rest. Adding it as a separate
source would duplicate every frame with contradictory annotations, so instead
each frame's scaffolding entries are swapped out and everything else kept.

Writes to labels_seg/ beside the originals rather than editing in place, so the
original pseudo-labels remain for comparison and nothing is destroyed. All
non-scaffolding boxes are converted to four-corner polygons here too, so the
output is already YOLO-seg.

Usage:
    python construction_yolo/datasets/apply_scaffolding_relabel.py \
        --prelabel /run/media/dvaghani/Expansion/Yolo/prelabel_scaffolding \
        --split train
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.taxonomy import CLASS_ID, CLASS_NAMES  # noqa: E402

OWN = Path("/run/media/dvaghani/Expansion/Yolo/construction_dataset")
SCAFFOLD_ID = CLASS_ID["scaffolding"]
MIN_AREA_FRAC = 0.0008


def box_to_polygon(cls_id, cx, cy, w, h) -> str:
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    pts = [x0, y0, x1, y0, x1, y1, x0, y1]
    pts = [min(max(v, 0.0), 1.0) for v in pts]
    return "%d %s" % (cls_id, " ".join("%.6f" % v for v in pts))


def polygon_area(pts) -> float:
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prelabel", required=True)
    p.add_argument("--split", default="train", choices=["train", "val"])
    args = p.parse_args()

    pre = Path(args.prelabel)
    src_lbl = OWN / "labels" / args.split
    out_lbl = OWN / "labels_seg" / args.split
    out_lbl.mkdir(parents=True, exist_ok=True)

    before, after = Counter(), Counter()
    n_files = n_with_new = n_lost = 0

    for lbl in sorted(src_lbl.glob("*.txt")):
        kept = []
        had_scaffold = False
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            before[cid] += 1
            if cid == SCAFFOLD_ID:
                had_scaffold = True
                continue                      # dropped, replaced below
            vals = [float(v) for v in parts[1:]]
            kept.append(box_to_polygon(cid, *vals))
            after[cid] += 1

        jf = pre / (lbl.stem + ".json")
        n_new = 0
        if jf.exists():
            doc = json.loads(jf.read_text())
            W, H = doc.get("imageWidth"), doc.get("imageHeight")
            for s in doc.get("shapes", []):
                pts = s.get("points") or []
                if s.get("shape_type") != "polygon" or len(pts) < 3:
                    continue
                if not W or not H or polygon_area(pts) < MIN_AREA_FRAC * W * H:
                    continue
                coords = []
                for x, y in pts:
                    coords.append(min(max(x / W, 0.0), 1.0))
                    coords.append(min(max(y / H, 0.0), 1.0))
                kept.append("%d %s" % (SCAFFOLD_ID,
                                       " ".join("%.6f" % c for c in coords)))
                after[SCAFFOLD_ID] += 1
                n_new += 1

        # a frame that had scaffolding before and none now would silently lose
        # the class; worth counting rather than hiding
        if had_scaffold and n_new == 0:
            n_lost += 1
        if n_new:
            n_with_new += 1

        if kept:
            (out_lbl / lbl.name).write_text("\n".join(kept))
            n_files += 1

    print("%s: wrote %d label files -> %s" % (args.split, n_files, out_lbl))
    print("  %d frames gained auto-labelled scaffolding" % n_with_new)
    print("  %d frames had scaffolding before but none now (check these)" % n_lost)
    print("\n%-16s %10s %10s" % ("class", "before", "after"))
    for i, name in enumerate(CLASS_NAMES):
        if before[i] or after[i]:
            print("%-16s %10d %10d" % (name, before[i], after[i]))


if __name__ == "__main__":
    main()

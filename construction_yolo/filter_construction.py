"""
Filter the pre-labeled pool down to construction-relevant frames.

Most extracted frames are street scenes whose only content is cars/people —
classes base YOLO already handles and that flood the labeling budget. This
keeps only frames containing at least one "construction" class, and reports
how many frames each class appears in (frame coverage, not box count).

Usage:
    python filter_construction.py \
        --pool "/run/media/dvaghani/Expansion/Yolo/labeling_pool" \
        --dst  "/run/media/dvaghani/Expansion/Yolo/label_subset"
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

CLASS_NAMES = ["container", "pole", "scaffolding", "crane", "person",
               "machinery", "suspended-loads", "vehicle", "building",
               "bridge", "barrier", "cable"]

# Classes that make a frame worth labeling. person/vehicle alone don't.
CONSTRUCTION = {0, 2, 3, 5, 6, 8, 9, 10, 11}   # exclude pole(1), person(4), vehicle(7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--dst",  required=True)
    p.add_argument("--min-construction", type=int, default=1,
                   help="Min. number of construction-class boxes to keep a frame.")
    args = p.parse_args()

    pool = Path(args.pool)
    img_src, lbl_src = pool / "images", pool / "labels"
    dst = Path(args.dst)
    img_dst, lbl_dst = dst / "images", dst / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    frame_cov = Counter()      # frames containing >=1 box of a class
    kept = 0
    total = 0
    for lbl in sorted(lbl_src.glob("*.txt")):
        total += 1
        classes = [int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()]
        n_constr = sum(1 for c in classes if c in CONSTRUCTION)
        if n_constr < args.min_construction:
            continue
        img = img_src / f"{lbl.stem}.jpg"
        if not img.exists():
            continue
        shutil.copy(img, img_dst / img.name)
        shutil.copy(lbl, lbl_dst / lbl.name)
        kept += 1
        for c in set(classes):
            frame_cov[c] += 1

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (dst / "dataset.yaml").write_text(
        f"path: {dst}\ntrain: images\nval: images  # split later\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n{names}\n")

    print(f"Kept {kept} / {total} frames (>= {args.min_construction} construction box)")
    print("\nFrame coverage (how many kept frames contain each class):")
    for i, n in enumerate(CLASS_NAMES):
        tag = "  <- construction" if i in CONSTRUCTION else ""
        print(f"  {i:2d} {n:16s} {frame_cov[i]:5d}{tag}")
    print(f"\n-> {img_dst}")


if __name__ == "__main__":
    main()

"""
Merge round-1 and round-2 construction subsets into one labeling dataset.

- Drops dead classes (cable, bridge, barrier) that don't exist in the footage.
- Remaps the surviving classes to a compact 0..N taxonomy.
- Drops the false-positive-heavy suspended-loads by default (--keep-suspended
  to keep it).
- Cross-dedups so near-identical frames shared between rounds aren't repeated.
- Drops label files that become empty after class removal.

Usage:
    python merge_rounds.py \
        --r1 "/run/media/dvaghani/Expansion/Yolo/label_subset" \
        --r2 "/run/media/dvaghani/Expansion/Yolo/label_subset_round2" \
        --dst "/run/media/dvaghani/Expansion/Yolo/construction_dataset"
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

OLD = ["container", "pole", "scaffolding", "crane", "person", "machinery",
       "suspended-loads", "vehicle", "building", "bridge", "barrier", "cable"]


def ahash(img, size=16):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return (g > g.mean()).flatten()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r1", required=True)
    p.add_argument("--r2", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--keep-suspended", action="store_true")
    p.add_argument("--min-hamming", type=int, default=12,
                   help="Cross-dedup threshold.")
    args = p.parse_args()

    drop = {"bridge", "barrier", "cable"}
    if not args.keep_suspended:
        drop.add("suspended-loads")
    survivors = [c for c in OLD if c not in drop]
    old2new = {OLD.index(c): i for i, c in enumerate(survivors)}
    print("New taxonomy:", {i: c for i, c in enumerate(survivors)})

    dst = Path(args.dst)
    img_dst, lbl_dst = dst / "images", dst / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    hashes = []           # kept-frame hashes for cross-dedup
    cov = Counter()
    kept = 0
    empty = 0
    dup = 0

    for src in (args.r1, args.r2):
        src = Path(src)
        for lbl in sorted((src / "labels").glob("*.txt")):
            img = src / "images" / f"{lbl.stem}.jpg"
            if not img.exists():
                continue

            # remap + filter labels
            out_lines = []
            for line in lbl.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cid = int(parts[0])
                if cid not in old2new:
                    continue
                out_lines.append(f"{old2new[cid]} {' '.join(parts[1:])}")
            if not out_lines:            # nothing survived → skip frame
                empty += 1
                continue

            # cross-dedup
            im = cv2.imread(str(img))
            if im is None:
                continue
            h = ahash(im)
            if any(np.count_nonzero(h != hh) < args.min_hamming for hh in hashes):
                dup += 1
                continue
            hashes.append(h)

            # write (prefix name with round to avoid collisions)
            stem = f"{src.name}_{lbl.stem}"
            shutil.copy(img, img_dst / f"{stem}.jpg")
            (lbl_dst / f"{stem}.txt").write_text("\n".join(out_lines))
            for l in out_lines:
                cov[int(l.split()[0])] += 1
            kept += 1

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(survivors))
    (dst / "dataset.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(survivors)}\nnames:\n{names}\n")

    print(f"\nKept {kept} frames  (dropped {empty} now-empty, {dup} cross-dups)")
    print("Box counts in merged set:")
    for i, c in enumerate(survivors):
        print(f"  {i} {c:16s} {cov[i]}")
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()

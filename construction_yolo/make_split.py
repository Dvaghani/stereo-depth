"""
Create a train/val split for the merged construction dataset in the layout
Ultralytics YOLO expects, using symlinks (no copying, saves disk).

    construction_dataset/
        images/train/*.jpg   labels/train/*.txt
        images/val/*.jpg     labels/val/*.txt

Split is grouped by source video so near-duplicate frames of one scene don't
straddle train and val (which would inflate val scores).

Usage:
    python make_split.py --ds "/run/media/dvaghani/Expansion/Yolo/construction_dataset" --val-frac 0.15
"""
import argparse
import os
import random
import re
import shutil
from pathlib import Path


def source_key(stem: str) -> str:
    # round-1: label_subset_DJI_0006_frame_000295            -> DJI_0006
    m = re.search(r"DJI_\d+(?=_frame_)", stem)
    if m:
        return m.group(0)
    # round-1 flat: label_subset_DJI_0014_0032               -> DJI_0014
    m = re.search(r"DJI_\d+(?=_\d+$)", stem)
    if m:
        return m.group(0)
    # round-2: label_subset_round2_DJI_20260429110044_0002_W_0005 -> DJI_..._0002_W
    m = re.search(r"DJI_\d+_\d+_[A-Z]", stem)
    if m:
        return m.group(0)
    return stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ds", required=True)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds = Path(args.ds)
    img_flat, lbl_flat = ds / "images", ds / "labels"
    stems = sorted(f.stem for f in img_flat.glob("*.jpg"))

    # group by source video, split whole groups into val
    groups = {}
    for s in stems:
        groups.setdefault(source_key(s), []).append(s)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)

    n_val_target = int(len(stems) * args.val_frac)
    val_stems, n_val = set(), 0
    for k in keys:
        if n_val < n_val_target:
            val_stems.update(groups[k]); n_val += len(groups[k])

    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)

    def link(src, dst):
        # exFAT/NTFS external drives don't support symlinks — copy instead.
        if dst.exists():
            dst.unlink()
        shutil.copy(src, dst)

    n_tr = 0
    for s in stems:
        split = "val" if s in val_stems else "train"
        n_tr += split == "train"
        link(img_flat / f"{s}.jpg", ds / "images" / split / f"{s}.jpg")
        link(lbl_flat / f"{s}.txt", ds / "labels" / split / f"{s}.txt")

    # rewrite dataset.yaml with split paths
    names = ["container", "pole", "scaffolding", "crane",
             "person", "machinery", "vehicle", "building"]
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    (ds / "dataset.yaml").write_text(
        f"path: {ds}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames:\n{names_yaml}\n")

    print(f"train {n_tr}  val {len(val_stems)}  "
          f"(val videos: {sorted({source_key(s) for s in val_stems})})")


if __name__ == "__main__":
    main()

"""
Build a focused correction set: only the frames that contain container, pole,
or crane (the 3 classes the baseline showed were too noisy to train on).

Copies those images + their existing (imperfect) labels into a separate
folder, ready to open directly in labelImg. Correcting only these frames is
far faster than reviewing the whole 2,521-frame dataset.

Usage:
    python build_correction_set.py \
        --ds "/run/media/dvaghani/Expansion/Yolo/construction_dataset" \
        --out "/run/media/dvaghani/Expansion/Yolo/correction_set"
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

# class ids to target, per construction_dataset/dataset.yaml
TARGET = {0: "container", 1: "pole", 3: "crane"}
CLASS_NAMES = ["container", "pole", "scaffolding", "crane",
               "person", "machinery", "vehicle", "building"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ds",  required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train", choices=["train", "val", "both"])
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total frames, evenly sampled, for a manageable first pass.")
    args = p.parse_args()

    ds = Path(args.ds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val"] if args.split == "both" else [args.split]

    # First pass: find every candidate frame, tagging which target classes it has.
    candidates = []
    for split in splits:
        lbl_dir = ds / "labels" / split
        img_dir = ds / "images" / split
        for lbl in sorted(lbl_dir.glob("*.txt")):
            classes = {int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}
            hit = classes & TARGET.keys()
            if not hit:
                continue
            img = img_dir / f"{lbl.stem}.jpg"
            if not img.exists():
                continue
            candidates.append((lbl, img, hit))

    if args.limit and len(candidates) > args.limit:
        # round-robin over classes so the capped set stays balanced, not
        # dominated by whichever class happens to appear most often (pole).
        buckets = {c: [] for c in TARGET}
        for item in candidates:
            for c in item[2]:
                buckets[c].append(item)
        chosen, seen = [], set()
        while len(chosen) < args.limit and any(buckets.values()):
            for c in list(TARGET):
                if len(chosen) >= args.limit:
                    break
                while buckets[c]:
                    item = buckets[c].pop(0)
                    if item[0] not in seen:
                        seen.add(item[0]); chosen.append(item)
                        break
        candidates = chosen

    kept = 0
    per_class = Counter()
    for lbl, img, hit in candidates:
        shutil.copy(img, out / img.name)
        shutil.copy(lbl, out / lbl.name)
        kept += 1
        for c in hit:
            per_class[c] += 1

    (out / "classes.txt").write_text("\n".join(CLASS_NAMES))

    print(f"Correction set: {kept} frames -> {out}")
    print("Frames containing each target class:")
    for cid, name in TARGET.items():
        print(f"  {name:12s} {per_class[cid]}")
    print(f"\nOpen with:")
    print(f'  labelImg "{out}" "{out}/classes.txt" "{out}"')


if __name__ == "__main__":
    main()

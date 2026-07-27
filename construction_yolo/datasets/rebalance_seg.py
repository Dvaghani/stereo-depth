"""Rebalance the merged dataset by subsampling person/vehicle-only images.

person is 60% of all training instances and vehicle another 21%, which starves
the classes the detector actually exists for. This thins that dominance.

Two rules make it safe:

  Removal is IMAGE-LEVEL, never label-level. Deleting person boxes from an
  image while keeping the image would leave those people unlabelled, teaching
  the model they are background — the partial-labelling trap the converters
  already avoid.

  Any image containing a rare class is kept unconditionally. Only images whose
  sole content is person and/or vehicle are candidates, so no container, pole,
  scaffolding, crane, building, suspended-load, barrier or cable instance is
  ever lost.

Sampling is stratified by source prefix, so the COCO / OpenImages / MOCS /
Roboflow mix is preserved rather than accidentally dropping one viewpoint
entirely — COCO in particular supplies the ground-level geometry that matches
the Brio rig.

Note person cannot fall below ~60k however aggressive this is: MOCS crane and
machinery frames are full of workers, and those images must be kept.

Usage:
    python construction_yolo/datasets/rebalance_seg.py --dry-run
    python construction_yolo/datasets/rebalance_seg.py --keep-frac 0.25 --apply
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.taxonomy import CLASS_NAMES  # noqa: E402

DATASET = Path("/run/media/dvaghani/Expansion/Yolo/unified_seg_dataset")
COMMON = {CLASS_NAMES.index("person"), CLASS_NAMES.index("vehicle")}


def source_of(stem: str) -> str:
    """Group by originating converter so sampling stays proportional."""
    for pre in ("coco_", "roboflow_", "openimages_", "mocs_", "barricade_barrier_",
                "utility_pole_", "owncable_", "rfcable_"):
        if stem.startswith(pre):
            return pre.rstrip("_")
    return "own"


def scan(lbl_dir: Path):
    per_image = {}
    for f in sorted(lbl_dir.glob("*.txt")):
        counts = Counter()
        for line in f.read_text().splitlines():
            p = line.split()
            if p:
                counts[int(p[0])] += 1
        per_image[f] = counts
    return per_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keep-frac", type=float, default=0.25,
                   help="fraction of person/vehicle-only images to retain")
    p.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    lbl_dir = DATASET / "labels" / "train"
    img_dir = DATASET / "images" / "train"
    per_image = scan(lbl_dir)

    before = Counter()
    candidates = defaultdict(list)
    for f, counts in per_image.items():
        before.update(counts)
        if set(counts) <= COMMON:
            candidates[source_of(f.stem)].append(f)

    rng = random.Random(args.seed)
    drop = []
    for src, files in candidates.items():
        files = sorted(files)
        rng.shuffle(files)
        n_keep = int(round(len(files) * args.keep_frac))
        drop.extend(files[n_keep:])
        print("  %-18s %5d candidates -> keep %4d, drop %5d"
              % (src, len(files), n_keep, len(files) - n_keep))

    removed = Counter()
    for f in drop:
        removed.update(per_image[f])
    after = before - removed

    tot_b, tot_a = sum(before.values()), sum(after.values())
    print("\n%-16s %12s %12s %10s" % ("class", "before", "after", "share"))
    for i, name in enumerate(CLASS_NAMES):
        print("%-16s %12d %12d %9.1f%%"
              % (name, before[i], after[i], 100.0 * after[i] / max(tot_a, 1)))
    print("\nimages   %d -> %d  (drop %d, %.0f%%)"
          % (len(per_image), len(per_image) - len(drop), len(drop),
             100.0 * len(drop) / len(per_image)))
    print("person share %.0f%% -> %.0f%%"
          % (100.0 * before[CLASS_NAMES.index("person")] / tot_b,
             100.0 * after[CLASS_NAMES.index("person")] / tot_a))
    print("est. epoch time change: %.0f%% faster" % (100.0 * len(drop) / len(per_image)))

    if not args.apply:
        print("\nDRY RUN — nothing deleted. Re-run with --apply to commit.")
        return

    n = 0
    for f in drop:
        for ext in (".jpg", ".jpeg", ".png"):
            img = img_dir / (f.stem + ext)
            if img.exists():
                img.unlink()
                break
        f.unlink()
        n += 1
    print("\ndeleted %d image/label pairs from train" % n)


if __name__ == "__main__":
    main()

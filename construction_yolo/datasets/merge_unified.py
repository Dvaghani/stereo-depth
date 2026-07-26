"""
Merge our own labeled construction frames with the public-dataset converters
(COCO, Roboflow construction-safety, Open Images) into one unified YOLO
dataset, ready for training.

Design:
  - Our own train/val split (construction_dataset/) stays as the backbone —
    val is UNCHANGED so results remain comparable to the YOLO-World-only
    baseline already trained.
  - All public-dataset images (real ground truth, not pseudo-labels) are
    added to TRAIN ONLY — they broaden training signal for person/vehicle/
    machinery/building but shouldn't determine the in-domain validation
    score, which should reflect the actual deployment scenes.
  - Validates every image opens and every label file parses before copying.
  - Reports final per-class box counts and image counts per split.

Usage:
    python merge_unified.py --out "/run/media/dvaghani/Expansion/Yolo/unified_dataset"
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.taxonomy import CLASS_NAMES

OWN_DATASET = Path("/run/media/dvaghani/Expansion/Yolo/construction_dataset")
PUBLIC_SOURCES = [
    Path("/run/media/dvaghani/Expansion/Yolo/sources/coco_yolo"),
    Path("/run/media/dvaghani/Expansion/Yolo/sources/roboflow_yolo"),
    Path("/run/media/dvaghani/Expansion/Yolo/sources/openimages_yolo"),
    Path("/run/media/dvaghani/Expansion/Yolo/sources/mocs_yolo"),
]


def validate_pair(img_path: Path, lbl_path: Path) -> list[str] | None:
    """Returns parsed, validated label lines, or None if the pair is bad."""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            im.verify()
    except Exception:
        return None
    if not lbl_path.exists():
        return None
    lines = []
    for line in lbl_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if not (0 <= cid < len(CLASS_NAMES)):
            continue
        if not all(0.0 <= c <= 1.0 for c in coords):
            continue
        lines.append(line)
    return lines if lines else None


def copy_split(img_dir: Path, lbl_dir: Path, out_img: Path, out_lbl: Path,
               counts: Counter, img_counter: list, prefix: str = ""):
    n_bad = 0
    for lbl in sorted(lbl_dir.glob("*.txt")):
        stem = lbl.stem
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = img_dir / f"{stem}{ext}"
            if cand.exists():
                img = cand
                break
        if img is None:
            n_bad += 1
            continue
        lines = validate_pair(img, lbl)
        if lines is None:
            n_bad += 1
            continue
        name = f"{prefix}{stem}" if prefix else stem
        shutil.copy(img, out_img / f"{name}{img.suffix}")
        (out_lbl / f"{name}.txt").write_text("\n".join(lines))
        img_counter[0] += 1
        for line in lines:
            counts[int(line.split()[0])] += 1
    return n_bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_bad = 0

    # ── own dataset: both splits copied as-is ──────────────────────────────
    for split in ("train", "val"):
        counts = Counter(); n_img = [0]
        n_bad = copy_split(
            OWN_DATASET / "images" / split, OWN_DATASET / "labels" / split,
            out / "images" / split, out / "labels" / split,
            counts, n_img)
        total_bad += n_bad
        print(f"own/{split}: {n_img[0]} images validated, {n_bad} skipped (bad)")

    # ── public sources: all go into train ──────────────────────────────────
    for src in PUBLIC_SOURCES:
        if not src.exists():
            print(f"skip (not found): {src}")
            continue
        counts = Counter(); n_img = [0]
        n_bad = copy_split(
            src / "images", src / "labels",
            out / "images" / "train", out / "labels" / "train",
            counts, n_img, prefix="")
        total_bad += n_bad
        print(f"{src.name}: {n_img[0]} images merged into train, {n_bad} skipped")

    # ── final stats ───────────────────────────────────────────────────────
    print(f"\nTotal invalid pairs skipped: {total_bad}")
    final_counts = {"train": Counter(), "val": Counter()}
    final_imgs = {"train": 0, "val": 0}
    for split in ("train", "val"):
        lbl_dir = out / "labels" / split
        final_imgs[split] = len(list(lbl_dir.glob("*.txt")))
        for lbl in lbl_dir.glob("*.txt"):
            for line in lbl.read_text().splitlines():
                if line.strip():
                    final_counts[split][int(line.split()[0])] += 1

    print(f"\n=== Unified dataset ===")
    print(f"train: {final_imgs['train']} images   val: {final_imgs['val']} images\n")
    print(f"{'class':14s} {'train boxes':>12s} {'val boxes':>10s}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:14s} {final_counts['train'][i]:>12d} {final_counts['val'][i]:>10d}")

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (out / "dataset.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n{names_yaml}\n")
    print(f"\ndataset.yaml written -> {out}/dataset.yaml")


if __name__ == "__main__":
    main()

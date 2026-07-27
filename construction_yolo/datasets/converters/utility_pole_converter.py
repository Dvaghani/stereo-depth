"""Roboflow unstructured/utility-pole-detection -> unified taxonomy.

Source: https://universe.roboflow.com/unstructured/utility-pole-detection-birhf
Single class 'pole', 1310 street-level images of standing utility poles.

Why this one: `pole` scored 0.07 in the baseline — effectively broken — because
it had no ground truth at all, only YOLO-World pseudo-labels that fire on any
vertical line. Five other pole/container/scaffolding datasets were downloaded
and rejected on inspection (disaster-damage scenes with misaligned boxes,
container ships at ports, synthetic renders, and component-level scaffolding
labels), so this is the one that survived visual verification.

Its val split is routed to our val, giving `pole` real ground-truth validation
instead of measuring agreement with its own pseudo-labels.

Usage:
    python construction_yolo/datasets/converters/utility_pole_converter.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.converters.base import DatasetConverter  # noqa: E402
from datasets.taxonomy import CLASS_ID  # noqa: E402

SRC_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources_raw/utility_pole")
OUT_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/utility_pole_yolo")

# source is nc=1: index 0 -> 'pole'
SRC_MAP = {0: "pole"}

# our merge routes files by this prefix, mirroring the barrier/MOCS convention
SPLIT_PREFIX = {"train": "train", "valid": "valid", "test": "train"}


class UtilityPoleConverter(DatasetConverter):
    name = "utility_pole"

    def __init__(self):
        super().__init__(SRC_ROOT, OUT_ROOT)

    def convert(self) -> Counter:
        counts = Counter()
        for split in ("train", "valid", "test"):
            img_dir, lbl_dir = SRC_ROOT / split / "images", SRC_ROOT / split / "labels"
            if not img_dir.exists():
                continue
            for lbl in sorted(lbl_dir.glob("*.txt")):
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
                    if len(parts) != 5:
                        continue
                    src_id = int(parts[0])
                    if src_id not in SRC_MAP:
                        continue
                    unified = SRC_MAP[src_id]
                    lines.append("%d %s" % (CLASS_ID[unified], " ".join(parts[1:])))
                    counts[unified] += 1

                self._write_pair(img, "%s_%s" % (SPLIT_PREFIX[split], lbl.stem), lines)
        return counts


if __name__ == "__main__":
    conv = UtilityPoleConverter()
    counts = conv.convert()
    print("Utility-pole boxes emitted:")
    for name, n in counts.items():
        print("  %-12s %d" % (name, n))
    print("\n-> %s" % OUT_ROOT)

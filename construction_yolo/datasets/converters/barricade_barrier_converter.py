"""
Roboflow divolka/barricade-barrier -> unified taxonomy converter.

Source: https://universe.roboflow.com/divolka/barricade-barrier (v1)
Source classes: [barricade, barrier] (both red/white striped construction
barrier types — barricade = horizontal fence-panel type, barrier = free-
standing pole/panel type). Both map to our single unified `barrier` class.

Usage:
    python barricade_barrier_converter.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.converters.base import DatasetConverter
from datasets.taxonomy import CLASS_ID

SRC_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources_raw/barricade_barrier")
OUT_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/barricade_barrier_yolo")

# source class index -> unified class name (both map to the same class)
SRC_MAP = {0: "barrier", 1: "barrier"}


class BarricadeBarrierConverter(DatasetConverter):
    name = "barricade_barrier"

    def __init__(self):
        super().__init__(SRC_ROOT, OUT_ROOT)

    def convert(self) -> Counter:
        counts = Counter()
        for split in ("train", "valid", "test"):
            img_dir = SRC_ROOT / split / "images"
            lbl_dir = SRC_ROOT / split / "labels"
            if not img_dir.exists():
                continue
            for lbl in sorted(lbl_dir.glob("*.txt")):
                img = None
                for ext in (".jpg", ".jpeg", ".png"):
                    cand = img_dir / f"{lbl.stem}{ext}"
                    if cand.exists():
                        img = cand
                        break
                if img is None:
                    continue

                lines = []
                for line in lbl.read_text().splitlines():
                    if not line.strip():
                        continue
                    parts = line.split()
                    src_id = int(parts[0])
                    if src_id not in SRC_MAP:
                        continue
                    unified = SRC_MAP[src_id]
                    cls_id = CLASS_ID[unified]
                    lines.append(f"{cls_id} {' '.join(parts[1:])}")
                    counts[unified] += 1

                self._write_pair(img, f"{split}_{lbl.stem}", lines)
        return counts


if __name__ == "__main__":
    conv = BarricadeBarrierConverter()
    counts = conv.convert()
    print("Barricade-barrier boxes emitted:")
    for name, n in counts.items():
        print(f"  {name:12s} {n}")
    print(f"\n-> {OUT_ROOT}")

"""
Roboflow construction-safety dataset -> unified taxonomy converter.

Source classes: [Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
                 Safety Cone, Safety Vest, machinery, vehicle]
Only Person/machinery/vehicle map into our taxonomy — the PPE-compliance
classes (Hardhat, Mask, Safety Vest, Safety Cone, ...) have no equivalent and
are dropped (not emitted as any class).

Usage:
    python roboflow_converter.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.converters.base import DatasetConverter
from datasets.taxonomy import CLASS_ID

SRC_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/roboflow_raw/css-data")
OUT_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/roboflow_yolo")

# source class index -> unified class name (only mapped ones listed)
SRC_MAP = {5: "person", 8: "machinery", 9: "vehicle"}


class RoboflowConverter(DatasetConverter):
    name = "roboflow"

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
    conv = RoboflowConverter()
    counts = conv.convert()
    print("Roboflow boxes emitted:")
    for name, n in counts.items():
        print(f"  {name:12s} {n}")
    print(f"\n-> {OUT_ROOT}")

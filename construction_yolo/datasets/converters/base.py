"""
Base interface every per-dataset converter implements.

A converter's job: read a source dataset in its native format, and write
YOLO-format (image, label.txt) pairs into a common output folder, with class
ids already remapped to the unified 8-class taxonomy (see datasets/taxonomy.py).

Contract:
  - Every emitted label line uses the unified class ids.
  - Source classes with NO mapping are simply not emitted (never invented).
  - Images with zero mapped boxes after filtering are skipped (avoids the
    "partial-labeling" trap of teaching used categories as background).
  - Each converter reports a per-class box count so coverage is auditable.
"""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path


class DatasetConverter(ABC):
    name: str = "base"

    def __init__(self, src_root: str | Path, out_root: str | Path):
        self.src_root = Path(src_root)
        self.out_root = Path(out_root)
        (self.out_root / "images").mkdir(parents=True, exist_ok=True)
        (self.out_root / "labels").mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def convert(self) -> Counter:
        """Run the conversion. Returns a Counter of {class_name: box_count}."""
        raise NotImplementedError

    def _write_pair(self, img_src: Path, stem: str, yolo_lines: list[str]):
        """Copy image + write label under a converter-prefixed stem, avoiding
        filename collisions when multiple sources are later merged."""
        if not yolo_lines:
            return False
        out_stem = f"{self.name}_{stem}"
        shutil.copy(img_src, self.out_root / "images" / f"{out_stem}{img_src.suffix}")
        (self.out_root / "labels" / f"{out_stem}.txt").write_text("\n".join(yolo_lines))
        return True

    @staticmethod
    def to_yolo_line(cls_id: int, x1: float, y1: float, x2: float, y2: float,
                     img_w: int, img_h: int) -> str:
        cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
        bw, bh = (x2 - x1) / img_w, (y2 - y1) / img_h
        return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

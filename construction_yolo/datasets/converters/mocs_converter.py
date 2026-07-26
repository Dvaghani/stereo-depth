"""
MOCS (Moving Objects in Construction Sites) -> unified taxonomy converter.

Source: https://www.kaggle.com/datasets/xiaopan9802/mocs-dataset

MOCS's 13 official categories map to our taxonomy as follows (agreed mapping):
    Worker                    -> person
    Static crane, Crane       -> crane
    Hanging head              -> suspended-load
    Roller, Bulldozer,
      Excavator, Loader,
      Pile driving             -> machinery
    Truck, Pump truck,
      Concrete mixer,
      Other vehicle            -> vehicle

No MOCS category maps to container / pole / scaffolding / building / barrier /
cable — those remain unlabeled by this source.

This converter auto-detects the actual on-disk layout after extraction
(structure varies release to release — COCO-style JSON, or per-image XML,
or already-YOLO txt) and handles the common cases.

Usage:
    python mocs_converter.py --src /path/to/extracted/mocs
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.converters.base import DatasetConverter
from datasets.taxonomy import CLASS_ID

OUT_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/mocs_yolo")

# MOCS category name (as it appears in their annotation files) -> unified class
MOCS_MAP = {
    "worker": "person",
    "static crane": "crane",
    "crane": "crane",
    "hanging head": "suspended-load",
    "roller": "machinery",
    "bulldozer": "machinery",
    "excavator": "machinery",
    "loader": "machinery",
    "pile driving": "machinery",
    "truck": "vehicle",
    "pump truck": "vehicle",
    "concrete mixer": "vehicle",
    "concrete transport mixer": "vehicle",
    "other vehicle": "vehicle",
}


class MOCSConverter(DatasetConverter):
    name = "mocs"

    def __init__(self, src_root: Path):
        super().__init__(src_root, OUT_ROOT)

    def convert(self) -> Counter:
        # Try COCO-format json first (most common export format for MOCS mirrors)
        json_files = list(self.src_root.rglob("*.json"))
        if json_files:
            return self._convert_coco_json(json_files)

        # Fallback: already YOLO txt labels sitting next to images
        txt_files = list(self.src_root.rglob("*.txt"))
        if txt_files:
            return self._convert_existing_yolo()

        raise SystemExit(f"No .json or .txt annotation files found under {self.src_root}. "
                         f"Run with --inspect to see the actual layout.")

    def _convert_coco_json(self, json_files: list[Path]) -> Counter:
        counts = Counter()

        # Build a filename -> path index once (handles arbitrary nesting,
        # e.g. MOCS's instances_train/instances_train/*.jpg layout).
        print("Indexing image files...")
        by_name: dict[str, Path] = {}
        for p in self.src_root.rglob("*.jpg"):
            by_name[p.name] = p
        print(f"  indexed {len(by_name)} images")

        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
                continue   # not a COCO-style annotation file

            split_tag = jf.stem.replace("instances_", "")   # train / val
            cat_name = {c["id"]: c["name"].strip().lower() for c in data.get("categories", [])}
            img_info = {im["id"]: im for im in data["images"]}
            anns_by_img: dict[int, list] = {}
            for a in data["annotations"]:
                anns_by_img.setdefault(a["image_id"], []).append(a)

            n_missing = 0
            for img_id, anns in anns_by_img.items():
                info = img_info.get(img_id)
                if info is None:
                    continue
                img_path = by_name.get(info["file_name"])
                if img_path is None:
                    n_missing += 1
                    continue

                W, H = info.get("width"), info.get("height")
                lines = []
                for a in anns:
                    src_cat = cat_name.get(a["category_id"], "")
                    unified = MOCS_MAP.get(src_cat)
                    if unified is None:
                        continue
                    cls_id = CLASS_ID[unified]
                    x, y, w, h = a["bbox"]
                    lines.append(self.to_yolo_line(cls_id, x, y, x + w, y + h, W, H))
                    counts[unified] += 1

                self._write_pair(img_path, f"{split_tag}_{Path(info['file_name']).stem}", lines)
            if n_missing:
                print(f"  {jf.name}: {n_missing} images referenced but not found on disk")

        return counts

    def _convert_existing_yolo(self) -> Counter:
        # Placeholder for the (less common) already-YOLO-format release.
        # Needs the actual classes.txt / data.yaml to map indices correctly —
        # inspect the layout first if this path is hit.
        raise SystemExit("Found .txt files but no COCO json — inspect layout "
                         "manually (classes.txt / data.yaml ordering needed).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Root of the extracted MOCS dataset")
    p.add_argument("--inspect", action="store_true",
                   help="Just print the directory structure, don't convert.")
    args = p.parse_args()

    src = Path(args.src)
    if args.inspect:
        for p in sorted(src.rglob("*"))[:60]:
            print(p.relative_to(src))
        raise SystemExit(0)

    conv = MOCSConverter(src)
    counts = conv.convert()
    print("\nMOCS boxes emitted:")
    for name, n in counts.items():
        print(f"  {name:12s} {n}")
    print(f"\n-> {OUT_ROOT}")

"""
Open Images V7 (validation split) -> unified taxonomy converter.

Uses the official Google-hosted CSVs + the public S3 image mirror
(s3.amazonaws.com/open-images-dataset) — no API key, no fiftyone dependency.
Validation split only (not the 2GB+ train bbox file) — still gives several
thousand images per class, proportionate to how much this source adds.

Open Images mapping — this is the ONLY source in the pipeline with a real
"Container" class, which directly targets our weakest category:
    Person            -> person
    Car, Truck, Van    -> vehicle
    Building           -> building
    Container          -> container
(No Open Images equivalent for pole/scaffolding/crane — genuinely absent.)

Usage:
    python openimages_converter.py --limit 1200
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.converters.base import DatasetConverter
from datasets.taxonomy import CLASS_ID

SRC_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/openimages")
OUT_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/openimages_yolo")
S3_BASE = "https://s3.amazonaws.com/open-images-dataset/validation"

# MID (Open Images machine id) -> unified class name
MID_MAP = {
    "/m/01g317": "person",
    "/m/0k4j":   "vehicle",   # Car
    "/m/07r04":  "vehicle",   # Truck
    "/m/0h2r6":  "vehicle",   # Van
    "/m/0cgh4":  "building",
    "/m/011q46kg": "container",
}


class OpenImagesConverter(DatasetConverter):
    name = "openimages"

    def __init__(self, limit_per_class: int):
        super().__init__(SRC_ROOT, OUT_ROOT)
        self.limit_per_class = limit_per_class

    def convert(self) -> Counter:
        bbox_csv = SRC_ROOT / "val-bbox.csv"

        # Pass 1: group rows by image, but only keep images that contain at
        # least one target class, capped per class for a balanced pull.
        rows_by_image: dict[str, list[dict]] = defaultdict(list)
        per_class_img_count = Counter()
        chosen_images: set[str] = set()

        with open(bbox_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row["LabelName"]
                if mid not in MID_MAP:
                    continue
                unified = MID_MAP[mid]
                img_id = row["ImageID"]
                if img_id not in chosen_images:
                    if per_class_img_count[unified] >= self.limit_per_class:
                        continue
                    chosen_images.add(img_id)
                    per_class_img_count[unified] += 1
                rows_by_image[img_id].append(row)

        print(f"Open Images: downloading {len(chosen_images)} images...")

        img_dir = SRC_ROOT / "images"
        img_dir.mkdir(exist_ok=True)

        def fetch(img_id):
            dst = img_dir / f"{img_id}.jpg"
            if dst.exists():
                return img_id
            for attempt in range(3):
                try:
                    r = requests.get(f"{S3_BASE}/{img_id}.jpg", timeout=15)
                    if r.ok:
                        dst.write_bytes(r.content)
                        return img_id
                    return None
                except requests.exceptions.RequestException:
                    if attempt == 2:
                        return None
            return None

        downloaded = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(fetch, iid) for iid in chosen_images]
            for i, fut in enumerate(as_completed(futs)):
                res = fut.result()
                if res:
                    downloaded.append(res)
                if (i + 1) % 200 == 0:
                    print(f"  {i+1}/{len(chosen_images)}")

        counts = Counter()
        from PIL import Image
        for img_id in downloaded:
            img_path = img_dir / f"{img_id}.jpg"
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                continue

            lines = []
            for row in rows_by_image[img_id]:
                unified = MID_MAP[row["LabelName"]]
                cls_id = CLASS_ID[unified]
                # Open Images boxes are already normalized 0-1, XMin/XMax/YMin/YMax
                x1, x2 = float(row["XMin"]), float(row["XMax"])
                y1, y2 = float(row["YMin"]), float(row["YMax"])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bw, bh = (x2 - x1), (y2 - y1)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                counts[unified] += 1

            self._write_pair(img_path, img_id, lines)

        return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=1200,
                   help="Max images per class (validation split only).")
    args = p.parse_args()

    conv = OpenImagesConverter(limit_per_class=args.limit)
    counts = conv.convert()
    print("\nOpen Images boxes emitted:")
    for name, n in counts.items():
        print(f"  {name:12s} {n}")
    print(f"\n-> {OUT_ROOT}")

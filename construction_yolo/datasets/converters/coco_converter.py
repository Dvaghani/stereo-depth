"""
COCO -> unified taxonomy converter.

Downloads ONLY the annotation file (not the 19GB image archive) and then
fetches individual images for the categories we care about via their
coco_url. This keeps the pull small and targeted instead of grabbing all of
COCO.

COCO mapping (only 2 of our 8 classes exist in COCO — everything else is
simply absent from this dataset, so nothing else is emitted):
    person            -> person
    car, truck, bus   -> vehicle

Usage:
    python coco_converter.py --limit 2500
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.converters.base import DatasetConverter
from datasets.taxonomy import CLASS_ID

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_CATS = {"person": "person", "car": "vehicle", "truck": "vehicle", "bus": "vehicle"}

SRC_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/coco")
OUT_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/coco_yolo")


def ensure_annotations():
    ann_dir = SRC_ROOT / "annotations"
    ann_file = ann_dir / "instances_train2017.json"
    if ann_file.exists():
        return ann_file
    SRC_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = SRC_ROOT / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print("Downloading COCO annotations (~241MB)...")
        r = requests.get(ANN_URL, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extract("annotations/instances_train2017.json", SRC_ROOT)
    zip_path.unlink()   # free the zip, keep only the json we need
    return ann_file


class COCOConverter(DatasetConverter):
    name = "coco"

    def __init__(self, limit_per_class: int):
        super().__init__(SRC_ROOT, OUT_ROOT)
        self.limit_per_class = limit_per_class

    def convert(self) -> Counter:
        from pycocotools.coco import COCO

        ann_file = ensure_annotations()
        coco = COCO(str(ann_file))

        cat_ids = coco.getCatIds(catNms=list(COCO_CATS))
        cat_name_by_id = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}

        # Pick a capped, class-balanced set of image ids.
        chosen_img_ids: set[int] = set()
        for cat_id in cat_ids:
            img_ids = coco.getImgIds(catIds=[cat_id])
            for iid in img_ids[: self.limit_per_class]:
                chosen_img_ids.add(iid)

        print(f"COCO: downloading {len(chosen_img_ids)} images "
              f"(capped at {self.limit_per_class}/class)...")

        img_infos = coco.loadImgs(list(chosen_img_ids))
        img_dir = SRC_ROOT / "images"
        img_dir.mkdir(exist_ok=True)

        def fetch(info):
            dst = img_dir / info["file_name"]
            if not dst.exists():
                r = requests.get(info["coco_url"], timeout=15)
                if r.ok:
                    dst.write_bytes(r.content)
                else:
                    return None
            return info

        downloaded = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(fetch, info) for info in img_infos]
            for i, fut in enumerate(as_completed(futs)):
                res = fut.result()
                if res:
                    downloaded.append(res)
                if (i + 1) % 200 == 0:
                    print(f"  {i+1}/{len(img_infos)}")

        counts = Counter()
        for info in downloaded:
            img_path = img_dir / info["file_name"]
            ann_ids = coco.getAnnIds(imgIds=info["id"], catIds=cat_ids, iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            lines = []
            for a in anns:
                src_name = cat_name_by_id[a["category_id"]]
                unified = COCO_CATS[src_name]
                cls_id = CLASS_ID[unified]
                x, y, w, h = a["bbox"]
                lines.append(self.to_yolo_line(cls_id, x, y, x + w, y + h,
                                               info["width"], info["height"]))
                counts[unified] += 1
            self._write_pair(img_path, Path(info["file_name"]).stem, lines)

        return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=1500,
                   help="Max images per class to pull (kept small — COCO only "
                        "covers person/vehicle, which already work well).")
    args = p.parse_args()

    conv = COCOConverter(limit_per_class=args.limit)
    counts = conv.convert()
    print("\nCOCO boxes emitted:")
    for name, n in counts.items():
        print(f"  {name:12s} {n}")
    print(f"\n-> {OUT_ROOT}")

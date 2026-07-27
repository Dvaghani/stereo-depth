"""Roboflow project-gvpak/container-detection VERSION 1 -> unified taxonomy.

Version matters. v1 is real port photography with accurate boxes; v3 mixes in
satellite imagery and synthetic renders; v7 is mostly 3D renders whose boxes are
systematically misaligned — offset from the object, or a sliver at the frame
edge — and its class name "Healthy Container" reveals it as a damage-inspection
set rather than a detection one. Only v1 survived rendering its annotations.

These are ISO shipping containers, which the user's sites do not have. They are
included because a Baucontainer — the site office/storage cabin that does appear
in the footage — is the same object visually: a large corrugated steel box. The
transfer is by appearance, not by category.

The user's `container` class also covers skips (Absetzmulden). No usable skip
data was found: CDW-Seg annotates bin surfaces while photographing waste from
inside the bin, and the six Roboflow candidates were ports, renders or
disaster imagery. That half of the class remains unsupported.

Usage:
    python construction_yolo/datasets/converters/container_v1_converter.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.converters.base import DatasetConverter  # noqa: E402
from datasets.taxonomy import CLASS_ID  # noqa: E402

SRC_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources_raw/container_v1")
OUT_ROOT = Path("/run/media/dvaghani/Expansion/Yolo/sources/container_v1_yolo")

SRC_MAP = {0: "container"}
# 'valid' keeps its own prefix so the merge can route it to val
SPLIT_PREFIX = {"train": "train", "valid": "valid", "test": "train"}


class ContainerV1Converter(DatasetConverter):
    name = "container_v1"

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
    conv = ContainerV1Converter()
    counts = conv.convert()
    print("Container-v1 boxes emitted:")
    for name, n in counts.items():
        print("  %-12s %d" % (name, n))
    print("\n-> %s" % OUT_ROOT)

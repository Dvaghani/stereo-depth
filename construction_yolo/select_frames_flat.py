"""
Deduplicate frames that live in a single flat directory.

Groups frames by source-video prefix (filename without the trailing _NNNN)
and keeps a frame only if its perceptual hash differs enough from the last
kept frame of the same source.

Usage:
    python select_frames_flat.py --src FRAMES_DIR --dst OUT_DIR --min-hamming 14
"""
import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


def ahash(img, size=16):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return (g > g.mean()).flatten()


def source_key(name: str) -> str:
    # strip the trailing _NNNN(.jpg) frame index
    return re.sub(r"_\d+$", "", Path(name).stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--min-hamming", type=int, default=14)
    args = p.parse_args()

    dst = Path(args.dst); dst.mkdir(parents=True, exist_ok=True)
    frames = sorted(Path(args.src).glob("*.jpg"))

    last_hash = {}
    kept = 0
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        k = source_key(f.name)
        h = ahash(img)
        prev = last_hash.get(k)
        if prev is None or np.count_nonzero(h != prev) >= args.min_hamming:
            shutil.copy(f, dst / f.name)
            last_hash[k] = h
            kept += 1

    print(f"{len(frames)} -> {kept} frames in {dst}")


if __name__ == "__main__":
    main()

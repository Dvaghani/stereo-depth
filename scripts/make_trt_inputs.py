"""Produce raw float32 .bin inputs for `trtexec --loadInputs`, matching
compare_disparity.py's preprocessing exactly (imported, not duplicated) so
the desktop reference and the on-device engine see byte-identical pixels.

Usage:
    python scripts/make_trt_inputs.py \
        --left outputs/.../left.png --right outputs/.../right.png \
        --width 640 --height 480 --out-dir outputs/trt_inputs_480x640
"""
from __future__ import annotations

import argparse
from pathlib import Path

from compare_disparity import preprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, path in [("left", args.left), ("right", args.right)]:
        t = preprocess(path, args.width, args.height)
        arr = t.numpy()  # (1,3,H,W) float32, raw 0-255, no normalization
        bin_path = out / f"{name}.bin"
        arr.tofile(bin_path)
        print(f"{name}: {path} -> {bin_path}  shape {arr.shape}  "
              f"{bin_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()

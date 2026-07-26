"""Convert a trtexec --exportOutput JSON dump into a .npy disparity map.

Lets you validate an engine without pycuda on the device: preprocess on the
desktop, ship raw .bin inputs, run trtexec --loadInputs/--exportOutput, then
convert the result here and feed it to compare_disparity.py.

Usage:
    python scripts/trtexec_json_to_npy.py --json trt_out.json --out disp.npy
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="disparity",
                   help="output binding name to extract")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    args = p.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    # trtexec writes a list of {"name": ..., "dimensions": ..., "values": [...]}
    entry = None
    for item in data:
        if item.get("name") == args.name:
            entry = item
            break
    if entry is None:
        names = [item.get("name") for item in data]
        raise SystemExit(f"binding {args.name!r} not found; available: {names}")

    values = np.asarray(entry["values"], dtype=np.float32)
    expected = args.height * args.width
    if values.size != expected:
        raise SystemExit(
            f"expected {expected} values for {args.height}x{args.width}, "
            f"got {values.size} (dimensions field: {entry.get('dimensions')})")

    disp = values.reshape(args.height, args.width)
    np.save(args.out, disp)

    finite = np.isfinite(disp)
    print(f"disparity {disp.shape}  range [{disp.min():.3f}, {disp.max():.3f}]  "
          f"mean {disp.mean():.3f}")
    if not finite.all():
        print(f"WARNING: {(~finite).sum()} non-finite values — engine is broken")
    if disp.max() - disp.min() < 1e-3:
        print("WARNING: disparity is nearly constant — engine likely broken")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

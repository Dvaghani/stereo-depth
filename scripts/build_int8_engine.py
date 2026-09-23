"""
Build an INT8 TensorRT engine on the Jetson, with a real calibration pass.

Why this exists
----------------
`trtexec --int8 --calib=<file>` only READS an existing calibration cache; it
cannot produce one. Generating the cache needs an IInt8Calibrator fed with
representative data, which is what this does. Without it, TensorRT falls back
to guessing dynamic ranges and the resulting engine is fast but wrong -- the
exact failure mode the evaluation protocol exists to catch.

Runs on the Orin (TensorRT 10.x, tensor-name API). Generic over input tensor
names, so it works for RAFT (2 inputs) and YOLO-seg (1 input) unchanged.

Calibration data layouts understood:
  --calib-dir with left/ and right/ subdirs   -> RAFT-style stereo pairs
  --calib-dir with images (recursively found) -> single-input models

Preprocessing matches scripts/compare_disparity.py and trt10_bench.py exactly:
raw 0-255 float32 RGB, CHW, resized bilinear to the engine's input shape.

Usage:
    python3 build_int8_engine.py \
        --onnx onnx/raft_i4_480x640.onnx \
        --calib-dir int8_calib_raft \
        --out raft_i4_int8.trt --cache raft_i4_int8.cache
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401  -- creates the CUDA context
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_chw(path, h, w):
    """Raw 0-255 float32 RGB, CHW, contiguous. Must match the other scripts."""
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
    return np.ascontiguousarray(arr[None])


class Calibrator(trt.IInt8EntropyCalibrator2):
    """Feeds real data through the network so TensorRT can pick INT8 scales."""

    def __init__(self, batches, input_shapes, cache_path):
        super().__init__()
        self.batches = batches          # list of {tensor_name: ndarray}
        self.input_shapes = input_shapes
        self.cache_path = Path(cache_path)
        self.idx = 0
        self.device = {}
        for name, arr in batches[0].items():
            self.device[name] = cuda.mem_alloc(arr.nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= len(self.batches):
            return None
        batch = self.batches[self.idx]
        self.idx += 1
        if self.idx % 25 == 0 or self.idx == 1:
            print(f"  calibrating {self.idx}/{len(self.batches)}", flush=True)
        ptrs = []
        for name in names:
            if name not in batch:
                raise SystemExit(f"calibrator has no data for tensor {name!r}")
            cuda.memcpy_htod(self.device[name], batch[name])
            ptrs.append(int(self.device[name]))
        return ptrs

    def read_calibration_cache(self):
        if self.cache_path.exists():
            print(f"  reusing cache {self.cache_path}")
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)
        print(f"  wrote cache -> {self.cache_path}")


def collect_batches(calib_dir, input_names, shapes, limit):
    """Build per-tensor batches from the calibration directory."""
    calib_dir = Path(calib_dir)
    left_d, right_d = calib_dir / "left", calib_dir / "right"

    if len(input_names) == 2 and left_d.is_dir() and right_d.is_dir():
        lefts = sorted(left_d.glob("*.png"))[:limit]
        batches = []
        for lp in lefts:
            rp = right_d / lp.name
            if not rp.exists():
                continue
            ln, rn = input_names          # engine order, e.g. ("left","right")
            h, w = shapes[ln][2], shapes[ln][3]
            batches.append({ln: load_chw(lp, h, w), rn: load_chw(rp, h, w)})
        return batches

    if len(input_names) == 1:
        imgs = [p for p in calib_dir.rglob("*")
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")][:limit]
        name = input_names[0]
        h, w = shapes[name][2], shapes[name][3]
        return [{name: load_chw(p, h, w)} for p in imgs]

    raise SystemExit(
        f"cannot map {calib_dir} onto inputs {input_names}: expected left/ and "
        f"right/ subdirs for a 2-input model, or images for a 1-input model")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--calib-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=None)
    p.add_argument("--limit", type=int, default=100,
                   help="max calibration samples (100 is ample; more is slower)")
    p.add_argument("--fp16-fallback", action="store_true", default=True,
                   help="allow FP16 for layers INT8 cannot handle (recommended)")
    p.add_argument("--dla-core", type=int, default=None,
                   help="also target a DLA core, with GPU fallback")
    args = p.parse_args()

    cache = args.cache or (str(Path(args.out).with_suffix("")) + ".cache")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i))
            raise SystemExit("failed to parse ONNX")

    input_names, shapes = [], {}
    for i in range(network.num_inputs):
        t = network.get_input(i)
        input_names.append(t.name)
        shapes[t.name] = tuple(t.shape)
        print(f"input {t.name}: {tuple(t.shape)}")

    print(f"\ncollecting calibration data from {args.calib_dir} ...")
    batches = collect_batches(args.calib_dir, input_names, shapes, args.limit)
    if not batches:
        raise SystemExit("no calibration samples found")
    print(f"  {len(batches)} calibration sample(s)")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    cfg.set_flag(trt.BuilderFlag.INT8)
    if args.fp16_fallback:
        cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.int8_calibrator = Calibrator(batches, shapes, cache)
    if args.dla_core is not None:
        cfg.default_device_type = trt.DeviceType.DLA
        cfg.DLA_core = args.dla_core
        cfg.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        print(f"targeting DLA core {args.dla_core} with GPU fallback")

    print("\nbuilding INT8 engine (this runs the calibration pass; slow) ...")
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise SystemExit("engine build FAILED")
    Path(args.out).write_bytes(serialized)
    mb = Path(args.out).stat().st_size / 1e6
    print(f"\nsaved {args.out} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()

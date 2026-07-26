"""Run a TensorRT engine on a real stereo pair — for on-device validation.

Runs ON THE JETSON (Python 3.6, TensorRT 8.2.x, pycuda — all shipped with
JetPack; no PyTorch needed). Saves the disparity as .npy so it can be diffed
against the desktop PyTorch reference, which is what actually tells you whether
FP16 quantisation hurt anything. trtexec only ever feeds random noise, so it
cannot catch a numerically broken engine.

Preprocessing must match RAFT-Stereo's convention exactly: the model normalises
internally with 2*(x/255)-1, so the input tensor is RAW 0-255 float RGB in CHW
order. Feeding [0,1] silently produces garbage.

Usage (on the Jetson):
    python3 scripts/jetson_trt_infer.py \
        --engine raft_i7_fp16.trt \
        --left left.png --right right.png \
        --out disp_trt_fp16.npy --runs 20
"""
import argparse
import time

import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.autoinit  # noqa: F401  (initialises the CUDA context)
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def preprocess(path, width, height):
    """RAFT-Stereo expects raw 0-255 float RGB, CHW, batch 1."""
    img = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)          # HWC, 0-255
    arr = np.transpose(arr, (2, 0, 1))                # CHW
    return np.ascontiguousarray(arr[None])            # NCHW


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--out", default="disp_trt.npy")
    p.add_argument("--runs", type=int, default=20,
                   help="timed runs after one warm-up")
    args = p.parse_args()

    engine = load_engine(args.engine)
    context = engine.create_execution_context()

    # Map binding names -> index; the export named them left/right/disparity.
    bindings_meta = {}
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = tuple(engine.get_binding_shape(i))
        dtype = trt.nptype(engine.get_binding_dtype(i))
        bindings_meta[name] = (i, shape, dtype)
        kind = "input" if engine.binding_is_input(i) else "output"
        print("  binding %-10s %-6s %-20s %s" % (name, kind, shape, np.dtype(dtype).name))

    _, in_shape, _ = bindings_meta["left"]
    _, out_shape, out_dtype = bindings_meta["disparity"]
    height, width = in_shape[2], in_shape[3]

    left = preprocess(args.left, width, height)
    right = preprocess(args.right, width, height)
    print("input range: left [%.1f, %.1f]  (expect ~0-255)" % (left.min(), left.max()))

    out_host = np.empty(out_shape, dtype=out_dtype)
    d_left = cuda.mem_alloc(left.nbytes)
    d_right = cuda.mem_alloc(right.nbytes)
    d_out = cuda.mem_alloc(out_host.nbytes)

    bindings = [None] * engine.num_bindings
    bindings[bindings_meta["left"][0]] = int(d_left)
    bindings[bindings_meta["right"][0]] = int(d_right)
    bindings[bindings_meta["disparity"][0]] = int(d_out)

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_left, left, stream)
    cuda.memcpy_htod_async(d_right, right, stream)

    # warm-up (first call includes lazy init and would skew the timing)
    context.execute_async_v2(bindings, stream.handle)
    stream.synchronize()

    times = []
    for _ in range(args.runs):
        t0 = time.time()
        context.execute_async_v2(bindings, stream.handle)
        stream.synchronize()
        times.append((time.time() - t0) * 1000.0)

    cuda.memcpy_dtoh_async(out_host, d_out, stream)
    stream.synchronize()

    disp = out_host[0, 0]
    times = np.array(times)
    print("\nlatency  mean %.1f ms   min %.1f   max %.1f   (%.2f FPS)"
          % (times.mean(), times.min(), times.max(), 1000.0 / times.mean()))
    print("disparity  shape %s  range [%.3f, %.3f]  mean %.3f"
          % (disp.shape, disp.min(), disp.max(), disp.mean()))

    finite = np.isfinite(disp)
    if not finite.all():
        print("WARNING: %d non-finite values (NaN/Inf) — engine is broken"
              % (~finite).sum())
    if disp.max() - disp.min() < 1e-3:
        print("WARNING: disparity is nearly constant — engine likely broken")

    np.save(args.out, disp)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()

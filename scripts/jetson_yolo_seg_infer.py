"""Run the 11-class YOLO11s-seg TensorRT engine on a real image — for on-device
correctness validation, not just speed.

trtexec only ever benchmarks with random noise, so a 140ms number tells you
nothing about whether the engine produces sane detections. This decodes the
raw TRT outputs (box+class+mask-coefficients, plus mask prototypes) into real
boxes and masks, prints them, and saves an annotated image so a broken engine
is obvious rather than silently trusted.

Output layout (from export): output0 is (1, 4+nc+32, 8400) — 4 box coords, 11
class scores, 32 mask coefficients per anchor, transposed from Ultralytics'
usual (8400, C) so C is first. output1 is (1, 32, 160, 160) — the mask
prototypes; a detection's final mask is sigmoid(coeffs @ prototypes).

Preprocessing is 0-1 float RGB — NOT the raw 0-255 RAFT-Stereo expects. Mixing
the two conventions up is an easy mistake since both scripts sit side by side.

No letterboxing: images are resized directly to the square input, matching
every other script in this project (compare_disparity.py, jetson_trt_infer.py,
etc). Good enough for a sanity/timing check; do not read exact box coordinates
here as ground truth for accuracy claims — use the desktop `yolo val` numbers
for that.

Usage (on the Jetson):
    python3 scripts/jetson_yolo_seg_infer.py \
        --engine yolo11s_seg_11class_fp16.trt \
        --image left.jpg --out annotated.jpg --runs 20
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2

import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_seg_postprocess import decode, CLASS_NAMES  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(path):
    # trtexec registers the standard plugin creators for you; the Python API
    # does not, so a plugin layer deserializes as None without this.
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise SystemExit("Engine deserialization failed (version/arch mismatch?)")
    return engine


def preprocess(path, width, height):
    """Ultralytics YOLO expects 0-1 float RGB, CHW — unlike RAFT's raw 0-255."""
    img = cv2.imread(path)
    if img is None:
        raise SystemExit("could not read image: %s" % path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    arr = img.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None])


def find_bindings(engine):
    """-> (input_name, box_name, proto_name). Identified by rank, not name,
    since export names may vary: the 4D output is the mask-prototype tensor,
    the 3D output is boxes+classes+coeffs, the input has 3 channels."""
    in_name = box_name = proto_name = None
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = tuple(engine.get_binding_shape(i))
        if engine.binding_is_input(i):
            in_name = name
        elif len(shape) == 4:
            proto_name = name
        else:
            box_name = name
    if not all([in_name, box_name, proto_name]):
        raise SystemExit("could not identify input/box/proto bindings")
    return in_name, box_name, proto_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="yolo_seg_annotated.jpg")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--runs", type=int, default=20)
    args = p.parse_args()

    engine = load_engine(args.engine)
    context = engine.create_execution_context()

    bindings_meta = {}
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = tuple(engine.get_binding_shape(i))
        dtype = trt.nptype(engine.get_binding_dtype(i))
        bindings_meta[name] = (i, shape, dtype)
        kind = "input" if engine.binding_is_input(i) else "output"
        print("  binding %-10s %-6s %-20s %s" % (name, kind, shape, np.dtype(dtype).name))

    in_name, box_name, proto_name = find_bindings(engine)
    _, in_shape, _ = bindings_meta[in_name]
    height, width = in_shape[2], in_shape[3]

    img = preprocess(args.image, width, height)
    print("input range: [%.3f, %.3f]  (expect ~0-1)" % (img.min(), img.max()))

    box_host = np.empty(bindings_meta[box_name][1], dtype=bindings_meta[box_name][2])
    proto_host = np.empty(bindings_meta[proto_name][1], dtype=bindings_meta[proto_name][2])

    d_in = cuda.mem_alloc(img.nbytes)
    d_box = cuda.mem_alloc(box_host.nbytes)
    d_proto = cuda.mem_alloc(proto_host.nbytes)

    bindings = [None] * engine.num_bindings
    bindings[bindings_meta[in_name][0]] = int(d_in)
    bindings[bindings_meta[box_name][0]] = int(d_box)
    bindings[bindings_meta[proto_name][0]] = int(d_proto)

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_in, img, stream)
    context.execute_async_v2(bindings, stream.handle)   # warm-up
    stream.synchronize()

    times = []
    for _ in range(args.runs):
        t0 = time.time()
        context.execute_async_v2(bindings, stream.handle)
        stream.synchronize()
        times.append((time.time() - t0) * 1000.0)

    cuda.memcpy_dtoh_async(box_host, d_box, stream)
    cuda.memcpy_dtoh_async(proto_host, d_proto, stream)
    stream.synchronize()

    times = np.array(times)
    print("\nlatency  mean %.1f ms   min %.1f   max %.1f   (%.2f FPS)"
          % (times.mean(), times.min(), times.max(), 1000.0 / times.mean()))

    detections = decode(box_host[0], proto_host[0], conf_thres=args.conf, img_size=width)
    print("\n%d detections above conf=%.2f:" % (len(detections), args.conf))
    for d in sorted(detections, key=lambda d: -d["conf"]):
        print("  %-16s conf=%.2f  box=[%.0f,%.0f,%.0f,%.0f]"
              % (CLASS_NAMES[d["class_id"]], d["conf"], *d["box"]))

    if not detections:
        print("\nWARNING: zero detections at conf=%.2f — try --conf 0.05 to check "
              "whether the engine sees anything at all before assuming it's broken."
              % args.conf)

    vis = cv2.resize(cv2.imread(args.image), (width, height))
    rng = np.random.RandomState(0)
    colors = rng.randint(60, 255, size=(len(CLASS_NAMES), 3))
    for d in detections:
        color = tuple(int(v) for v in colors[d["class_id"]])
        overlay = vis.copy()
        overlay[d["mask"]] = color
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        x0, y0, x1, y1 = [int(v) for v in d["box"]]
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cv2.putText(vis, "%s %.2f" % (CLASS_NAMES[d["class_id"]], d["conf"]),
                    (x0, max(12, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(args.out, vis)
    print("\nannotated -> %s  (copy back and eyeball it — masks/boxes should land "
          "on real objects, not float in empty space)" % args.out)


if __name__ == "__main__":
    main()

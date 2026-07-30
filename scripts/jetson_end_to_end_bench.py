"""End-to-end pipeline benchmark on the Jetson: Brio capture -> rectify ->
RAFT-Stereo depth -> YOLO11-seg detection -> fusion.

Every number published so far (105ms YOLO, 686ms RAFT, 791ms combined) is
GPU-inference-only, timed on pre-loaded static images. None of it includes USB
capture, rectification, resizing, or fusing depth with detections — the
docs/jetson_deployment_results.md write-up explicitly estimated those at
100-200ms rather than measuring them. This closes that gap with real cameras
and real timing, stage by stage.

Two RUN MODES:
  --probe-cams          list detected /dev/video* devices with resolution, to
                         figure out which index is left/right before running
                         the real thing
  --left-cam/--right-cam  the real deployment path: capture N stereo pairs,
                         time capture separately from compute (grab() on both
                         before retrieve() on either, to minimise the left/right
                         time skew), then run the full compute pipeline
  --left-img/--right-img  fallback for a dry run without cameras attached, or
                         to sanity-check the pipeline logic against a known pair

Depth/box fusion follows scripts/live_detect.py's convention but simplified:
median and min depth per detected box, not the full radial range map — enough
for a timing and sanity check, not a replacement for the desktop pipeline.

No letterboxing anywhere (matches every other script in this project) — good
enough for timing and a sanity check, not for reading off exact box coordinates
as ground truth.

Usage:
    python3 scripts/jetson_end_to_end_bench.py --probe-cams

    python3 scripts/jetson_end_to_end_bench.py \
        --left-cam 0 --right-cam 2 \
        --calib stereo_calib.npz \
        --raft-engine raft_i4_fp16.trt \
        --yolo-engine yolo11s_seg_11class_fp16.trt \
        --frames 20 --out annotated.jpg
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
from yolo_seg_postprocess import decode as decode_yolo_seg, CLASS_NAMES  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ── engine loading (shared pattern with jetson_trt_infer.py) ────────────────

def load_engine(path):
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise SystemExit("Engine deserialization failed for %s "
                         "(version/arch mismatch?)" % path)
    return engine


class RaftEngine:
    def __init__(self, path):
        self.engine = load_engine(path)
        self.ctx = self.engine.create_execution_context()
        meta = {}
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            meta[name] = (i, shape, dtype)
        self.meta = meta
        _, in_shape, _ = meta["left"]
        self.height, self.width = in_shape[2], in_shape[3]
        out_idx, out_shape, out_dtype = meta["disparity"]
        self.out_host = np.empty(out_shape, dtype=out_dtype)
        left_nbytes = int(np.prod(meta["left"][1])) * 4
        self.d_left = cuda.mem_alloc(left_nbytes)
        self.d_right = cuda.mem_alloc(left_nbytes)
        self.d_out = cuda.mem_alloc(self.out_host.nbytes)
        self.bindings = [None] * self.engine.num_bindings
        self.bindings[meta["left"][0]] = int(self.d_left)
        self.bindings[meta["right"][0]] = int(self.d_right)
        self.bindings[out_idx] = int(self.d_out)
        self.stream = cuda.Stream()

    def preprocess(self, bgr):
        """RAFT-Stereo: raw 0-255 float RGB, CHW."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        arr = rgb.astype(np.float32)
        return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None])

    def infer(self, left_bgr, right_bgr):
        left = self.preprocess(left_bgr)
        right = self.preprocess(right_bgr)
        cuda.memcpy_htod_async(self.d_left, left, self.stream)
        cuda.memcpy_htod_async(self.d_right, right, self.stream)
        self.ctx.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.out_host, self.d_out, self.stream)
        self.stream.synchronize()
        return self.out_host[0, 0].copy()


class YoloSegEngine:
    def __init__(self, path, conf_thres=0.25, iou_thres=0.45):
        self.engine = load_engine(path)
        self.ctx = self.engine.create_execution_context()
        self.conf_thres, self.iou_thres = conf_thres, iou_thres
        meta = {}
        in_name = box_name = proto_name = None
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            meta[name] = (i, shape, dtype)
            if self.engine.binding_is_input(i):
                in_name = name
            elif len(shape) == 4:
                proto_name = name
            else:
                box_name = name
        self.meta, self.in_name, self.box_name, self.proto_name = (
            meta, in_name, box_name, proto_name)
        _, in_shape, _ = meta[in_name]
        self.height, self.width = in_shape[2], in_shape[3]

        self.box_host = np.empty(meta[box_name][1], dtype=meta[box_name][2])
        self.proto_host = np.empty(meta[proto_name][1], dtype=meta[proto_name][2])
        in_nbytes = int(np.prod(meta[in_name][1])) * 4
        self.d_in = cuda.mem_alloc(in_nbytes)
        self.d_box = cuda.mem_alloc(self.box_host.nbytes)
        self.d_proto = cuda.mem_alloc(self.proto_host.nbytes)
        self.bindings = [None] * self.engine.num_bindings
        self.bindings[meta[in_name][0]] = int(self.d_in)
        self.bindings[meta[box_name][0]] = int(self.d_box)
        self.bindings[meta[proto_name][0]] = int(self.d_proto)
        self.stream = cuda.Stream()

    def preprocess(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        arr = rgb.astype(np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None])

    def infer(self, bgr):
        img = self.preprocess(bgr)
        cuda.memcpy_htod_async(self.d_in, img, self.stream)
        self.ctx.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.box_host, self.d_box, self.stream)
        cuda.memcpy_dtoh_async(self.proto_host, self.d_proto, self.stream)
        self.stream.synchronize()
        return decode_yolo_seg(self.box_host[0], self.proto_host[0],
                               conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                               img_size=self.width)


# ── camera handling ──────────────────────────────────────────────────────────

def probe_cameras(max_index=10):
    print("Probing /dev/video0 .. /dev/video%d ...\n" % max_index)
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_s = "".join(chr((fourcc >> 8 * k) & 0xFF) for k in range(4))
        print("  /dev/video%-2d  read_ok=%-5s  %dx%d  fourcc=%s"
              % (i, ok, w, h, fourcc_s))
        if ok:
            found.append(i)
        cap.release()
    print("\nReadable indices: %s" % found)
    print("Brios usually expose 2 nodes each (video+metadata) at consecutive "
          "indices — the one that reads a real frame above is the one to use.")


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit("could not open camera index %d" % index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # let auto-exposure/focus settle and flush the initial low-quality frames
    for _ in range(5):
        cap.read()
    return cap


def capture_pair(cap_l, cap_r):
    """grab() both before retrieve() on either, to minimise left/right skew —
    retrieve() decodes the MJPEG frame, which is the slow part."""
    ok_l = cap_l.grab()
    ok_r = cap_r.grab()
    if not (ok_l and ok_r):
        return None, None
    ok_l, left = cap_l.retrieve()
    ok_r, right = cap_r.retrieve()
    if not (ok_l and ok_r):
        return None, None
    return left, right


# ── fusion ───────────────────────────────────────────────────────────────────

def scalar(x):
    return float(np.asarray(x).flat[0])


def fuse_depth(disparity, detections, focal_px_at_disp_res, baseline_m):
    """Per detection: median/min depth (m) over valid disparity pixels in its
    box, mapped from the YOLO input resolution into the (possibly different)
    RAFT/disparity resolution."""
    dh, dw = disparity.shape
    for det in detections:
        # det['box'] is in YOLO input pixel space; rescale into disparity space
        yb = det["_yolo_size"]
        sx, sy = dw / float(yb[0]), dh / float(yb[1])
        x0, y0, x1, y1 = det["box"]
        x0, x1 = int(x0 * sx), int(x1 * sx)
        y0, y1 = int(y0 * sy), int(y1 * sy)
        x0, x1 = max(0, x0), min(dw, x1)
        y0, y1 = max(0, y0), min(dh, y1)
        if x1 <= x0 or y1 <= y0:
            det["depth_m"] = None
            continue
        roi = disparity[y0:y1, x0:x1]
        valid = roi[roi > 0.5]
        if valid.size == 0:
            det["depth_m"] = None
            continue
        depth = focal_px_at_disp_res * baseline_m / valid
        det["depth_m"] = (float(np.median(depth)), float(np.min(depth)))
    return detections


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-cams", action="store_true")
    p.add_argument("--left-cam", type=int, default=None)
    p.add_argument("--right-cam", type=int, default=None)
    p.add_argument("--left-img", default=None, help="dry-run fallback, no cameras")
    p.add_argument("--right-img", default=None)
    p.add_argument("--calib", required=False, help="stereo_calib.npz")
    p.add_argument("--raft-engine", required=False)
    p.add_argument("--yolo-engine", required=False)
    p.add_argument("--frames", type=int, default=20,
                   help="stereo pairs to capture for the capture-latency stat")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default="e2e_annotated.jpg")
    args = p.parse_args()

    if args.probe_cams:
        probe_cameras()
        return

    if not args.calib or not args.raft_engine or not args.yolo_engine:
        raise SystemExit("--calib, --raft-engine and --yolo-engine are required "
                         "unless using --probe-cams")

    calib = np.load(args.calib)
    map1L, map2L = calib["map1L"], calib["map2L"]
    map1R, map2R = calib["map1R"], calib["map2R"]
    cap_w, cap_h = (int(v) for v in calib["image_size"])
    baseline_m = scalar(calib["baseline_mm"]) / 1000.0
    focal_px_native = scalar(calib["focal_px"])
    print("calibration: %dx%d capture, baseline %.1f mm, focal %.1f px (native)"
          % (cap_w, cap_h, baseline_m * 1000, focal_px_native))

    raft = RaftEngine(args.raft_engine)
    yolo = YoloSegEngine(args.yolo_engine, conf_thres=args.conf)
    print("RAFT engine input:  %dx%d" % (raft.width, raft.height))
    print("YOLO engine input:  %dx%d" % (yolo.width, yolo.height))

    timings = {"capture": [], "rectify": [], "raft": [], "yolo": [], "fuse": []}

    # ── capture (or load static images) ─────────────────────────────────────
    frames = []
    if args.left_img and args.right_img:
        print("\nDRY RUN: using static images, not live cameras")
        left = cv2.imread(args.left_img)
        right = cv2.imread(args.right_img)
        if left is None or right is None:
            raise SystemExit("could not read --left-img/--right-img")
        frames = [(left, right)] * max(args.frames, 1)
        timings["capture"] = [0.0]   # not meaningful in dry-run mode
    else:
        if args.left_cam is None or args.right_cam is None:
            raise SystemExit("--left-cam and --right-cam are required "
                             "(or use --left-img/--right-img for a dry run, "
                             "or --probe-cams to find indices)")
        cap_l = open_camera(args.left_cam, cap_w, cap_h)
        cap_r = open_camera(args.right_cam, cap_w, cap_h)
        print("\ncapturing %d stereo pairs..." % args.frames)
        for _ in range(args.frames):
            t0 = time.time()
            left, right = capture_pair(cap_l, cap_r)
            timings["capture"].append((time.time() - t0) * 1000.0)
            if left is None:
                continue
            frames.append((left, right))
        cap_l.release()
        cap_r.release()
        if not frames:
            raise SystemExit("captured zero valid frames — check camera indices "
                             "and USB bandwidth (try MJPEG, separate controllers)")

    # ── warm-up: each engine's first call pays a one-time cold-start cost
    # (CUDA context lazy-init, algorithm caching) that has nothing to do with
    # steady-state deployment performance. jetson_trt_infer.py and
    # jetson_yolo_seg_infer.py both do an untimed call before their timed loop;
    # this pipeline needs the same discipline or that cost silently pollutes
    # frame 1 of the "real" measurement — exactly what happened before this
    # fix landed (a ~1.6s outlier, reproducible to <2ms across separate runs,
    # which is the signature of a deterministic one-time cost, not noise).
    print("\nwarming up both engines (untimed)...")
    warm_left, warm_right = frames[0]
    warm_rectL = cv2.remap(warm_left, map1L, map2L, cv2.INTER_LINEAR)
    warm_rectR = cv2.remap(warm_right, map1R, map2R, cv2.INTER_LINEAR)
    raft.infer(warm_rectL, warm_rectR)
    yolo.infer(warm_rectL)

    # ── compute pipeline, timed per stage over the captured frames ──────────
    last_detections, last_disparity, last_left_rect = None, None, None
    for left, right in frames:
        t0 = time.time()
        rectL = cv2.remap(left, map1L, map2L, cv2.INTER_LINEAR)
        rectR = cv2.remap(right, map1R, map2R, cv2.INTER_LINEAR)
        timings["rectify"].append((time.time() - t0) * 1000.0)

        t0 = time.time()
        disparity = raft.infer(rectL, rectR)
        timings["raft"].append((time.time() - t0) * 1000.0)

        t0 = time.time()
        detections = yolo.infer(rectL)
        for d in detections:
            d["_yolo_size"] = (yolo.width, yolo.height)
        timings["yolo"].append((time.time() - t0) * 1000.0)

        t0 = time.time()
        focal_at_disp_res = focal_px_native * (raft.width / float(cap_w))
        detections = fuse_depth(disparity, detections, focal_at_disp_res, baseline_m)
        timings["fuse"].append((time.time() - t0) * 1000.0)

        last_detections, last_disparity, last_left_rect = detections, disparity, rectL

    # ── report ───────────────────────────────────────────────────────────────
    print("\n=== per-stage latency (mean over %d frames) ===" % len(frames))
    total = 0.0
    for stage in ("capture", "rectify", "raft", "yolo", "fuse"):
        vals = np.array(timings[stage])
        if len(vals) == 0:
            continue
        m = vals.mean()
        total += m
        print("  %-10s %7.1f ms   (min %.1f, max %.1f)" % (stage, m, vals.min(), vals.max()))
    print("  %-10s %7.1f ms   -> %.2f FPS" % ("TOTAL", total, 1000.0 / total if total else 0))

    print("\n=== last frame: %d detections ===" % len(last_detections or []))
    for d in sorted(last_detections or [], key=lambda d: -d["conf"]):
        depth_str = ("median %.2fm / near %.2fm" % d["depth_m"]
                    if d["depth_m"] else "no valid disparity in box")
        print("  %-16s conf=%.2f  %s" % (CLASS_NAMES[d["class_id"]], d["conf"], depth_str))

    if last_left_rect is not None:
        vis = cv2.resize(last_left_rect, (yolo.width, yolo.height))
        rng = np.random.RandomState(0)
        colors = rng.randint(60, 255, size=(len(CLASS_NAMES), 3))
        for d in (last_detections or []):
            color = tuple(int(v) for v in colors[d["class_id"]])
            overlay = vis.copy()
            overlay[d["mask"]] = color
            vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
            x0, y0, x1, y1 = [int(v) for v in d["box"]]
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            label = CLASS_NAMES[d["class_id"]]
            if d["depth_m"]:
                label += " %.1fm" % d["depth_m"][0]
            cv2.putText(vis, label, (x0, max(12, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(args.out, vis)
        print("\nannotated -> %s" % args.out)


if __name__ == "__main__":
    main()

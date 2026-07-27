"""Sustained-load benchmark for a TensorRT engine on Jetson: latency + thermals + power.

Runs ON THE JETSON (Python 3.6, TensorRT, pycuda). Loops inference for a fixed
duration while sampling temperatures and the onboard INA3221 power rails, then
writes a CSV and prints a summary.

Answers three questions a short trtexec run cannot:
  - does latency degrade once the board heats up (thermal throttling)?
  - what does it actually draw, in watts and joules per frame?
  - how does that change between nvpmodel MAXN and 5W?

Run once per power mode:
    sudo nvpmodel -m 0 && sudo jetson_clocks
    python3 scripts/jetson_sustained_bench.py --engine raft_i7_fp16.trt \
        --left left.png --right right.png --minutes 20 --tag maxn

    sudo nvpmodel -m 1 && sudo reboot     # then, after reboot:
    python3 scripts/jetson_sustained_bench.py --engine raft_i7_fp16.trt \
        --left left.png --right right.png --minutes 20 --tag 5w
"""
import argparse
import glob
import os
import time

import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def discover_thermal_zones():
    """-> list of (label, path_to_temp_file). Values are millidegrees C."""
    zones = []
    for zone in sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*")):
        try:
            with open(os.path.join(zone, "type")) as f:
                label = f.read().strip()
        except IOError:
            continue
        temp_path = os.path.join(zone, "temp")
        if os.path.exists(temp_path):
            zones.append((label, temp_path))
    return zones


def discover_power_rails():
    """-> list of (label, path). INA3221 rails, values in milliwatts.

    Path layout differs across L4T releases, so glob broadly rather than
    hardcoding the i2c address."""
    rails = []
    patterns = [
        "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input",
        "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/power*_input",
    ]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            label = None
            # try the matching *_label file, else fall back to the filename
            for suffix in ("_label", "_input"):
                cand = path.replace("_input", suffix)
                if suffix == "_label" and os.path.exists(cand):
                    try:
                        with open(cand) as f:
                            label = f.read().strip()
                    except IOError:
                        pass
            rails.append((label or os.path.basename(path), path))
        if rails:
            break
    return rails


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (IOError, ValueError):
        return None


def load_engine(path):
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise SystemExit("Engine deserialization failed (version/arch mismatch?)")
    return engine


def preprocess(path, width, height):
    img = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--sample-every", type=float, default=5.0,
                   help="seconds between thermal/power samples")
    p.add_argument("--tag", default="run", help="label for the output CSV")
    args = p.parse_args()

    zones = discover_thermal_zones()
    rails = discover_power_rails()
    print("thermal zones: %s" % ", ".join(z[0] for z in zones) if zones else
          "thermal zones: NONE FOUND")
    print("power rails:   %s" % ", ".join(r[0] for r in rails) if rails else
          "power rails:   NONE FOUND (power logging disabled)")

    engine = load_engine(args.engine)
    context = engine.create_execution_context()

    meta = {}
    for i in range(engine.num_bindings):
        meta[engine.get_binding_name(i)] = (i, tuple(engine.get_binding_shape(i)),
                                            trt.nptype(engine.get_binding_dtype(i)))
    in_shape = meta["left"][1]
    out_shape, out_dtype = meta["disparity"][1], meta["disparity"][2]
    height, width = in_shape[2], in_shape[3]

    left = preprocess(args.left, width, height)
    right = preprocess(args.right, width, height)
    out_host = np.empty(out_shape, dtype=out_dtype)

    d_left = cuda.mem_alloc(left.nbytes)
    d_right = cuda.mem_alloc(right.nbytes)
    d_out = cuda.mem_alloc(out_host.nbytes)
    bindings = [None] * engine.num_bindings
    bindings[meta["left"][0]] = int(d_left)
    bindings[meta["right"][0]] = int(d_right)
    bindings[meta["disparity"][0]] = int(d_out)

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_left, left, stream)
    cuda.memcpy_htod_async(d_right, right, stream)
    context.execute_async_v2(bindings, stream.handle)   # warm-up
    stream.synchronize()

    csv_path = "bench_%s.csv" % args.tag
    csv = open(csv_path, "w")
    header = ["elapsed_s", "frame", "latency_ms"]
    header += ["temp_" + z[0] for z in zones]
    header += ["mW_" + r[0] for r in rails]
    csv.write(",".join(header) + "\n")

    print("\nrunning for %.1f min — logging to %s\n" % (args.minutes, csv_path))
    deadline = time.time() + args.minutes * 60.0
    next_sample = 0.0
    start = time.time()
    frame = 0
    latencies = []

    while time.time() < deadline:
        t0 = time.time()
        context.execute_async_v2(bindings, stream.handle)
        stream.synchronize()
        lat = (time.time() - t0) * 1000.0
        latencies.append(lat)
        frame += 1
        elapsed = time.time() - start

        if elapsed >= next_sample:
            temps = [read_int(z[1]) for z in zones]
            powers = [read_int(r[1]) for r in rails]
            row = ["%.1f" % elapsed, str(frame), "%.2f" % lat]
            row += ["" if t is None else "%.1f" % (t / 1000.0) for t in temps]
            row += ["" if w is None else str(w) for w in powers]
            csv.write(",".join(row) + "\n")
            csv.flush()

            hot = max([t for t in temps if t is not None] or [0]) / 1000.0
            tot = powers[0] if powers and powers[0] is not None else None
            print("  %5.0fs  frame %5d  %6.1f ms  max %.1fC%s"
                  % (elapsed, frame, lat, hot,
                     "  %.2f W" % (tot / 1000.0) if tot else ""))
            next_sample = elapsed + args.sample_every

    csv.close()
    lat = np.array(latencies)
    n = len(lat)
    first, last = lat[: max(n // 10, 1)], lat[-max(n // 10, 1):]

    print("\n=== summary (%s) ===" % args.tag)
    print("  frames            %d over %.1f min" % (n, args.minutes))
    print("  latency mean      %.1f ms  (%.2f FPS)" % (lat.mean(), 1000.0 / lat.mean()))
    print("  latency min/max   %.1f / %.1f ms" % (lat.min(), lat.max()))
    print("  first 10%% mean    %.1f ms" % first.mean())
    print("  last  10%% mean    %.1f ms" % last.mean())
    drift = 100.0 * (last.mean() - first.mean()) / first.mean()
    print("  drift             %+.1f %%  %s"
          % (drift, "(throttling)" if drift > 5 else "(stable)"))
    print("  csv               %s" % csv_path)


if __name__ == "__main__":
    main()

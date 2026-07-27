"""Benchmark classical stereo (VPI and OpenCV SGBM) against learned RAFT-Stereo.

Runs ON THE JETSON. Answers the "why not just use classical stereo, it's 20x
faster?" question with measurements instead of assertion: classical matchers
are cheap but fail on textureless regions and thin structures — which is
exactly the case (cables against sky) that motivated using RAFT at all.

Saves each disparity map as .npy so accuracy can be compared on the desktop
against the RAFT reference with compare_disparity.py.

Note on disparity range: the reference pair produces 29-122 px disparity at
480x640, so a matcher limited to 64 will clip badly — that clipping is itself
a finding worth reporting, not a bug to hide.

Usage:
    python3 scripts/jetson_classical_stereo_bench.py \
        --left left.png --right right.png --width 640 --height 480 --maxdisp 128
"""
import argparse
import time

import numpy as np
from PIL import Image


def load_gray(path, width, height):
    img = Image.open(path).convert("L").resize((width, height), Image.BILINEAR)
    return np.asarray(img).astype(np.uint8)


def bench(fn, runs):
    fn()                                  # warm-up
    times = []
    for _ in range(runs):
        t0 = time.time()
        out = fn()
        times.append((time.time() - t0) * 1000.0)
    return out, np.array(times)


def report(name, disp, times, out_path):
    valid = disp[np.isfinite(disp) & (disp > 0)]
    print("\n=== %s ===" % name)
    print("  latency mean %.1f ms  min %.1f  max %.1f  (%.1f FPS)"
          % (times.mean(), times.min(), times.max(), 1000.0 / times.mean()))
    if valid.size:
        print("  disparity valid %.1f %% of pixels, range [%.1f, %.1f], mean %.1f"
              % (100.0 * valid.size / disp.size, valid.min(), valid.max(), valid.mean()))
    else:
        print("  WARNING: no valid disparity produced")
    np.save(out_path, disp.astype(np.float32))
    print("  saved -> %s" % out_path)


def run_vpi(left, right, maxdisp, runs):
    try:
        import vpi
    except ImportError:
        print("\n=== VPI ===\n  not available (no `vpi` module) — skipping")
        return
    print("\nVPI version: %s" % getattr(vpi, "__version__", "unknown"))

    # stereodisp wants 16-bit grayscale; formats/backends vary by VPI release,
    # so try CUDA then fall back to CPU.
    for backend_name in ("CUDA", "CPU"):
        backend = getattr(vpi.Backend, backend_name, None)
        if backend is None:
            continue
        try:
            with vpi.Backend.CUDA:
                vl = vpi.asimage(left).convert(vpi.Format.Y16_ER)
                vr = vpi.asimage(right).convert(vpi.Format.Y16_ER)

            def go():
                with backend:
                    d = vpi.stereodisp(vl, vr, maxdisp=maxdisp)
                    d = d.convert(vpi.Format.U16, scale=1.0)
                with d.rlock_cpu() as arr:
                    # VPI returns Q10.5 fixed point -> divide by 32 for pixels
                    return np.array(arr).astype(np.float32) / 32.0

            disp, times = bench(go, runs)
            report("VPI stereodisp (%s, maxdisp=%d)" % (backend_name, maxdisp),
                   disp, times, "disp_vpi_%s.npy" % backend_name.lower())
            return
        except Exception as exc:
            print("  VPI %s backend failed: %s" % (backend_name, exc))
    print("  all VPI backends failed — reporting OpenCV SGBM only")


def run_sgbm(left, right, maxdisp, runs):
    import cv2
    # numDisparities must be a multiple of 16
    numdisp = int(np.ceil(maxdisp / 16.0) * 16)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numdisp,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=1,
    )

    def go():
        # SGBM returns fixed-point disparity scaled by 16
        return matcher.compute(left, right).astype(np.float32) / 16.0

    disp, times = bench(go, runs)
    report("OpenCV StereoSGBM (CPU, numDisparities=%d)" % numdisp,
           disp, times, "disp_sgbm.npy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--maxdisp", type=int, default=128,
                   help="reference pair reaches 122 px at 640x480; 64 will clip")
    p.add_argument("--runs", type=int, default=20)
    args = p.parse_args()

    left = load_gray(args.left, args.width, args.height)
    right = load_gray(args.right, args.width, args.height)
    print("input %dx%d grayscale, maxdisp=%d" % (args.width, args.height, args.maxdisp))

    run_vpi(left, right, args.maxdisp, args.runs)
    run_sgbm(left, right, args.maxdisp, args.runs)

    print("\nCompare accuracy on the desktop, e.g.:")
    print("  python scripts/compare_disparity.py --ckpt <raft.pth> \\")
    print("      --left <left.png> --right <right.png> --trt disp_sgbm.npy --iters 7")


if __name__ == "__main__":
    main()

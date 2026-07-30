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


VPI_MAX_DISPARITY = 64   # hard limit in VPI 1.x, regardless of backend


def _as_2d(obj):
    """Accept only a genuine 2D pixel buffer.

    VPI 1.0's rlock() yields a lock object rather than the data, and
    np.array() silently wraps that as a 0-d object array — which then reads as
    a single NaN instead of failing. Validate rather than trust."""
    arr = np.asarray(obj)
    if arr.ndim >= 2 and arr.size > 1:
        return arr
    raise TypeError("not a pixel buffer (ndim=%d, size=%d)" % (arr.ndim, arr.size))


def _vpi_to_numpy(img):
    """Read a VPI image back to numpy. The accessor moved across releases:
    1.0 exposes .cpu(); later versions use .rlock_cpu()/.rlock()."""
    errors = []
    fn = getattr(img, "cpu", None)
    if fn is not None:
        try:
            return _as_2d(fn())
        except Exception as exc:
            errors.append("cpu(): %s" % exc)

    for attr in ("rlock_cpu", "rlock"):
        fn = getattr(img, attr, None)
        if fn is None:
            continue
        try:
            with fn() as data:
                return _as_2d(data)
        except Exception as exc:
            errors.append("%s(): %s" % (attr, exc))

    raise RuntimeError("no working readback on vpi.Image — tried %s"
                       % "; ".join(errors))


def run_vpi(left, right, maxdisp, runs):
    try:
        import vpi
    except ImportError:
        print("\n=== VPI ===\n  not available (no `vpi` module) — skipping")
        return

    if maxdisp > VPI_MAX_DISPARITY:
        print("\n  NOTE: VPI caps maximum disparity at %d; requested %d. Clamping."
              "\n  The scene reaches ~122 px, so the near field WILL be clipped —"
              "\n  that limitation is itself a result worth reporting."
              % (VPI_MAX_DISPARITY, maxdisp))
        maxdisp = VPI_MAX_DISPARITY

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
                # stereodisp returns Q10.5 fixed point -> /32 for pixel units
                return _vpi_to_numpy(d).astype(np.float32) / 32.0

            disp, times = bench(go, runs)
            report("VPI stereodisp (%s, maxdisp=%d)" % (backend_name, maxdisp),
                   disp, times, "disp_vpi_%s.npy" % backend_name.lower())
            return
        except Exception as exc:
            print("  VPI %s backend failed: %s" % (backend_name, exc))
    print("  all VPI backends failed")


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
    p.add_argument("--only", choices=["vpi", "sgbm"], default=None,
                   help="run just one matcher — VPI can destabilise the process, "
                        "so run them separately if the other segfaults")
    args = p.parse_args()

    left = load_gray(args.left, args.width, args.height)
    right = load_gray(args.right, args.width, args.height)
    print("input %dx%d grayscale, maxdisp=%d" % (args.width, args.height, args.maxdisp))

    if args.only != "sgbm":
        run_vpi(left, right, args.maxdisp, args.runs)
    if args.only != "vpi":
        run_sgbm(left, right, args.maxdisp, args.runs)

    print("\nCompare accuracy on the desktop, e.g.:")
    print("  python scripts/compare_disparity.py --ckpt <raft.pth> \\")
    print("      --left <left.png> --right <right.png> --trt disp_sgbm.npy --iters 7")


if __name__ == "__main__":
    main()

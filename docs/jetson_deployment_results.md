# Jetson Nano Deployment: RAFT-Stereo + YOLO11

Measured results for running the depth + detection pipeline on a Jetson Nano
(4 GB), July 2026. All figures are measured on-device unless stated otherwise.

**Hardware:** NVIDIA Jetson Nano 4 GB — Tegra X1, Compute Capability 5.3,
**1 SM / 128 CUDA cores**. JetPack 4.6.1, TensorRT 8.2.1, CUDA 10.2.

---

## 1. Summary

The pipeline **runs correctly** on the Nano but **not fast enough for reactive
obstacle avoidance**. Detection is comfortably real-time; dense stereo is not,
and dense stereo dominates the budget.

| stage | GPU inference only | real end-to-end (§9) |
|---|---:|---:|
| YOLO11s-seg, 11 classes (640×640) | 140.8 ms | 177.0 ms |
| RAFT-Stereo depth (480×640, 7 iters) | 686 ms | 694.5 ms |
| capture (2× Brio, USB) | — | 82.6 ms |
| rectify | — | 52.8 ms |
| depth/detection fusion | — | 12.2 ms |
| **TOTAL** | **826.8 ms → 1.21 FPS** | **1019.0 ms → 0.98 FPS** |

The GPU-only figure is what every earlier section on this page reports —
useful for isolating model cost, but it undercounts real deployment by **~23%**
once capture, rectification, and pre/postprocessing are included. §9 has the
full breakdown and how it was measured.

---

## 2. Correctness was verified, not assumed

`trtexec` benchmarks with random input, so it measures speed without
establishing that the engine computes anything meaningful. The exported engine
was therefore run on a real rectified stereo pair and compared against the
desktop PyTorch FP32 reference.

| metric | value |
|---|---:|
| mean absolute disparity error | **0.0345 px** |
| median | 0.0175 px |
| 95th percentile | 0.123 px |
| pixels wrong by > 3 px (D1) | **0.00 %** |

On disparities of 30–122 px this is ~0.06 % relative error, concentrated at
object edges. **FP16 quantisation is effectively free** — no accuracy argument
against it.

---

## 3. Speed vs accuracy

Latency measured on-device; accuracy computed against a 32-iteration
full-resolution reference.

| configuration | latency | FPS | mean err | **p95 err** | > 3 px |
|---|---:|---:|---:|---:|---:|
| i7 @ 480×640 *(baseline)* | 686 ms | 1.46 | 0.672 px | 2.97 px | 4.95 % |
| **i4 @ 480×640** *(recommended)* | **530 ms** | **1.89** | 1.073 px | 5.22 px | 8.85 % |
| i7 @ 320×480 | 340 ms | 2.94 | 1.835 px | 12.27 px | 11.5 % |
| i4 @ 320×480 | 266 ms | 3.77 | 1.949 px | 12.56 px | 12.0 % |

Two findings:

**Reducing resolution costs more than reducing iterations.** `i7 @ 320×480`
(1.84 px) is worse than `i4 @ 480×640` (1.07 px), and its p95 error more than
doubles — **12.3 px vs 5.2 px**. Low resolution does not degrade uniformly; it
destroys thin structures. That matters directly, since the `cable` class exists
for collision avoidance and a cable is a few pixels wide.

**`i2 @ 480×640` is Pareto-dominated** — `i7 @ 320×480` is both faster and more
accurate. Cutting iterations to 2 is the wrong lever.

---

## 4. A predictive cost model

Fitted on the three 480×640 points, it predicts all five configurations —
including the two lower-resolution ones it never saw — to within ±2.6 ms:

```
latency_ms ≈ (328.2 + 50.9 × iterations) × (pixels / 307200)
```

- **328 ms is fixed cost**: feature encoding, correlation volume, upsampling
- **50.9 ms per refinement iteration** is the only part iterations can remove

So at 480×640 there is a **hard floor near 328 ms (3.05 FPS)** regardless of
iteration count. Resolution is the only lever that reaches the dominant term.

### Consequence: ≥ 10 FPS is arithmetically impossible

YOLO11s alone costs 105 ms, which exceeds the entire 100 ms budget for 10 FPS.
**Even with infinitely fast depth, the ceiling is 9.5 FPS.** Reaching just
5 FPS combined would leave RAFT 95 ms, requiring roughly a 271×203 input — at
which a cable becomes sub-pixel and invisible, defeating the purpose.

This is a **hardware/workload mismatch**, not a tuning problem.

---

## 5. Power and thermals

Measured via the onboard INA3221 rails over 20-minute sustained runs.

| | MAXN | 5W mode |
|---|---:|---:|
| latency | 686 ms | 918 ms |
| total board power (VDD_IN) | 6.82 W | 4.14 W |
| GPU+CPU rail | 3.33 W | 1.34 W |
| **energy per frame** | **4.68 J** | **3.80 J** |
| peak CPU temperature | 46.5 °C | 36.0 °C |
| drift over 20 min | +0.2 % | −0.0 % |

**5W mode is 19 % more energy-efficient per frame** despite being 34 % slower —
MAXN's extra speed costs disproportionately more power. For an
endurance-limited platform, 5W is the better operating point.

**Neither mode throttles.** Peak CPU was 46.5 °C against a ~97 °C limit, and
latency drifted < 0.2 % over 20 minutes, so the short-burst figures hold
indefinitely.

---

## 6. Why a learned stereo method, not a classical one

A natural objection is that classical stereo is far cheaper. Measured on the
same hardware and image pair:

| | RAFT-Stereo | OpenCV SGBM | VPI (hardware) |
|---|---:|---:|---:|
| latency | 686 ms | **608 ms** | **47 ms** |
| speedup vs RAFT | 1.0× | 1.13× | 14.6× |
| coverage | **100 %** | 67.8 % | 83.2 % |
| min resolvable distance | 0.52 m | ~0.6 m | **1.04 m** |
| > 3 px outliers | — | 22.5 % | — |

**SGBM is only 13 % faster** and leaves a third of the image undefined. The
reason is architectural: SGBM runs on four weak ARM cores while RAFT runs on
the GPU through TensorRT, so the "cheap" algorithm has no cheap place to
execute. It is not a trade — it is simply worse here.

**VPI is genuinely fast but has a hard 64 px disparity ceiling**, which with a
161 mm baseline is a **minimum-distance wall at 1.04 m**. 34.8 % of the test
scene fell inside it. Splitting the error by range makes the cause clear:

| region | share of image | mean error | > 3 px |
|---|---:|---:|---:|
| within range (≤ 64 px) | 60.5 % | 2.15 px | 15.6 % |
| beyond cap (> 64 px) | 22.7 % | **31.9 px** | **90.3 %** |

Inside its envelope VPI is good (median 0.37 px). Beyond it, output is not
degraded but simply wrong. **VPI is mismatched to this rig, not inferior.**

### This points at rig geometry, not algorithm choice

Disparity scales with baseline, so the minimum resolvable distance is
`f·B / 64`:

| baseline | min distance with VPI |
|---|---|
| 161 mm (current) | 1.04 m |
| 110 mm (already calibrated) | 0.71 m |
| ~77 mm | 0.50 m |

**A shorter-baseline rig would bring the scene inside VPI's envelope and make
21 FPS depth viable on this board**, at the cost of far-field depth precision.
For collision avoidance — where near objects are the hazard — that may be the
right trade, and it is worth evaluating.

---

## 7. Engineering notes

**RAFT-Stereo does not export to ONNX out of the box.** The iterative
refinement loop is fine (a Python `int`, so tracing unrolls it), but the
correlation lookup calls `F.grid_sample`, which ONNX supports only from opset
16 while TensorRT parses `GridSample` only from 8.5 — JetPack 4.6.1 ships 8.2.1.
Raising the opset moves the failure from export time to build time.

Resolved by exploiting a property specific to RAFT-*Stereo*: the correlation
volume is one pixel tall and the sampled y is always zero, so `grid_sample`
degenerates to 1-D interpolation along x, expressible in ops TensorRT 8.2
supports. Verified numerically identical to `F.grid_sample` (max difference
7.6 × 10⁻⁶) including out-of-bounds behaviour.

**INT8 is unavailable** on this board — it requires Compute Capability ≥ 6.1
for DP4A; the Nano is 5.3. FP16 is the floor.

---

## 8. Options

1. **Better hardware** — Orin Nano offers roughly 20× the compute in the same
   form factor and would move the pipeline into real-time.
2. **Shorter-baseline rig + VPI** — 21 FPS depth on this board, trading
   far-field precision. Testable today with the existing 110 mm calibration.
3. **Split the workload** — run detection on-device at 9.5 FPS and treat depth
   as a lower-rate or off-board signal.

Option 2 is the cheapest to evaluate and the most interesting result, since it
reframes the bottleneck as a design parameter rather than a hardware limit.

---

## 9. Real end-to-end pipeline (with cameras)

Every figure above this section is GPU-inference-only, timed on pre-loaded
static images. It excludes USB capture, rectification, resizing, and fusing
depth with detections — all real costs in an actual deployment. This section
closes that gap with two live Brio cameras on the actual rig, running the
final trained 11-class YOLO11s-seg model (box mAP50 0.621).

`scripts/jetson_end_to_end_bench.py` runs the full loop — capture, rectify
(using the existing stereo calibration), RAFT depth, YOLO detection, depth-per-
box fusion — timed stage by stage over 20 real stereo pairs, at a 110 mm
baseline (rig baseline at time of test; see the baseline note below).

| stage | mean | range |
|---|---:|---:|
| capture (2× Brio, USB, MJPEG) | 82.6 ms | 80.3–86.0 ms |
| rectify (`cv2.remap` ×2) | 52.8 ms | 49.8–66.9 ms |
| RAFT-Stereo (480×640, 7 iters) | 694.5 ms | 689.6–700.4 ms |
| YOLO11s-seg (640×640, 11 classes) | 177.0 ms | 174.3–183.8 ms |
| depth/detection fusion | 12.2 ms | 11.3–14.4 ms |
| **TOTAL** | **1019.0 ms** | **→ 0.98 FPS** |

RAFT's 694.5 ms matches the isolated 686–689 ms figure almost exactly — the
engine performs identically whether fed a static test image or a live captured
frame. YOLO's 177.0 ms sits 36 ms above its isolated 140.8 ms because the
isolated number only timed the raw `execute()` call; this figure honestly
includes real preprocessing (resize, BGR→RGB) and postprocessing (per-class
NMS, sigmoid mask reconstruction from 32 prototype coefficients) — a superset
of what the GPU-only number measures, not a discrepancy to explain away.

**Real deployment is ~23% slower than the GPU-only estimate implied** (826.8 ms
→ 1019.0 ms). Capture and rectification alone cost 135.4 ms — more than the
entire YOLO inference — and would have gone completely unaccounted for in any
comparison based purely on `trtexec` numbers.

### Two bugs, both worth documenting

**`cv2.dnn.NMSBoxes` crashes on JetPack's OpenCV build.** It worked in desktop
testing but failed on-device with `SystemError: <built-in function NMSBoxes>
returned NULL without setting an error` — an uncaught C++ exception that never
reaches Python as a catchable error. The trigger: numpy `float32` scalars
passed where the binding expects native Python floats, or any box with
non-positive width/height. Fixed by casting every coordinate explicitly and
filtering degenerate boxes before the call, in `scripts/yolo_seg_postprocess.py`
(shared by both the correctness-check script and the full pipeline, so one fix
covers both — verified with a synthetic case reproducing the exact trigger:
mixed valid float32 detections plus one negative-width box).

**Missing warm-up inflated the first measurement by ~10×.** The initial
end-to-end run showed a single-frame outlier — RAFT hit 1471.9 ms, YOLO hit
1607.8 ms — against means of ~700 ms and ~180 ms respectively. The diagnostic
that ruled out thermal throttling: `jetson_clocks --show` confirmed CPU, GPU,
and EMC all pinned at their maximum, ruling out DVFS. The real tell was
re-running the whole test independently: YOLO's outlier landed at 1607.8 ms
the first time and **1609.6 ms the second — under 2 ms apart.** Thermal or
scheduling noise doesn't reproduce that precisely; that precision is the
signature of a deterministic one-time cost. The cause: this script never
discarded an untimed first inference before starting the timed loop, unlike
`jetson_trt_infer.py` and `jetson_yolo_seg_infer.py`, which both do. Frame 1
silently paid for CUDA context lazy-init and algorithm caching for *both*
engines inside the "real" measurement. Adding the same warm-up discipline
removed the outlier entirely — the corrected run's RAFT max/min spread is
689.6–700.4 ms, tight and reproducible.

### Capture resolution vs. latency

The 1019.0 ms figure above was captured at the Brio's native 1920×1080. Since
capture resolution and inference resolution are independent — RAFT and YOLO
always run at their fixed engine input size (480×640 and 640×640) regardless
of what resolution the frame arrives at, because the rectified frame is
resized down before inference either way — the question is how much of the
pipeline's cost is actually sensitive to capture resolution.

`--capture-width`/`--capture-height` on `jetson_end_to_end_bench.py` request a
non-native resolution from both cameras and regenerate the stereo
rectification maps to match (`rescale_calibration()`): camera intrinsics
(fx, fy, cx, cy) scale linearly with the resolution ratio, distortion
coefficients and rotation are resolution-independent, so K and P are scaled
and `cv2.initUndistortRectifyMap` reruns at the new size — reusing the
native-resolution maps on a differently-sized frame would silently misalign
the stereo pair instead of erroring. Both cameras were confirmed (via a
per-resolution fresh-`VideoCapture` probe, since a live GStreamer pipeline
won't renegotiate resolution once opened) to actually honor 1920×1080,
1280×720, and 640×480 over MJPEG — 4K is not offered by this hardware.

| Capture res | capture | rectify | RAFT | YOLO | fuse | **TOTAL** | **FPS** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1920×1080 (native) | 82.8 ms | 54.3 ms | 695.4 ms | 171.9 ms | 9.3 ms | **1013.7 ms** | **0.99** |
| 1280×720 | 50.2 ms | 26.8 ms | 692.4 ms | 172.1 ms | 9.1 ms | **950.6 ms** | **1.05** |
| 640×480 | 34.9 ms | 11.8 ms | 684.8 ms | 168.2 ms | 9.2 ms | **909.0 ms** | **1.10** |

RAFT and YOLO stay flat within noise across all three, as expected. The
savings come entirely from cheaper capture and rectification: 1080p→480p cuts
capture by 58% and rectify by 78%, but total latency only drops ~10%
(1013.7→909.0 ms, +11% FPS), because RAFT alone is 68–73% of the total budget
and is untouched by capture resolution. Lowering capture resolution is a real
but modest lever here — the bottleneck is RAFT's inference cost, not I/O.

### Baseline note

This run used a 110 mm baseline (the rig's configuration at the time of
testing), not the 161 mm baseline referenced in §6. **Baseline has no effect on
any of the timing above** — RAFT and YOLO process pixels at a fixed resolution
regardless of the physical distance between the two cameras; baseline only
enters via the final `depth = focal × baseline / disparity` division, which is
microseconds of arithmetic and already included in the 12.2 ms fusion figure.
Re-testing at multiple baselines would not change the FPS number and was not
done for that reason. Baseline *does* matter for §6's VPI disparity-ceiling
argument (a genuinely different question — minimum resolvable distance, not
speed) — re-validating that claim with real captures at each baseline remains
a separate, optional follow-up, not part of this latency measurement.

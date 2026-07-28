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

| stage | latency | rate |
|---|---:|---:|
| YOLO11s detection (640×640) | 105 ms | 9.5 FPS |
| RAFT-Stereo depth (480×640, 7 iters) | 686 ms | 1.46 FPS |
| **Combined (GPU inference only)** | **791 ms** | **1.26 FPS** |

At 1.26 FPS a drone moving at 2 m/s travels **1.58 m between depth frames**.

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
| ~~i2 @ 480×640~~ *(dominated)* | 431 ms | 2.32 | 2.167 px | 11.48 px | 16.0 % |
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

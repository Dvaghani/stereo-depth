# Orin NX 16GB — benchmark plan

Target: reproduce the Jetson Nano measurements on the Orin NX so the two are
directly comparable, then push for ≥ 25 FPS using the accelerators the Nano
does not have, and finally run the Brio stereo rig live.

**Hardware:** Holybro Pixhawk 6X kit — Orin NX 16 GB, Holybro Jetson baseboard,
512 GB NVMe, Intel 8265 WiFi, IMX219-200 CSI camera, PM02D power module.

---

## 0. Before Friday

- [x] Reconnect the Expansion drive and finish the training run
- [x] **YOLO INT8 calibration set** — `outputs/int8_calib_yolo/`, 322 frames,
      119 MB, all 11 classes ≥ 30 frames. Built by
      `scripts/build_int8_calib_set.py` (stratified, since a uniform sample of
      the val split left `cable` at 8 frames — topped up from train to 30).
- [x] **RAFT INT8 calibration set** — `outputs/int8_calib_raft/`, 145 rig
      pairs, 149 MB, 100% domain match by construction. Built from two
      `capture_calib_burst.py` bursts (317 pairs total) filtered by
      `build_raft_calib_set.py --min-sharpness 150`, which dropped 172 pairs
      to motion blur (timer-driven capture catching mid-motion frames — see
      §5 note below). 145 clean pairs is within the normal range for INT8
      calibration; did not pursue a third capture round.
- [ ] Copy both calibration sets to the Orin (USB stick or scp)
- [ ] Copy `docs/jetson_deployment_results.md` so Nano figures are at hand
- [ ] Check the baseboard manual for **how many CSI lanes** are exposed (§7)
- [ ] **Check whether the kit arrived pre-flashed** (power on with a display
      attached) — if not, arrange a spare Ubuntu 20.04/22.04 host PC for
      flashing before Friday, not on the day. See §0.5.

Scripts to carry over — all already written and portable:

```
scripts/jetson_trt_infer.py              # engine correctness check
scripts/jetson_sustained_bench.py        # latency + thermals + power
scripts/jetson_classical_stereo_bench.py # SGBM / VPI comparison
scripts/export_raft_onnx.py              # ONNX export (runs on desktop)
scripts/compare_disparity.py             # accuracy vs desktop reference
```

---

## 0.5 Initial flash (only if it isn't pre-flashed)

This is the one part of Friday's setup that doesn't work like the Nano. The
Nano boots off a microSD card — reflashing it is just writing a new image with
balenaEtcher, no host PC required. **The Orin NX module has no onboard eMMC.**
It boots entirely from external storage — the 512 GB NVMe in the kit — and
only has a small onboard QSPI flash, which holds the bootloader, not the OS.
So if JetPack was never written to that NVMe, the board cannot boot on its own
yet, and there's no SD-card-style shortcut around that.

**Check this before Friday, not on the day**: Holybro's "Komplettset" kits
sometimes ship pre-flashed. Check the box/manual, or just power it on with a
display attached — if it reaches a desktop or login prompt, skip this section
entirely and go to §1.

### If it needs flashing

Requires a **separate Ubuntu host PC** — this is not optional hardware, and
it's the thing most likely to be missing on the day if not arranged in
advance. JetPack 6's SDK Manager wants Ubuntu 20.04 or 22.04 on the host (a VM
is possible but flakier for USB device passthrough during flashing — a real
Ubuntu machine is safer if one is available).

1. Install **NVIDIA SDK Manager** on the host: https://developer.nvidia.com/sdk-manager
2. Connect host → the Orin's recovery port. NVIDIA's reference Orin NX/Nano
   carrier uses **USB-C** for this; Holybro's baseboard is custom, so confirm
   in its manual rather than assuming — and confirm it's not the same port
   used for normal peripherals. Cable: USB-C on the module end, USB-A or
   USB-C on the host end depending on what the host has. **Use a cable known
   to carry data, not just power** — many USB-C cables (especially bundled
   charging cables) have no data lines wired, in which case the board powers
   on but never enumerates over USB, which looks identical to "recovery mode
   didn't work" and is a bad thing to debug live.
3. Put the module into **Force Recovery mode**: hold the **REC** button, tap
   **RST** (or power on) while still holding REC, then release REC after ~2s.
   Exact button layout is baseboard-specific — confirm against the manual
   rather than assuming Nano/Xavier conventions carry over.
4. Confirm recovery mode from the host: `lsusb | grep -i nvidia` should show
   the module enumerated as a USB device (not a normal boot)
5. In SDK Manager: select the Orin NX target, JetPack 6.x, and **NVMe** as the
   storage target for the rootfs (not eMMC — there isn't one). Flash.
6. Takes roughly **30–45 minutes**. Once done, the board boots from the NVMe
   on every subsequent power-on exactly like a normal Linux box — this step
   does not repeat.

After this, §1 "First boot" onward applies normally — `nvpmodel`, `jtop`,
`dpkg -l | grep tensorrt`, etc. all assume JetPack is already running, which is
true from here on.

---

## 1. First boot

```bash
sudo nvpmodel -q                 # note available modes; Orin NX has several
sudo nvpmodel -m 0               # MAXN
sudo jetson_clocks
sudo pip3 install -U jetson-stats && sudo reboot   # then: jtop
free -h; df -h /                 # 16 GB RAM, 512 GB NVMe
dpkg -l | grep -i tensorrt       # expect 8.5+ (vs 8.2.1 on Nano)
```

Unlike the Nano, **do not** disable the GUI for RAM — 16 GB is ample.

**Power modes matter more here.** Orin NX offers 10 W / 15 W / 25 W profiles.
Benchmark at least MAXN and one lower profile so the efficiency comparison
mirrors the Nano's MAXN-vs-5W result.

---

## 2. Phase 1 — PyTorch baseline (easiest first)

The Nano forced the ONNX→TensorRT route because JetPack 4.6.1 shipped Python
3.6 and no usable PyTorch. **JetPack 6 on Orin ships Python 3.10 with an
official PyTorch wheel**, so the existing code can run nearly unmodified.

```bash
# NVIDIA's PyTorch wheel for JetPack 6
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/<wheel>
pip3 install ultralytics

git clone git@github.com:Dvaghani/stereo-depth.git
```

This gives a working end-to-end pipeline in an hour rather than a day, and a
PyTorch-vs-TensorRT comparison the Nano could not provide.

Expect PyTorch to be perhaps 2–3× slower than TensorRT — useful as a floor, not
as the headline number.

---

## 3. Phase 2 — TensorRT FP16 (direct Nano comparison)

Run exactly what was run on the Nano so the columns line up.

```bash
# YOLO — export on-device (engines are not portable across devices)
yolo export model=best.pt format=engine half=True imgsz=640

# RAFT — export ONNX on the desktop, build the engine here
/usr/src/tensorrt/bin/trtexec --onnx=raft_realtime_i4_480x640.onnx \
  --saveEngine=raft_i4_fp16.trt --fp16
/usr/src/tensorrt/bin/trtexec --loadEngine=raft_i4_fp16.trt
```

**Note:** TensorRT 8.5+ supports `GridSample` natively, so the 1-D sampler
workaround is no longer required. Export both ways and compare — if stock
`grid_sample` now builds, that is worth reporting as a platform difference.

Then correctness and sustained behaviour, as on the Nano:

```bash
python3 jetson_trt_infer.py --engine raft_i4_fp16.trt \
  --left left.png --right right.png --out disp_orin.npy --runs 50

python3 jetson_sustained_bench.py --engine raft_i4_fp16.trt \
  --left left.png --right right.png --minutes 20 --tag orin_maxn
```

The power-rail paths differ on Orin; `jetson_sustained_bench.py` globs for them
and will report `power rails: NONE FOUND` if the layout has moved. Fix the glob
rather than dropping the measurement — the energy-per-frame comparison is one
of the more interesting results.

---

## 4. Phase 2.5 — spend the headroom on resolution, not just speed

Phase 2 pins RAFT at 480×640 on purpose, to isolate hardware speed from
workload — do not skip it. This phase asks the *opposite* question: now that
speed is no longer the constraint, what is the best **resolution** to run at,
i.e. the highest quality threshold Orin can afford.

This matters because resolution is not just a speed knob on this project — the
Nano's §3 accuracy study (`jetson_deployment_results.md`) found that dropping
resolution **destroys thin structures disproportionately**: `i7 @ 320×480`
(1.84 px mean error) was *worse* than `i4 @ 480×640` (1.07 px) despite fewer
iterations, and p95 error more than doubled (12.3 px vs 5.2 px). The `cable`
class is a few pixels wide, so it is exactly what a low-resolution engine loses
first. On the Nano this was a hard constraint — there was no headroom to spend.
On Orin there might be, and if so it directly strengthens the cable-detection
argument in the thesis.

**Steps:**

1. Re-export RAFT ONNX at one or two resolutions above 480×640 — e.g.
   640×896 and, if it still fits, closer to native (960×1280 or 1080×1920).
   `scripts/export_raft_onnx.py` already parameterises input shape.
2. Build an FP16 engine per resolution (same `trtexec --fp16` command as
   Phase 2, just a different `--onnx` input and `--saveEngine` name).
3. Measure latency for each with `jetson_sustained_bench.py`, same as Phase 2.
4. Measure accuracy for each with `compare_disparity.py` against the same
   32-iteration full-resolution reference the Nano study used, so the numbers
   are directly comparable across both boards. If `compare_disparity.py`
   doesn't already break out error on thin/edge structures specifically,
   extend it to — that per-region number is what makes the cable argument
   quantitative on Orin, not just qualitative.
5. Pick the **highest resolution that still clears the latency budget** you
   need (target ≥ 25 FPS combined with YOLO, per §7 below) — that resolution
   *is* the quality threshold this phase exists to find, not just whichever
   config happens to be fastest.

| resolution | pixels | RAFT latency | combined FPS | mean err | p95 err | fits budget? |
|---|---:|---:|---:|---:|---:|---|
| 480×640 *(Nano-comparable)* | 307,200 | | | | | |
| 640×896 | 573,440 | | | | | |
| 960×1280 / native | | | | | | |

---

## 5. Phase 3 — INT8 and DLA (the push to 25 FPS)

Both are unavailable on the Nano (CC 5.3 lacks DP4A; no DLA hardware).

### INT8

Needs a calibration set of representative images — random data produces poor
scales. Ultralytics handles calibration internally:

```bash
yolo export model=best.pt format=engine int8=True data=dataset.yaml imgsz=640
```

For RAFT, INT8 needs a custom calibrator feeding real stereo pairs. Worth
attempting only after FP16 numbers are recorded; disparity regression must be
checked with `compare_disparity.py`, since INT8 on a regression task is far
riskier than on classification.

#### The RAFT calibration set has a domain-match problem

`scripts/build_raft_calib_set.py` builds the set and reports how much of its
disparity distribution overlaps the rig's actual operating range (~30–122 px at
480×640). That number is the RAFT analogue of class coverage for YOLO: INT8
scales are activation ranges, and RAFT's correlation-volume activations are
driven by disparity magnitude, so a set centred on the wrong disparities
calibrates for an operating point the rig never reaches.

Measured overlap with the public data already on the Expansion drive. The rig
range depends on baseline (disparity scales with it), so both are shown —
`--rig-disp-range` sets which one is scored against:

| calibration source | pairs | median disp | overlap @160 mm (30–122 px) | overlap @110 mm (21–84 px) |
|---|---:|---:|---:|---:|
| KITTI-dominated mix | 200 | 17.3 px | **13 %** | — |
| Middlebury only | 23 | 26.1 px | **41 %** | **54 %** |
| rig captures | 10 available | — | 100 % by construction | 100 % |

KITTI is the worst source despite being the standard benchmark: at 1242×375
(3.3:1) squeezed into the 1.33:1 engine input, its disparities scale down to a
median of 17 px. Middlebury is better (~1.5:1) and scores 54 % at the rig's
current 110 mm baseline — but only 23 scenes exist, far short of the ~200 a
calibration set wants.

**Therefore: capture rig pairs before building the final set.**
`scripts/capture_calib_burst.py` captures unattended on a timer (the existing
`capture_stereo.py` does one pair per invocation, which does not scale to 200).
It also runs a quick calibration check itself before the burst starts — same
SIFT residual-disparity method as `check_calib.py`, no checkerboard — and
**aborts if the rig has drifted** rather than silently baking bad
rectification into all 200 pairs. Point at a texture-rich scene (bookshelf,
cluttered desk) for the check to have something to match on:

```bash
python scripts/capture_calib_burst.py \
    --calib outputs/calibration_110mm/stereo_calib.npz \
    --left-index 2 --right-index 0 --count 200 --interval 1.0 --no-preview
# if it aborts as drifted:
#   python scripts/quick_recalib.py --calib outputs/calibration_110mm/stereo_calib.npz \
#       --out outputs/calibration_110mm/stereo_calib_refreshed.npz
#   then re-run capture_calib_burst.py with the refreshed file

python scripts/build_raft_calib_set.py --sources rig \
    --rig-glob "outputs/rig_burst_*/pair_*" \
    --out outputs/int8_calib_raft --count 200
```

Vary distance while capturing — especially the near end, where disparity is
largest. 200 near-identical frames of one wall calibrate for a single operating
point.

### Ground truth: use KITTI/Middlebury for scoring, not calibrating

The public data is still valuable, just for a different job. **The Brio rig has
no ground-truth disparity**, so every accuracy figure in
`jetson_deployment_results.md` is *relative* — §2 compares against a desktop
FP32 reference, §3 against a 32-iteration reference. Those measure quantisation
drift and self-consistency, not true accuracy.

KITTI 2015 (200 pairs) and Middlebury 2014 (23 scenes) on the Expansion drive
both ship GT disparity, and `scripts/eval_raft.py` already evaluates against
them (`--dataset kitti|middlebury --data-root ...`). It is PyTorch-based, so it
could not run on the Nano — but **JetPack 6 on the Orin has PyTorch**, so it can
run there. That upgrades the thesis claim from "FP16 drifts 0.0345 px from our
own FP32 output" to absolute EPE / D1-all against ground truth, directly
comparable to published RAFT-Stereo numbers.

### DLA offload

Orin NX has **two NVDLA v2 accelerators**. Running YOLO on a DLA frees the GPU
entirely for depth, so detection cost leaves the critical path instead of adding
to it.

```bash
/usr/src/tensorrt/bin/trtexec --onnx=yolo.onnx --saveEngine=yolo_dla.trt \
  --useDLACore=0 --fp16 --allowGPUFallback
```

Check the log for layers falling back to GPU — heavy fallback negates the
benefit. Then measure YOLO-on-DLA and RAFT-on-GPU **concurrently**, not
serially, since concurrency is the entire point.

---

## 6. Numbers to fill in

This table is the fixed-480×640 comparison only (Phase 2 + Phase 3). The
resolution-sweep results from Phase 2.5 live in their own table in §4 — the
"best config" for the thesis may end up being a Phase 2.5 row, not one of
these, if it clears the FPS budget at higher resolution.

| stage | Nano (measured) | Orin FP16 | Orin INT8 | Orin INT8+DLA |
|---|---:|---:|---:|---:|
| YOLO11s @ 640 | 105 ms | | | |
| RAFT i4 @ 480×640 | 530 ms | | | |
| combined | 635 ms | | | |
| **FPS** | **1.57** | | | |
| power (W) | 6.82 | | | |
| energy/frame (J) | 4.68 | | | |
| peak temp (°C) | 46.5 | | | |
| disparity error vs FP32 | 0.0345 px | | | |

Also worth re-running the **classical stereo comparison** — VPI on Orin has PVA
hardware the Nano lacks, so its 47 ms could drop further. And its 64 px
disparity ceiling may have been raised in newer VPI versions, which would change
the baseline-geometry conclusion entirely.

---

## 7. Phase 4 — Brio stereo rig on Orin

### Calibration transfers unchanged

Same cameras, same rig, same capture resolution ⇒ `outputs/calibration_160mm/`
and `calibration_110mm/` remain valid. Copy them over; **do not** recalibrate
unless the physical rig is disturbed.

### USB bandwidth is the thing to watch

Two Brios at 1920×1080 will saturate a shared USB controller if sent
uncompressed. Force MJPEG, as `live_detect.py` already does:

```python
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
```

Verify what each camera actually negotiates:
```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

If throughput is a problem, put the two cameras on **different USB controllers**
rather than one hub — check `lsusb -t` for the tree.

### Consider the CSI alternative

The kit ships an IMX219-200 **CSI** camera. If the baseboard exposes two CSI
lanes, a CSI stereo pair beats USB substantially: lower latency, hardware
frame-sync (genuinely important for stereo — unsynchronised USB frames introduce
disparity error on anything moving), and no USB contention. Worth checking the
baseboard spec even if the Brios are used initially.

### Measure the full loop, not just inference

This closes task #12, still open from the Nano. All figures so far are GPU
inference on pre-loaded images; a real loop also pays for capture,
rectification, resize, NMS and fusion — estimated at 100–200 ms on the Nano but
never measured.

Instrument each stage separately:

```
capture → rectify → resize → RAFT → YOLO → fuse → display
```

The per-stage breakdown matters more than the total, because it shows whether
the bottleneck is still depth or has moved to capture.

---

## 8. What would make this a strong thesis chapter

1. **Two-platform comparison** with identical methodology — the Nano numbers
   become a baseline rather than a dead end
2. **Cost model validation** — does `(328 + 50.9·iters) × pixels/307200` hold on
   Orin with different constants? If the *structure* transfers and only the
   coefficients change, that is a genuinely useful result
3. **Where the bottleneck moves** — on the Nano, depth dominates. If Orin makes
   depth cheap enough that capture or rectification dominates, that reframes the
   whole system design
4. **Whether 25 FPS is met**, and by which combination of levers
5. **The quality threshold from Phase 2.5** — not just "how fast can Orin go,"
   but "how much resolution can Orin afford before it must trade accuracy for
   speed the way the Nano was forced to." A board that removes that forced
   tradeoff for the cable class is a stronger result than raw FPS alone.

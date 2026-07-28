# Orin NX 16GB — benchmark plan

Target: reproduce the Jetson Nano measurements on the Orin NX so the two are
directly comparable, then push for ≥ 25 FPS using the accelerators the Nano
does not have, and finally run the Brio stereo rig live.

**Hardware:** Holybro Pixhawk 6X kit — Orin NX 16 GB, Holybro Jetson baseboard,
512 GB NVMe, Intel 8265 WiFi, IMX219-200 CSI camera, PM02D power module.

---

## 0. Before Friday

- [ ] Reconnect the Expansion drive and finish the training run
- [ ] Build the INT8 calibration set (~300 representative frames — see §4)
- [ ] Copy `docs/jetson_deployment_results.md` so Nano figures are at hand
- [ ] Check the baseboard manual for **how many CSI lanes** are exposed (§6)

Scripts to carry over — all already written and portable:

```
scripts/jetson_trt_infer.py              # engine correctness check
scripts/jetson_sustained_bench.py        # latency + thermals + power
scripts/jetson_classical_stereo_bench.py # SGBM / VPI comparison
scripts/export_raft_onnx.py              # ONNX export (runs on desktop)
scripts/compare_disparity.py             # accuracy vs desktop reference
```

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

## 4. Phase 3 — INT8 and DLA (the push to 25 FPS)

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

## 5. Numbers to fill in

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

## 6. Phase 4 — Brio stereo rig on Orin

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

## 7. What would make this a strong thesis chapter

1. **Two-platform comparison** with identical methodology — the Nano numbers
   become a baseline rather than a dead end
2. **Cost model validation** — does `(328 + 50.9·iters) × pixels/307200` hold on
   Orin with different constants? If the *structure* transfers and only the
   coefficients change, that is a genuinely useful result
3. **Where the bottleneck moves** — on the Nano, depth dominates. If Orin makes
   depth cheap enough that capture or rectification dominates, that reframes the
   whole system design
4. **Whether 25 FPS is met**, and by which combination of levers

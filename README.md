# Stereo Depth + Construction-Object Detection

Code for the master's thesis *"Depth Map Generation based on Application
Specific Stereo-Vision"*: a stereo depth pipeline paired with an 11-class
construction-site segmentation model, measured on a real camera rig and
deployed to NVIDIA Jetson hardware.

```
Left + Right  ─►  rectify  ─┬─►  RAFT-Stereo  ─►  disparity  ─►  depth (m)
                            │                                      │
                            └─►  YOLO11s-seg  ─►  masks  ──────────┴─►  per-object depth
```

The two halves meet in `scripts/detect_depth.py` (desktop) and
`scripts/jetson_end_to_end_bench.py` (on-device): detection masks sample the
dense disparity map, giving a distance per detected object.

## Status

| part | state |
|---|---|
| Stereo matching | RAFT-Stereo (realtime config), fine-tuned; optional Laplace uncertainty head |
| Detection | YOLO11s-seg, 11 classes, box mAP50 **0.621** / mask mAP50 **0.502** |
| Rig | Dual Logitech Brio, calibrated at three baselines (110 / 160 / 280 mm) |
| Deployment | Jetson Nano measured end-to-end; Orin NX benchmark planned (`docs/orin_benchmark_plan.md`) |

Trained weights are **not** in this repository — `checkpoints/` and `weights/`
are gitignored. Paths in the examples below refer to local runs.

## Layout

```
stereo_unet/
├── configs/                 # YAML training configs (kitti, middlebury, sceneflow; unet/aanet variants)
├── csrc/                    # CUDA/C++ correlation kernel for the early StereoUNet model
├── construction_yolo/       # dataset pipeline for the 11-class detector
│   ├── extract_frames.py        # frames from site video
│   ├── prelabel_*.py            # YOLO-World / GroundingDINO / Roboflow / HSV pre-labelling
│   ├── merge_rounds.py          # merge labelling rounds into one dataset
│   └── datasets/                # taxonomy + unified-merge tooling
├── docs/                    # results write-ups (see "Documentation")
├── scripts/                 # everything runnable — see below
├── src/
│   ├── models/              # stereo_unet, aanet, raft_uncertainty, cost_volume, …
│   ├── datasets/            # kitti2015, middlebury2014, sceneflow, transforms
│   └── utils/               # losses, metrics, io (PFM/KITTI), online_rectify
├── third_party/             # aanet (vendored); RAFT-Stereo (gitignored, cloned by setup)
├── checkpoints/             # (gitignored) saved weights
├── outputs/                 # (gitignored) captures, calibrations, inference results
└── requirements.txt
```

### Scripts by purpose

**Rig and calibration**

```
check_cameras.py                  # enumerate cameras, confirm both Brios are present
which_camera_is_left.py           # decide the physical left camera by feature matching
identify_board.py                 # recover a checkerboard's inner-corner count from a photo
rig_stability_probe.py            # is the rig mechanically settled enough to calibrate?
capture_calibration.py            # interactive calibration-pair capture
calibrate_stereo.py               # solve intrinsics + extrinsics, write stereo_calib.npz
check_calib.py / verify_rectification.py   # sanity-check a calibration
refine_vertical_shift.py          # residual vertical-disparity correction
recalibrate_filtered.py / quick_recalib.py # re-solve from a filtered pair set
solve_extrinsics_from_stations.py # re-solve R/T from measurement captures (rig drift)
rerectify_captures.py             # re-rectify saved raw pairs against a new calibration
```

**Capture and live view**

```
capture_stereo.py                 # one rectified + raw stereo pair per invocation
capture_calib_burst.py            # timed burst capture (INT8 calibration sets)
live_depth.py / live_raft.py      # live disparity preview
live_detect.py / live_detect_nocal.py      # live detection preview
detect_depth.py                   # detection masks sampled against the depth map
```

**Training and evaluation**

```
train.py                          # StereoUNet / AANet training loop
eval.py / eval_fair.py            # KITTI / Middlebury metrics on a checkpoint
infer.py / infer_raft.py          # single stereo pair → disparity + depth (+ confidence)
eval_raft.py                      # RAFT-Stereo on the standard benchmarks
raft_accuracy_sweep.py            # desktop sweep over iterations and resolution
eval_depth_accuracy.py            # geometric floor: checkerboard corners, no matcher
eval_raft_stations.py             # RAFT on the same station captures
eval_raft_laser.py                # RAFT on ordinary objects vs laser rangefinder
baseline_stats.py                 # range + precision vs distance, derived from calibration
```

**Deployment (Jetson)**

```
export_raft_onnx.py               # RAFT-Stereo → ONNX for TensorRT
onnx_sampler1d.py                 # ONNX op shim needed by the export
build_raft_calib_set.py           # INT8 calibration set: real rig stereo pairs
build_int8_calib_set.py           # INT8 calibration set: class-stratified YOLO frames
build_int8_engine.py              # INT8 engine build with a real IInt8Calibrator
make_trt_inputs.py                # raw float32 .bin inputs for trtexec --loadInputs
jetson_trt_infer.py               # run + validate the engine on-device
jetson_yolo_seg_infer.py          # YOLO11s-seg via TensorRT
yolo_seg_postprocess.py           # mask decode + NMS (class names live here)
compare_disparity.py              # on-device engine vs desktop PyTorch reference
trtexec_json_to_npy.py            # convert trtexec output without pycuda
jetson_end_to_end_bench.py        # full pipeline with live cameras
jetson_sustained_bench.py         # latency, thermals, power under sustained load
jetson_classical_stereo_bench.py  # VPI / OpenCV SGBM comparison
```

### Detection classes

```
container   pole       scaffolding   crane    person   machinery
vehicle     building   suspended-load barrier  cable
```

Defined in `scripts/yolo_seg_postprocess.py` and
`construction_yolo/datasets/taxonomy.py`.

## Quick start

```bash
pip install -r requirements.txt

# AANet is vendored in third_party/; RAFT-Stereo is cloned separately
git clone https://github.com/princeton-vl/RAFT-Stereo third_party/RAFT-Stereo
```

Depth from a saved stereo pair:

```bash
python scripts/infer_raft.py \
    --ckpt  checkpoints/raft_middlebury_ft/best.pth \
    --left  outputs/capture_160mm/left.png \
    --right outputs/capture_160mm/right.png \
    --baseline 0.1606 --focal 1247.2 --scale 0.5 \
    --out   outputs/raft_infer/test1
```

Writes colorized disparity, raw `disparity.npy` in px, 16-bit `depth_mm.png`,
and — with an uncertainty checkpoint — a confidence map and a confidence-masked
disparity image.

Calibrate a newly mounted rig:

```bash
# 1. confirm which camera index is physically on the left
python scripts/which_camera_is_left.py --indices 0 4

# 2. confirm the rig has mechanically settled before capturing
python scripts/rig_stability_probe.py --left-index 0 --right-index 4

# 3. capture pairs -> outputs/calibration_160mm/{left,right}/
python scripts/capture_calibration.py \
    --baseline-mm 160 --pattern 9 6 --square-mm 25 \
    --left-index 0 --right-index 4

# 4. solve -> stereo_calib.npz
python scripts/calibrate_stereo.py \
    --input outputs/calibration_160mm \
    --baseline-mm 160 --pattern 9 6 --square-mm 25
```

Export for the Jetson:

```bash
python scripts/export_raft_onnx.py \
    --ckpt checkpoints/raft_middlebury_ft/best.pth \
    --out  raft_realtime_i7_480x640.onnx \
    --iters 7 --height 480 --width 640
```

## Results

### Baseline comparison (110 / 160 / 280 mm)

Depth precision `sigma_Z`, from plane-fit residuals on checkerboard corners:

| distance | 110 mm | 160 mm | 280 mm |
|---|---:|---:|---:|
| 1.0 m | 0.9 mm | 0.6 mm | 0.5 mm |
| 2.0 m | 2.9 mm | 2.4 mm | 1.3 mm |
| 4.0 m | 11.4 mm | 9.4 mm | 5.9 mm |
| 6.0 m | 31.3 mm | 21.1 mm | 10.2 mm |

Monotonic in baseline at every distance. Sub-pixel disparity noise is
effectively constant across the three rigs (0.110 / 0.121 / 0.129 px), so the
difference comes from `f·B` alone and depth precision follows `Z²/(f·B)` as
predicted. RAFT's own disparity error shows no trend with baseline, so the
geometric advantage survives the matcher.

Full write-up with method, PnP cross-check, the rig-drift finding and
limitations: `docs/baseline_comparison_results.md`. Session procedure:
`docs/baseline_capture_protocol.md`.

### Jetson Nano deployment

Measured on-device, 11-class YOLO11s-seg + RAFT-Stereo at 480×640:

| stage | GPU inference only | real end-to-end |
|---|---:|---:|
| YOLO11s-seg (640×640) | 140.8 ms | 177.0 ms |
| RAFT-Stereo (7 iters) | 686 ms | 694.5 ms |
| capture (2× Brio, USB) | — | 82.6 ms |
| rectify | — | 52.8 ms |
| fusion | — | 12.2 ms |
| **total** | **826.8 ms → 1.21 FPS** | **1019.0 ms → 0.98 FPS** |

The pipeline runs correctly but not fast enough for reactive obstacle
avoidance; dense stereo dominates and detection is comfortably real-time.
FP16 quantisation is effectively free (0.0345 px mean disparity error against
the desktop FP32 reference, 0.00 % D1). GPU-only timing undercounts real
deployment by ~23 %. Details, the predictive cost model, power/thermals and the
classical-stereo comparison: `docs/jetson_deployment_results.md`.

## Model history

Three stereo models live in `src/models/`, in the order they were tried:

**`stereo_unet.py`** — the original design: a Siamese encoder, dot-product
correlation into a *compressed* cost volume `(B, D, H, W)`, and a 2D U-Net
aggregator. The compression exists because state-of-the-art methods (PSMNet,
IGEV-Stereo) aggregate a 4D volume with 3D convolutions, which does not fit the
Jetson Nano's 4 GB at any usable resolution. Features are L2-normalized so the
correlation behaves like a cosine similarity and keeps the cost volume in
`[-1, 1]`, which makes the soft-argmin softmax well-behaved without temperature
tuning. `csrc/` holds a hand-written CUDA correlation kernel for this model,
deliberately register-friendly to fit sm_53 occupancy; without it the model
falls back to pure PyTorch and prints a one-time warning.

**`aanet.py`** — adaptive aggregation, and where the Laplace uncertainty head
was developed.

**`raft_uncertainty.py`** — RAFT-Stereo in its realtime configuration, which is
what the deployed pipeline uses. It extends the upstream model with the
uncertainty head ported from the AANet work: a small conv head on the finest
GRU hidden state predicts `log(b)`, the scale of a Laplace over the disparity
error, trained in a second phase with the backbone frozen under the NLL loss
`|d − d*| / b + log(2b)`. It needs `third_party/RAFT-Stereo` on `sys.path`.

Evaluation metrics are shared across all three: `EPE`, `D1-all` (`|err| > 3 px`
and `> 5 %` relative), and `bad-1` / `bad-2` / `bad-3`.

## Dataset layout

**KITTI 2015** (cvlibs.net/datasets/kitti):

```
kitti2015/
    training/
        image_2/         # 000000_10.png .. 000199_10.png (left)
        image_3/         # right
        disp_occ_0/      # 16-bit PNG ground-truth disparity (/256.0)
    testing/
        image_2/
        image_3/
```

**Middlebury 2014** (vision.middlebury.edu/stereo):

```
middlebury2014/
    Adirondack/
        im0.png          # left
        im1.png          # right
        disp0.pfm        # PFM disparity, inf = invalid
        calib.txt
    Backpack/
        ...
```

**SceneFlow** is supported by `src/datasets/sceneflow.py` for pretraining.

## Documentation

| file | contents |
|---|---|
| `docs/baseline_comparison_results.md` | measured accuracy and precision at 110 / 160 / 280 mm |
| `docs/baseline_capture_protocol.md` | the fixed per-baseline capture session |
| `docs/jetson_deployment_results.md` | Nano results: speed, accuracy, cost model, power, end-to-end |
| `docs/orin_benchmark_plan.md` | Orin NX setup, flashing, and benchmark plan |

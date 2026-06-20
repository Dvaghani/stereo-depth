# StereoUNet — Module A: Depth-Map Generation Pipeline

A lightweight 2D U-Net stereo matching pipeline for the master's thesis
*"Depth Map Generation based on Application Specific Stereo-Vision"*.
Designed to run in real time on the Holybro X500's NVIDIA Jetson Nano
(128 CUDA cores, 4 GB VRAM) while still being trainable on standard GPUs.

This repository implements **Module A** from the concept presentation:

```
Left + Right  ─►  Siamese CNN  ─►  Correlation cost volume  ─►  2D U-Net
              ─►  Soft-argmin   ─►  Disparity (px)           ─►  Depth (m)
```

Modules B (YOLO labelling) and C (depth/label fusion) are intentionally
not included here — they will be added in subsequent sessions.

## Layout

```
stereo_unet/
├── configs/                 # YAML training configs (kitti, middlebury)
├── csrc/                    # CUDA/C++ correlation kernel
│   ├── stereo_corr.cpp
│   ├── stereo_corr_kernel.cu
│   └── setup.py
├── scripts/
│   ├── train.py             # end-to-end training loop
│   ├── eval.py              # KITTI/Middlebury metrics on a checkpoint
│   └── infer.py             # single stereo pair → disparity + depth
├── src/
│   ├── models/
│   │   ├── feature_extractor.py
│   │   ├── cost_volume.py     # PyTorch + CUDA ext wrapper
│   │   ├── unet_aggregator.py
│   │   └── stereo_unet.py     # top-level model
│   ├── datasets/
│   │   ├── kitti2015.py
│   │   ├── middlebury2014.py
│   │   └── transforms.py
│   └── utils/
│       ├── losses.py
│       ├── metrics.py         # EPE, D1-all, bad-N
│       └── io.py              # PFM, KITTI disp PNG, viz
├── checkpoints/             # (gitignored) saved weights
├── outputs/                 # (gitignored) inference results
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Install Python deps (workstation, not Jetson — see below for Jetson)
pip install -r requirements.txt

# 2. (Optional) Build the CUDA correlation kernel. Without this the model
#    will use a pure-PyTorch fallback and print a one-time warning.
cd csrc
python setup.py build_ext --inplace
cd ..

# 3. Train on KITTI 2015
python scripts/train.py --config configs/kitti.yaml

# 4. Evaluate
python scripts/eval.py --ckpt checkpoints/kitti/best.pt \
    --dataset kitti --data-root /path/to/kitti2015

# 5. Run on a custom stereo pair
python scripts/infer.py --ckpt checkpoints/kitti/best.pt \
    --left left.png --right right.png \
    --baseline 0.14 --focal 700 --out outputs/run1
```

## Dataset layout

**KITTI 2015** (download from cvlibs.net/datasets/kitti):

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

**Middlebury 2014** (download from vision.middlebury.edu/stereo):

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

## Design notes

**Why a 2D U-Net (not 3D)?**
State-of-the-art methods (PSMNet, IGEV-Stereo, FoundationStereo) use 3D
convolutions over a 4D cost volume `(B, C, D, H, W)`. On a Jetson Nano this
exceeds the 4 GB VRAM budget at any usable resolution. We instead build a
*compressed* 3D volume `(B, D, H, W)` via dot-product correlation, treat the
disparity dimension as channels, and aggregate with a 2D U-Net.
This is the explicit trade-off described on slide 16 of the concept
presentation.

**Why an L2-normalized feature space?**
Dot-product correlation on L2-normalized features behaves like a cosine
similarity, which keeps the cost-volume range stable in `[-1, 1]` and makes
the softmax in the disparity head numerically well-behaved without
ad-hoc temperature tuning.

**Why a multi-scale loss?**
The auxiliary low-resolution loss (`disparity_low`) gives a direct gradient
to the cost-volume aggregator, which speeds up convergence vs. only
supervising the upsampled output.

**Validation metrics (slide 25 of the concept):**
- `EPE` — End-point error (mean absolute disparity error)
- `D1-all` — % pixels with `|err| > 3 px AND |err| / gt > 5 %`
  (target: `< 12 %`)
- `bad-1`, `bad-2`, `bad-3` — fraction over respective thresholds

## Jetson Nano deployment

The Python pipeline is the *training and validation* environment. For
deployment on the X500's Jetson Nano mission companion board:

1. Build the CUDA correlation extension with `TORCH_CUDA_ARCH_LIST="5.3"`
   (Maxwell). The kernel in `csrc/stereo_corr_kernel.cu` is intentionally
   simple and register-friendly so it fits sm_53 occupancy.
2. Export the trained model to ONNX, then convert to TensorRT (this is
   recommended by the thesis on slide 18 for Module B as well).
3. Replace the PyTorch `forward` path with the TensorRT engine; the CUDA
   correlation kernel can either be a TensorRT custom plugin or kept as a
   raw CUDA call before/after the TRT engine.
4. Configure stereo baseline based on the chosen X500 mounting (slides
   21-23: 80 mm / 140 mm / 280 mm) and pass it as `--baseline` to
   `infer.py`.

## Roadmap (next sessions)

- [ ] Module B: YOLOv11-Nano integration on the left RGB stream
- [ ] Module C: pixel-wise depth ↔ label fusion → semantically labeled point cloud
- [ ] ONNX export + TensorRT engine build script
- [ ] Self-supervised photometric loss for online adaptation (LWANet/MADNet style)
- [ ] Domain randomization / SceneFlow pretraining

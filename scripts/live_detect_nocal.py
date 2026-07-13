"""
Module B — Live detection + depth, NO calibration file needed.

Uses hardcoded focal length + baseline. Skips rectification.
Good for quick testing without a stereo_calib.npz.

Usage:
    python scripts/live_detect_nocal.py \
        --ckpt checkpoints/middlebury_aanet_v2_uncertainty/best.pt \
        --focal 1247.2 \
        --baseline 0.160

Keys:
    q / ESC  — quit
    s        — save current frame
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.models.aanet import AANetWrapper
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD

YOLO_CONF = 0.35
CLASS_COLORS = {
    "person":   (255,  80,  80),
    "bed":      ( 80, 200,  80),
    "chair":    ( 80,  80, 255),
    "suitcase": (255, 180,  50),
    "backpack": (180,  80, 255),
    "laptop":   ( 50, 200, 200),
    "desk":     (255, 100, 180),
    "table":    (100, 255, 180),
    "door":     (200, 200,  50),
    "tv":       ( 50, 220, 220),
}
PALETTE = [(255,80,80),(80,200,80),(80,80,255),(255,180,50),
           (180,80,255),(50,200,200),(255,100,180),(100,255,180)]


def _set_ctrl(dev, name, val):
    try:
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", f"{name}={val}"],
                       check=True, capture_output=True)
    except Exception:
        pass


def lock_focus(index, focus):
    dev = f"/dev/video{index}"
    _set_ctrl(dev, "focus_automatic_continuous", 0)
    _set_ctrl(dev, "focus_absolute", focus)


CMAPS = {"turbo": cv2.COLORMAP_TURBO, "plasma": cv2.COLORMAP_PLASMA,
         "jet": cv2.COLORMAP_JET}


def colorize_disparity(disp, max_disp, cmap="turbo", auto_norm=True):
    """auto_norm stretches contrast to the scene's actual disparity range."""
    if auto_norm:
        valid = disp[disp > 0]
        if len(valid) > 100:
            lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
            hi = max(hi, lo + 1.0)
        else:
            lo, hi = 0.0, float(max_disp)
    else:
        lo, hi = 0.0, float(max_disp)
    norm = np.clip((disp - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), CMAPS[cmap])


def compute_range_map(depth_full, cx, cy, focal_px):
    """Back-project depth (Z) to true Euclidean distance R = |X,Y,Z| per pixel."""
    H, W = depth_full.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    X = (uu - cx) * depth_full / focal_px
    Y = (vv - cy) * depth_full / focal_px
    R = np.sqrt(X**2 + Y**2 + depth_full**2)
    return np.where(depth_full > 0, R, 0.0)


def draw_detections(vis, results, depth_full, range_full, class_names):
    H, W = depth_full.shape
    for i, box in enumerate(results.boxes):
        cls  = int(box.cls)
        conf = float(box.conf)
        name = class_names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(W-1, x2); y2c = min(H-1, y2)

        roi   = depth_full[y1c:y2c, x1c:x2c]
        valid = roi[roi > 0]
        dist_med = float(np.median(valid)) if len(valid) else 0.0
        dist_min = float(np.min(valid))    if len(valid) else 0.0

        roi_r   = range_full[y1c:y2c, x1c:x2c]
        valid_r = roi_r[roi_r > 0]
        range_min = float(np.min(valid_r)) if len(valid_r) else 0.0

        color = CLASS_COLORS.get(name, PALETTE[i % len(PALETTE)])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        line1 = f"{name} {conf:.0%}"
        line2 = f"Z: med {dist_med:.2f}m  near {dist_min:.2f}m"
        line3 = f"true dist: {range_min:.2f}m"
        font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        (tw1, th), _ = cv2.getTextSize(line1, font, fs, ft)
        (tw2,  _), _ = cv2.getTextSize(line2, font, fs, ft)
        (tw3,  _), _ = cv2.getTextSize(line3, font, fs, ft)
        tw = max(tw1, tw2, tw3)
        bh = th * 3 + 20
        cv2.rectangle(vis, (x1, y1 - bh), (x1 + tw + 8, y1), color, -1)
        cv2.putText(vis, line1, (x1+4, y1 - 2*th - 12), font, fs, (255,255,255), ft)
        cv2.putText(vis, line2, (x1+4, y1 - th - 6),    font, fs, (255,255,255), ft)
        cv2.putText(vis, line3, (x1+4, y1 - 4),         font, fs, (255,255,255), ft)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--focal",       type=float, default=1247.2,
                   help="Focal length in pixels. Brio spec=1101, our calibrated=1247.")
    p.add_argument("--baseline",    type=float, default=0.160,
                   help="Stereo baseline in metres.")
    p.add_argument("--width",       type=int, default=1920)
    p.add_argument("--height",      type=int, default=1080)
    p.add_argument("--cx", type=float, default=None,
                   help="Principal point x (px). Default: width / 2.")
    p.add_argument("--cy", type=float, default=None,
                   help="Principal point y (px). Default: height / 2.")
    p.add_argument("--yolo",        default="yolo11n.pt")
    p.add_argument("--left-index",  type=int, default=0)
    p.add_argument("--right-index", type=int, default=4)
    p.add_argument("--scale",       type=float, default=0.5)
    p.add_argument("--max-disp",    type=int, default=192)
    p.add_argument("--conf",        type=float, default=YOLO_CONF)
    p.add_argument("--focus",       type=int, default=None)
    p.add_argument("--cmap",        choices=list(CMAPS), default="turbo",
                   help="Disparity colormap. turbo: near=red, far=blue.")
    p.add_argument("--fixed-norm",  action="store_true",
                   help="Normalize colors by 0..max_disp instead of the scene's actual range.")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    cap_w, cap_h = args.width, args.height
    cx_px = args.cx if args.cx is not None else cap_w / 2.0
    cy_px = args.cy if args.cy is not None else cap_h / 2.0

    print(f"Focal={args.focal}px  Baseline={args.baseline*1000:.0f}mm  "
          f"Resolution={cap_w}x{cap_h}  (no rectification)")

    # ── Load AANet ────────────────────────────────────────────────────────────
    print("Loading depth model...")
    state    = torch.load(args.ckpt, map_location=device)
    ckpt_cfg = state.get("config", {}) if isinstance(state, dict) else {}
    max_disp = ckpt_cfg.get("max_disp", args.max_disp)
    pred_unc = ckpt_cfg.get("predict_uncertainty", False)
    model = AANetWrapper(max_disp=max_disp, predict_uncertainty=pred_unc).to(device).eval()
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    if not any(k.startswith("backbone.") for k in sd):
        sd = {"backbone." + k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    # ── Load YOLO ─────────────────────────────────────────────────────────────
    print("Loading YOLO...")
    yolo = YOLO(args.yolo)

    # ── Cameras ───────────────────────────────────────────────────────────────
    def open_cam(idx):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {idx}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cap_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
        for _ in range(5): cap.read()
        return cap

    capL = open_cam(args.left_index)
    capR = open_cam(args.right_index)

    if args.focus is not None:
        print(f"Locking focus={args.focus}")
        lock_focus(args.left_index,  args.focus)
        lock_focus(args.right_index, args.focus)
        for _ in range(10):
            capL.read(); capR.read()

    mean_t    = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std_t     = torch.tensor(IMAGENET_STD,  device=device).view(3, 1, 1)
    focal_eff = args.focal * args.scale

    save_dir = Path("outputs/live_detect"); save_dir.mkdir(parents=True, exist_ok=True)
    fps_t = time.time()

    print("Live detection running — q/ESC to quit, s to save")
    print("NOTE: no rectification — depth accuracy lower than with calibration")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("Camera read failed"); break

        # No rectification — use raw frames directly
        def to_t(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t   = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
            return ((t - mean_t) / std_t).unsqueeze(0)

        L = F.interpolate(to_t(frameL), scale_factor=args.scale, mode="bilinear", align_corners=False)
        R = F.interpolate(to_t(frameR), scale_factor=args.scale, mode="bilinear", align_corners=False)
        H, W = L.shape[-2:]
        pH = (64 - H % 64) % 64
        pW = (64 - W % 64) % 64
        L = F.pad(L, (0, pW, 0, pH), mode="reflect")
        R = F.pad(R, (0, pW, 0, pH), mode="reflect")

        with torch.no_grad():
            out  = model(L, R)
            disp = out["disparity"][0, :H, :W].cpu().numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            depth_s = np.where(disp > 0, args.baseline * focal_eff / disp, 0.0)
        depth_full = cv2.resize(depth_s, (cap_w, cap_h), interpolation=cv2.INTER_LINEAR)
        range_full = compute_range_map(depth_full, cx_px, cy_px, args.focal)

        # YOLO
        yolo_res    = yolo(frameL, conf=args.conf, verbose=False)[0]
        class_names = yolo_res.names

        # Disparity colormap — boxes drawn on this
        disp_color = colorize_disparity(disp, float(max_disp),
                                        cmap=args.cmap, auto_norm=not args.fixed_norm)
        disp_color = cv2.resize(disp_color, (cap_w, cap_h))
        draw_detections(disp_color, yolo_res, depth_full, range_full, class_names)

        now = time.time()
        fps = 1.0 / max(now - fps_t, 1e-6)
        fps_t = now
        cv2.putText(disp_color, f"{fps:.1f} FPS  f={args.focal:.0f}px  b={args.baseline*1000:.0f}mm",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4)
        cv2.putText(disp_color, f"{fps:.1f} FPS  f={args.focal:.0f}px  b={args.baseline*1000:.0f}mm",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Clean RGB | disparity with boxes
        display = np.hstack([frameL, disp_color])
        display = cv2.resize(display, (display.shape[1]//2, display.shape[0]//2))
        cv2.imshow("RGB  |  Disparity + Detection  (no rectification)", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(save_dir / f"rgb_{ts}.png"),    frameL)
            cv2.imwrite(str(save_dir / f"detect_{ts}.png"), disp_color)
            print(f"Saved → {save_dir}/detect_{ts}.png")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

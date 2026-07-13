"""
Module B — Object detection + depth fusion.

Runs YOLOv11-Nano on the left image and fuses with the AANet disparity map
to label each detected object with its median distance in metres.

Usage:
    python scripts/detect_depth.py \
        --ckpt checkpoints/middlebury_aanet_v2_uncertainty/best.pt \
        --left  outputs/capture_160mm_20260608_131946/left.png \
        --right outputs/capture_160mm_20260608_131946/right.png \
        --baseline 0.1606 --focal 1247.2 --scale 0.5 \
        --out   outputs/detect_depth/test1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.models.aanet import AANetWrapper
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD


CONF_THRESH = 0.35   # YOLO confidence threshold
CMAPS = {"turbo": cv2.COLORMAP_TURBO, "plasma": cv2.COLORMAP_PLASMA,
         "jet": cv2.COLORMAP_JET}
COLORS = [
    (255, 80,  80),  (80,  200, 80),  (80,  80,  255),
    (255, 180, 50),  (180, 80,  255), (50,  200, 200),
    (255, 100, 180), (100, 255, 180), (200, 200, 50),
]


def load_depth_model(ckpt: str, device: torch.device):
    state = torch.load(ckpt, map_location=device)
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    max_disp = cfg.get("max_disp", 192)
    predict_unc = cfg.get("predict_uncertainty", False)
    model = AANetWrapper(max_disp=max_disp, predict_uncertainty=predict_unc).to(device).eval()
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    if not any(k.startswith("backbone.") for k in sd):
        sd = {"backbone." + k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model, max_disp


def run_depth(model, left_bgr, right_bgr, scale, device):
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std_t  = torch.tensor(IMAGENET_STD,  device=device).view(3, 1, 1)

    def to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
        return ((t - mean_t) / std_t).unsqueeze(0)

    L = F.interpolate(to_tensor(left_bgr),  scale_factor=scale, mode="bilinear", align_corners=False)
    R = F.interpolate(to_tensor(right_bgr), scale_factor=scale, mode="bilinear", align_corners=False)

    H, W = L.shape[-2:]
    pH = (64 - H % 64) % 64
    pW = (64 - W % 64) % 64
    L = F.pad(L, (0, pW, 0, pH), mode="reflect")
    R = F.pad(R, (0, pW, 0, pH), mode="reflect")

    with torch.no_grad():
        out = model(L, R)
    return out["disparity"][0, :H, :W].cpu().numpy()


def compute_range_map(depth_full, cx, cy, focal_px):
    """Back-project depth (Z) to true Euclidean distance R = |X,Y,Z| per pixel."""
    H, W = depth_full.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    X = (uu - cx) * depth_full / focal_px
    Y = (vv - cy) * depth_full / focal_px
    R = np.sqrt(X**2 + Y**2 + depth_full**2)
    return np.where(depth_full > 0, R, 0.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--left",  required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--out",   default="outputs/detect_depth")
    p.add_argument("--baseline", type=float, default=0.1606)
    p.add_argument("--focal",    type=float, default=1247.2)
    p.add_argument("--cx", type=float, default=None,
                   help="Principal point x (px). Default: image width / 2.")
    p.add_argument("--cy", type=float, default=None,
                   help="Principal point y (px). Default: image height / 2.")
    p.add_argument("--scale",    type=float, default=0.5)
    p.add_argument("--yolo",     default="yolo11n.pt")
    p.add_argument("--conf",     type=float, default=CONF_THRESH)
    p.add_argument("--cmap",     choices=list(CMAPS), default="turbo",
                   help="Disparity colormap. turbo: near=red, far=blue.")
    p.add_argument("--fixed-norm", action="store_true",
                   help="Normalize colors by 0..max_disp instead of the scene's actual range.")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading depth model...")
    depth_model, max_disp = load_depth_model(args.ckpt, device)

    print("Loading YOLO...")
    yolo = YOLO(args.yolo)

    # ── Load images ───────────────────────────────────────────────────────────
    left_bgr  = cv2.imread(args.left)
    right_bgr = cv2.imread(args.right)
    if left_bgr is None or right_bgr is None:
        raise SystemExit("Could not read input images.")
    H_full, W_full = left_bgr.shape[:2]

    # ── Run depth ─────────────────────────────────────────────────────────────
    print("Running depth estimation...")
    focal_eff = args.focal * args.scale
    disp = run_depth(depth_model, left_bgr, right_bgr, args.scale, device)

    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disp > 0, args.baseline * focal_eff / disp, 0.0)

    # Scale disparity/depth back to full image size for bbox lookup
    depth_full = cv2.resize(depth, (W_full, H_full), interpolation=cv2.INTER_LINEAR)

    # True 3D Euclidean distance (depth_full is only the Z coordinate)
    cx = args.cx if args.cx is not None else W_full / 2.0
    cy = args.cy if args.cy is not None else H_full / 2.0
    range_full = compute_range_map(depth_full, cx, cy, args.focal)

    # ── Run YOLO ──────────────────────────────────────────────────────────────
    print("Running detection...")
    results = yolo(left_bgr, conf=args.conf, verbose=False)[0]

    # ── Colorized disparity (boxes drawn on this) ─────────────────────────────
    disp_full = cv2.resize(disp, (W_full, H_full))
    if args.fixed_norm:
        lo, hi = 0.0, float(max_disp)
    else:
        # Stretch contrast to the scene's actual disparity range — far scenes
        # only occupy a fraction of max_disp and look flat with fixed norm.
        valid_d = disp_full[disp_full > 0]
        lo = np.percentile(valid_d, 2) if len(valid_d) > 100 else 0.0
        hi = np.percentile(valid_d, 98) if len(valid_d) > 100 else float(max_disp)
        hi = max(hi, lo + 1.0)
    norm = np.clip((disp_full - lo) / (hi - lo), 0, 1)
    disp_color = cv2.applyColorMap((norm * 255).astype(np.uint8), CMAPS[args.cmap])

    # ── Fuse: median depth inside each bbox ───────────────────────────────────
    detections = []

    for i, box in enumerate(results.boxes):
        cls  = int(box.cls)
        conf = float(box.conf)
        name = results.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W_full-1, x2), min(H_full-1, y2)

        roi = depth_full[y1c:y2c, x1c:x2c]
        valid = roi[roi > 0]
        dist_median = float(np.median(valid)) if len(valid) > 0 else 0.0
        dist_min    = float(np.min(valid))    if len(valid) > 0 else 0.0

        roi_r = range_full[y1c:y2c, x1c:x2c]
        valid_r = roi_r[roi_r > 0]
        range_min = float(np.min(valid_r)) if len(valid_r) > 0 else 0.0

        detections.append({"label": name, "conf": conf,
                            "dist_median_m": dist_median, "dist_min_m": dist_min,
                            "range_min_m": range_min,
                            "box": (x1, y1, x2, y2)})

        color = COLORS[i % len(COLORS)]

        # Draw bbox on disparity map
        cv2.rectangle(disp_color, (x1, y1), (x2, y2), color, 2)

        line1 = f"{name} ({conf:.0%})"
        line2 = f"Z: med {dist_median:.2f}m  near {dist_min:.2f}m"
        line3 = f"true dist: {range_min:.2f}m"
        font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (tw1, th), _ = cv2.getTextSize(line1, font, fs, ft)
        (tw2,  _), _ = cv2.getTextSize(line2, font, fs, ft)
        (tw3,  _), _ = cv2.getTextSize(line3, font, fs, ft)
        tw = max(tw1, tw2, tw3)
        box_h = th * 3 + 20
        cv2.rectangle(disp_color, (x1, y1 - box_h), (x1 + tw + 8, y1), color, -1)
        cv2.putText(disp_color, line1, (x1 + 4, y1 - 2*th - 12), font, fs, (255, 255, 255), ft)
        cv2.putText(disp_color, line2, (x1 + 4, y1 - th - 6),    font, fs, (255, 255, 255), ft)
        cv2.putText(disp_color, line3, (x1 + 4, y1 - 4),         font, fs, (255, 255, 255), ft)

    # ── Save outputs ──────────────────────────────────────────────────────────
    cv2.imwrite(str(out_dir / "detection_depth.png"), disp_color)
    cv2.imwrite(str(out_dir / "rgb.png"), left_bgr)

    # Side by side: RGB (clean) | disparity with boxes
    side = np.hstack([left_bgr, disp_color])
    cv2.imwrite(str(out_dir / "side_by_side.png"), side)

    print(f"\nDetected {len(detections)} object(s):")
    for d in detections:
        print(f"  {d['label']:15s}  conf={d['conf']:.2f}  Z_median={d['dist_median_m']:.2f}m  "
              f"Z_nearest={d['dist_min_m']:.2f}m  true_dist={d['range_min_m']:.2f}m")
    print(f"\nSaved → {out_dir}/detection_depth.png  disparity.png  side_by_side.png")


if __name__ == "__main__":
    main()

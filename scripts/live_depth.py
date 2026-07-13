"""
Live stereo depth map from Brio rig.

Captures rectified stereo pairs in real time, runs AANet inference,
and displays RGB | disparity side-by-side at ~2-5 FPS on CPU.

Usage:
    python scripts/live_depth.py \
        --ckpt checkpoints/middlebury_aanet_v2_uncertainty/best.pt \
        --calib outputs/calibration_160mm/stereo_calib.npz

Keys:
    q / ESC  — quit
    s        — save current frame to outputs/live_capture/
    +/-      — increase / decrease confidence threshold
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.models.aanet import AANetWrapper
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD


CONF_THRESHOLD = 8.0   # px — pixels with b < this are "confident"


def _set_ctrl(dev: str, name: str, val) -> None:
    try:
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", f"{name}={val}"],
                       check=True, capture_output=True)
    except Exception:
        pass


def lock_focus(index: int, focus: int) -> None:
    dev = f"/dev/video{index}"
    _set_ctrl(dev, "focus_automatic_continuous", 0)
    _set_ctrl(dev, "focus_absolute", focus)


def read_calib_focus(calib_path: Path) -> int | None:
    focus_file = calib_path.parent / "focus.txt"
    try:
        return int(focus_file.read_text().strip())
    except Exception:
        return None


CMAPS = {"turbo": cv2.COLORMAP_TURBO, "plasma": cv2.COLORMAP_PLASMA,
         "jet": cv2.COLORMAP_JET}


def colorize_disparity(disp_np: np.ndarray, max_disp: float,
                       cmap: str = "turbo", auto_norm: bool = True) -> np.ndarray:
    """auto_norm stretches contrast to the scene's actual disparity range
    (2-98 percentile) — far scenes only occupy a fraction of max_disp and
    look flat with fixed normalization."""
    if auto_norm:
        valid = disp_np[disp_np > 0]
        if len(valid) > 100:
            lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
            hi = max(hi, lo + 1.0)
        else:
            lo, hi = 0.0, max_disp
    else:
        lo, hi = 0.0, max_disp
    norm = np.clip((disp_np - lo) / (hi - lo), 0, 1)
    heat = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(heat, CMAPS[cmap])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--left-index",  type=int, default=0)
    p.add_argument("--right-index", type=int, default=4)
    p.add_argument("--scale", type=float, default=0.5,
                   help="Downsample before inference (0.5 = half-res, faster).")
    p.add_argument("--max-disp", type=int, default=192)
    p.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD,
                   help="b < threshold = confident pixel (press +/- to adjust live).")
    p.add_argument("--cmap", choices=list(CMAPS), default="turbo",
                   help="Disparity colormap. turbo: near=red, far=blue.")
    p.add_argument("--fixed-norm", action="store_true",
                   help="Normalize colors by 0..max_disp instead of the scene's actual range.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--focus", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    conf_thresh = args.conf_threshold

    # ── Load calibration ──────────────────────────────────────────────────────
    if not args.calib.exists():
        raise SystemExit(f"Calibration not found: {args.calib}")
    calib = np.load(args.calib)
    map1L, map2L = calib["map1L"], calib["map2L"]
    map1R, map2R = calib["map1R"], calib["map2R"]
    img_size = tuple(int(x) for x in calib["image_size"])  # (W, H)
    baseline_m = float(calib["baseline_mm"][0]) / 1000.0
    focal_px   = float(calib["focal_px"][0])
    cap_w, cap_h = img_size

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    state    = torch.load(args.ckpt, map_location=device)
    ckpt_cfg = state.get("config", {}) if isinstance(state, dict) else {}
    predict_unc = ckpt_cfg.get("predict_uncertainty", False)
    max_disp    = ckpt_cfg.get("max_disp", args.max_disp)

    model = AANetWrapper(max_disp=max_disp, predict_uncertainty=predict_unc).to(device).eval()
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    if not any(k.startswith("backbone.") for k in sd):
        sd = {"backbone." + k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"Model ready (uncertainty={predict_unc}, device={device})")

    # ── Open cameras ─────────────────────────────────────────────────────────
    focus = args.focus if args.focus is not None else read_calib_focus(args.calib)

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

    if focus is not None:
        print(f"Locking focus to {focus}")
        lock_focus(args.left_index,  focus)
        lock_focus(args.right_index, focus)
        for _ in range(10):
            capL.read(); capR.read()

    # ── ImageNet normalisation tensors ────────────────────────────────────────
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std_t  = torch.tensor(IMAGENET_STD,  device=device).view(3, 1, 1)

    save_dir = Path("outputs/live_capture")
    save_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    fps_t = time.time()
    fps_str = "-- FPS"

    print("Live depth running — press 'q'/ESC to quit, 's' to save, +/- to adjust confidence")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("Camera read failed"); break

        # Rectify
        rectL = cv2.remap(frameL, map1L, map2L, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, map1R, map2R, cv2.INTER_LINEAR)

        # BGR→RGB tensor, normalise
        def to_tensor(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
            return ((t - mean_t) / std_t).unsqueeze(0)

        L = to_tensor(rectL)
        R = to_tensor(rectR)

        # Downsample
        if args.scale != 1.0:
            L = F.interpolate(L, scale_factor=args.scale, mode="bilinear", align_corners=False)
            R = F.interpolate(R, scale_factor=args.scale, mode="bilinear", align_corners=False)
        focal_eff = focal_px * args.scale

        # Pad to multiple of 64
        H, W = L.shape[-2:]
        pH = (64 - H % 64) % 64
        pW = (64 - W % 64) % 64
        L = F.pad(L, (0, pW, 0, pH), mode="reflect")
        R = F.pad(R, (0, pW, 0, pH), mode="reflect")

        with torch.no_grad():
            out  = model(L, R)
            disp = out["disparity"][0, :H, :W].cpu().numpy()
            log_b = out.get("log_b")
            if log_b is not None:
                b_np = np.exp(log_b[0, :H, :W].clamp(-5, 5).cpu().numpy())
            else:
                b_np = None

        # Depth in metres
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(disp > 0, baseline_m * focal_eff / disp, 0.0)

        # Colorize disparity — PLASMA: high disp (near) = orange, low disp (far) = purple
        depth_color = colorize_disparity(disp, float(max_disp),
                                         cmap=args.cmap, auto_norm=not args.fixed_norm)
        depth_color = cv2.resize(depth_color, (rectL.shape[1], rectL.shape[0]))

        # Confidence stats only — no masking on middle panel
        if b_np is not None:
            conf_pct = 100.0 * (b_np < conf_thresh).mean()
        else:
            conf_pct = 100.0

        # FPS
        now = time.time()
        fps_str = f"{1.0/(now - fps_t):.1f} FPS"
        fps_t = now

        # Overlay text
        cv2.putText(depth_color, fps_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(depth_color, f"conf<{conf_thresh:.0f}px: {conf_pct:.0f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Confidence map (b values — bright = confident/low b, dark = uncertain/high b)
        # 2-panel display: RGB | Color disparity
        display = np.hstack([rectL, depth_color])
        display = cv2.resize(display, (display.shape[1]//2, display.shape[0]//2))
        cv2.imshow("RGB  |  Color Disparity", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(save_dir / f"left_{ts}.png"), rectL)
            cv2.imwrite(str(save_dir / f"right_{ts}.png"), rectR)
            cv2.imwrite(str(save_dir / f"disp_{ts}.png"), depth_color)
            print(f"Saved → {save_dir}/left_{ts}.png  right_{ts}.png  disp_{ts}.png")
            print(f"  Run: .venv/bin/python scripts/infer.py --ckpt checkpoints/middlebury_aanet_v2_uncertainty/best.pt "
                  f"--left {save_dir}/left_{ts}.png --right {save_dir}/right_{ts}.png "
                  f"--baseline 0.1606 --focal 1247.2 --scale 0.5 --out outputs/capture_{ts}")
            frame_idx += 1
        elif key == ord('+') or key == ord('='):
            conf_thresh = min(conf_thresh + 1.0, 50.0)
            print(f"Confidence threshold: {conf_thresh:.0f}px")
        elif key == ord('-'):
            conf_thresh = max(conf_thresh - 1.0, 1.0)
            print(f"Confidence threshold: {conf_thresh:.0f}px")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

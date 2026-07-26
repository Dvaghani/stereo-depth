"""
Module B — Live object detection + depth fusion from Brio stereo rig.

Runs stereo depth + YOLOv11-Nano on every frame. YOLO detections are overlaid
on the disparity map with median and nearest distance labels.

Two stereo backends:
    --backend raft    RAFT-Stereo realtime — robust zero-shot, works on any
                      scene/camera out of the box (default)
    --backend aanet   our trained AANet + uncertainty (needs --ckpt)

Usage:
    python scripts/live_detect.py \
        --calib outputs/calibration_160mm/stereo_calib.npz

    python scripts/live_detect.py --backend aanet \
        --ckpt  checkpoints/middlebury_aanet_v2_uncertainty/best.pt \
        --calib outputs/calibration_160mm/stereo_calib.npz

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

YOLO_CONF   = 0.35
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


def read_calib_focus(calib_path):
    try:
        return int((calib_path.parent / "focus.txt").read_text().strip())
    except Exception:
        return None


CMAPS = {"turbo": cv2.COLORMAP_TURBO, "plasma": cv2.COLORMAP_PLASMA,
         "jet": cv2.COLORMAP_JET}


def colorize_disparity(disp, max_disp, cmap="turbo", auto_norm=True):
    """Colorize disparity. auto_norm stretches contrast to the scene's actual
    disparity range (2-98 percentile) instead of the fixed 0..max_disp range —
    far scenes only use a fraction of max_disp and would look flat otherwise."""
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


def draw_detections(vis, results, depth_full, range_full, class_names,
                    b_full=None, b_thresh=2.0):
    """b_full: per-pixel Laplace scale (uncertainty). When given, distance
    statistics use only confident pixels (b < b_thresh) if enough exist,
    and the label shows the confident fraction inside the box."""
    H, W = depth_full.shape
    for i, box in enumerate(results.boxes):
        cls  = int(box.cls)
        conf = float(box.conf)
        name = class_names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(W-1, x2); y2c = min(H-1, y2)

        roi  = depth_full[y1c:y2c, x1c:x2c]
        mask = roi > 0
        conf_pct = None
        if b_full is not None:
            b_roi = b_full[y1c:y2c, x1c:x2c]
            conf_mask = mask & (b_roi < b_thresh)
            conf_pct = 100.0 * conf_mask.mean() if conf_mask.size else 0.0
            # prefer confident pixels for the distance stats when enough exist
            if conf_mask.sum() > 0.1 * max(mask.sum(), 1):
                mask = conf_mask
        valid = roi[mask]
        dist_med = float(np.median(valid)) if len(valid) else 0.0
        dist_min = float(np.min(valid))    if len(valid) else 0.0

        roi_r   = range_full[y1c:y2c, x1c:x2c][mask]
        valid_r = roi_r[roi_r > 0]
        range_min = float(np.min(valid_r)) if len(valid_r) else 0.0

        color = CLASS_COLORS.get(name, PALETTE[i % len(PALETTE)])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        line1 = f"{name} {conf:.0%}"
        if conf_pct is not None:
            line1 += f"  conf {conf_pct:.0f}%"
        line2 = f"Z: med {dist_med:.2f}m  near {dist_min:.2f}m"
        line3 = f"true dist: {range_min:.2f}m"
        font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        (tw1, th), _ = cv2.getTextSize(line1, font, fs, ft)
        (tw2,  _), _ = cv2.getTextSize(line2, font, fs, ft)
        (tw3,  _), _ = cv2.getTextSize(line3, font, fs, ft)
        tw  = max(tw1, tw2, tw3)
        bh  = th * 3 + 20
        cv2.rectangle(vis, (x1, y1 - bh), (x1 + tw + 8, y1), color, -1)
        cv2.putText(vis, line1, (x1+4, y1 - 2*th - 12), font, fs, (255,255,255), ft)
        cv2.putText(vis, line2, (x1+4, y1 - th - 6),    font, fs, (255,255,255), ft)
        cv2.putText(vis, line3, (x1+4, y1 - 4),         font, fs, (255,255,255), ft)


def load_raft(device, ckpt=None):
    """Load RAFT-Stereo realtime-config model from third_party.
    ckpt=None uses the pretrained realtime weights; pass a path to use a
    fine-tuned checkpoint. Checkpoints containing an uncertainty head
    (unc_head.* keys) are detected automatically and loaded as the
    uncertainty-aware variant. Returns (model, InputPadder, has_uncertainty)."""
    _raft_dir = _HERE.parent / "third_party" / "RAFT-Stereo"
    sys.path.insert(0, str(_raft_dir))
    sys.path.insert(0, str(_raft_dir / "core"))
    from utils.utils import InputPadder

    ckpt_path = Path(ckpt) if ckpt else _raft_dir / "models" / "raftstereo-realtime.pth"
    print(f"RAFT checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    has_unc = any(k.startswith("unc_head") for k in sd)

    if has_unc:
        from src.models.raft_uncertainty import build_raft_uncertainty, realtime_args
        print("Uncertainty head detected — confidence-aware mode.")
        model = build_raft_uncertainty(realtime_args(mixed_precision=False))
    else:
        from raft_stereo import RAFTStereo
        raft_args = argparse.Namespace(
            hidden_dims=[128, 128, 128],
            corr_implementation="reg",
            corr_levels=4, corr_radius=4,
            context_norm="batch",
            # fp16 produces NaN and is slower on GTX 16xx GPUs — keep fp32
            mixed_precision=False,
            shared_backbone=True, n_downsample=3,
            n_gru_layers=2, slow_fast_gru=True,
        )
        model = RAFTStereo(raft_args)
    model.load_state_dict(sd)
    return model.to(device).eval(), InputPadder, has_unc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend",     choices=["raft", "aanet"], default="raft",
                   help="raft: robust zero-shot (default). aanet: our trained model.")
    p.add_argument("--ckpt",        default=None,
                   help="AANet checkpoint (required for --backend aanet). For "
                        "--backend raft: optional fine-tuned RAFT checkpoint.")
    p.add_argument("--calib",       type=Path, required=True)
    p.add_argument("--yolo",        default="yolo11n.pt")
    p.add_argument("--left-index",  type=int, default=0)
    p.add_argument("--right-index", type=int, default=4)
    p.add_argument("--scale",       type=float, default=0.5)
    p.add_argument("--iters",       type=int, default=7,
                   help="RAFT refinement iterations (raft backend only).")
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

    # ── Calibration ───────────────────────────────────────────────────────────
    if not args.calib.exists():
        raise SystemExit(f"Calibration not found: {args.calib}")
    calib      = np.load(args.calib)
    map1L, map2L = calib["map1L"], calib["map2L"]
    map1R, map2R = calib["map1R"], calib["map2R"]
    cap_w, cap_h = (int(x) for x in calib["image_size"])
    baseline_m   = float(calib["baseline_mm"][0]) / 1000.0
    focal_px     = float(calib["focal_px"][0])
    cx_px        = float(calib["P1"][0, 2])
    cy_px        = float(calib["P1"][1, 2])

    # ── Load stereo backend ───────────────────────────────────────────────────
    max_disp = args.max_disp
    has_unc = False
    if args.backend == "raft":
        print("Loading RAFT-Stereo (realtime)...")
        model, InputPadder, has_unc = load_raft(device, ckpt=args.ckpt)
    else:
        if not args.ckpt:
            raise SystemExit("--backend aanet requires --ckpt")
        print("Loading AANet...")
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
        print(f"Locking focus={focus}")
        lock_focus(args.left_index,  focus)
        lock_focus(args.right_index, focus)
        for _ in range(10):
            capL.read(); capR.read()

    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std_t  = torch.tensor(IMAGENET_STD,  device=device).view(3, 1, 1)
    focal_eff = focal_px * args.scale

    save_dir = Path("outputs/live_detect"); save_dir.mkdir(parents=True, exist_ok=True)
    fps_t = time.time()

    print("Live detection running — q/ESC to quit, s to save")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("Camera read failed"); break

        rectL = cv2.remap(frameL, map1L, map2L, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, map1R, map2R, cv2.INTER_LINEAR)

        # ── Depth inference ───────────────────────────────────────────────────
        b_full = None
        if args.backend == "raft":
            # RAFT takes raw 0-255 RGB floats, pad to /32
            w = int(cap_w * args.scale); h = int(cap_h * args.scale)
            smallL = cv2.resize(rectL, (w, h), interpolation=cv2.INTER_AREA)
            smallR = cv2.resize(rectR, (w, h), interpolation=cv2.INTER_AREA)

            def to_raft(bgr):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(device)

            t1, t2 = to_raft(smallL), to_raft(smallR)
            padder = InputPadder(t1.shape, divis_by=32)
            t1, t2 = padder.pad(t1, t2)
            with torch.no_grad():
                if has_unc:
                    _, flow_up, log_b = model(t1, t2, iters=args.iters, test_mode=True)
                    b_small = padder.unpad(log_b).squeeze().exp().cpu().numpy()
                    b_full = cv2.resize(b_small, (cap_w, cap_h), interpolation=cv2.INTER_LINEAR)
                else:
                    _, flow_up = model(t1, t2, iters=args.iters, test_mode=True)
            disp = -padder.unpad(flow_up).squeeze().cpu().numpy()
        else:
            def to_t(bgr):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                t   = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
                return ((t - mean_t) / std_t).unsqueeze(0)

            L = F.interpolate(to_t(rectL), scale_factor=args.scale, mode="bilinear", align_corners=False)
            R = F.interpolate(to_t(rectR), scale_factor=args.scale, mode="bilinear", align_corners=False)
            H, W = L.shape[-2:]
            pH = (64 - H % 64) % 64
            pW = (64 - W % 64) % 64
            L = F.pad(L, (0, pW, 0, pH), mode="reflect")
            R = F.pad(R, (0, pW, 0, pH), mode="reflect")

            with torch.no_grad():
                out  = model(L, R)
                disp = out["disparity"][0, :H, :W].cpu().numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            depth_s = np.where(disp > 0, baseline_m * focal_eff / disp, 0.0)
        depth_full = cv2.resize(depth_s, (cap_w, cap_h), interpolation=cv2.INTER_LINEAR)
        range_full = compute_range_map(depth_full, cx_px, cy_px, focal_px)

        # ── YOLO detection ────────────────────────────────────────────────────
        yolo_res    = yolo(rectL, conf=args.conf, verbose=False)[0]
        class_names = yolo_res.names

        # ── Disparity colormap — boxes drawn on this ──────────────────────────
        disp_color = colorize_disparity(disp, float(max_disp),
                                        cmap=args.cmap, auto_norm=not args.fixed_norm)
        disp_color = cv2.resize(disp_color, (cap_w, cap_h))
        draw_detections(disp_color, yolo_res, depth_full, range_full, class_names,
                        b_full=b_full)

        # FPS on disparity panel
        now = time.time()
        fps = 1.0 / max(now - fps_t, 1e-6)
        fps_t = now
        hud = f"{fps:.1f} FPS  [{args.backend}]"
        cv2.putText(disp_color, hud, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
        cv2.putText(disp_color, hud, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # ── Display: clean RGB | disparity with boxes ─────────────────────────
        display = np.hstack([rectL, disp_color])
        display = cv2.resize(display, (display.shape[1]//2, display.shape[0]//2))
        cv2.imshow("RGB  |  Disparity + Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(save_dir / f"rgb_{ts}.png"),    rectL)
            cv2.imwrite(str(save_dir / f"detect_{ts}.png"), disp_color)
            print(f"Saved → {save_dir}/detect_{ts}.png")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

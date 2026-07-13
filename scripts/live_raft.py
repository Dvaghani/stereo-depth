"""
Live depth from the Brio rig using RAFT-Stereo (teacher model).

Two presets:
    --model realtime    fast checkpoint, ~5-15 FPS on GPU (default)
    --model middlebury  accurate checkpoint, ~0.5 FPS — for quality reference

Usage:
    python scripts/live_raft.py \
        --calib outputs/calibration_110mm/stereo_calib_refreshed.npz

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

_HERE = Path(__file__).resolve().parent
_RAFT = _HERE.parent / "third_party" / "RAFT-Stereo"
sys.path.insert(0, str(_RAFT))
sys.path.insert(0, str(_RAFT / "core"))

from raft_stereo import RAFTStereo
from utils.utils import InputPadder

CMAPS = {"turbo": cv2.COLORMAP_TURBO, "plasma": cv2.COLORMAP_PLASMA,
         "jet": cv2.COLORMAP_JET}

PRESETS = {
    "realtime": dict(
        ckpt="raftstereo-realtime.pth",
        shared_backbone=True, n_downsample=3, n_gru_layers=2,
        slow_fast_gru=True, valid_iters=7,
    ),
    "middlebury": dict(
        ckpt="raftstereo-middlebury.pth",
        shared_backbone=False, n_downsample=2, n_gru_layers=3,
        slow_fast_gru=False, valid_iters=32,
    ),
}


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


def colorize_disparity(disp, cmap="turbo"):
    valid = disp[disp > 0]
    if len(valid) > 100:
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        hi = max(hi, lo + 1.0)
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((disp - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), CMAPS[cmap])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--calib",       type=Path, required=True)
    p.add_argument("--model",       choices=list(PRESETS), default="realtime")
    p.add_argument("--left-index",  type=int, default=0)
    p.add_argument("--right-index", type=int, default=4)
    p.add_argument("--scale",       type=float, default=0.5,
                   help="Downsample before inference (0.5 = half-res).")
    p.add_argument("--iters",       type=int, default=None,
                   help="Refinement iterations. Default: preset value.")
    p.add_argument("--cmap",        choices=list(CMAPS), default="turbo")
    p.add_argument("--focus",       type=int, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    preset = PRESETS[args.model]
    iters  = args.iters if args.iters is not None else preset["valid_iters"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU — RAFT-Stereo will be very slow on CPU.")

    # ── Calibration ───────────────────────────────────────────────────────────
    calib = np.load(args.calib)
    map1L, map2L = calib["map1L"], calib["map2L"]
    map1R, map2R = calib["map1R"], calib["map2R"]
    cap_w, cap_h = (int(x) for x in calib["image_size"])
    baseline_m   = float(calib["baseline_mm"][0]) / 1000.0
    focal_px     = float(calib["focal_px"][0])

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading RAFT-Stereo ({args.model}, {iters} iters)...")
    raft_args = argparse.Namespace(
        hidden_dims=[128, 128, 128],
        corr_implementation="reg",
        corr_levels=4,
        corr_radius=4,
        context_norm="batch",
        # fp16 produces NaN and is SLOWER on GTX 16xx GPUs — keep fp32
        mixed_precision=False,
        shared_backbone=preset["shared_backbone"],
        n_downsample=preset["n_downsample"],
        n_gru_layers=preset["n_gru_layers"],
        slow_fast_gru=preset["slow_fast_gru"],
    )
    model = RAFTStereo(raft_args)
    ckpt_path = _RAFT / "models" / preset["ckpt"]
    sd = torch.load(ckpt_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(device).eval()

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

    save_dir = Path("outputs/live_raft"); save_dir.mkdir(parents=True, exist_ok=True)
    fps_t = time.time()

    print("Live RAFT-Stereo running — q/ESC to quit, s to save")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("Camera read failed"); break

        rectL = cv2.remap(frameL, map1L, map2L, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, map1R, map2R, cv2.INTER_LINEAR)

        # Downsample and convert — RAFT takes raw 0-255 RGB floats
        w = int(cap_w * args.scale); h = int(cap_h * args.scale)
        smallL = cv2.resize(rectL, (w, h), interpolation=cv2.INTER_AREA)
        smallR = cv2.resize(rectR, (w, h), interpolation=cv2.INTER_AREA)

        def to_t(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(device)

        t1, t2 = to_t(smallL), to_t(smallR)
        padder = InputPadder(t1.shape, divis_by=32)
        t1, t2 = padder.pad(t1, t2)

        with torch.no_grad():
            _, flow_up = model(t1, t2, iters=iters, test_mode=True)
        disp = -padder.unpad(flow_up).squeeze().cpu().numpy()

        disp_color = colorize_disparity(disp, cmap=args.cmap)
        disp_color = cv2.resize(disp_color, (cap_w, cap_h))

        # FPS
        now = time.time()
        fps = 1.0 / max(now - fps_t, 1e-6)
        fps_t = now
        cv2.putText(disp_color, f"{fps:.1f} FPS  RAFT-{args.model}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
        cv2.putText(disp_color, f"{fps:.1f} FPS  RAFT-{args.model}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        display = np.hstack([rectL, disp_color])
        display = cv2.resize(display, (display.shape[1]//2, display.shape[0]//2))
        cv2.imshow("RGB  |  RAFT-Stereo Disparity", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(save_dir / f"rgb_{ts}.png"),  rectL)
            cv2.imwrite(str(save_dir / f"disp_{ts}.png"), disp_color)
            np.save(str(save_dir / f"disp_{ts}.npy"), disp)
            print(f"Saved → {save_dir}/disp_{ts}.png")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

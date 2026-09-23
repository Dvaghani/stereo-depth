"""
RAFT-Stereo accuracy on ordinary objects, against a laser-rangefinder reference.

Why this exists
----------------
eval_depth_accuracy.py and eval_raft_stations.py both evaluate on the
checkerboard alone -- a high-contrast, planar, well-textured target, which is
close to the best case for a stereo matcher. The thesis already flags this as
a limitation: those figures understate RAFT's error on the textureless and
thin structures a checkerboard cannot represent. This script closes that gap
by sampling RAFT's dense disparity at a point on any real object, and
comparing it against a laser-rangefinder reading of the same point instead of
a tape measure -- a laser is accurate to ~1-2 mm, so unlike the station
protocol (limited by ~25 mm tape/board-placement repeatability, see
Table tab:distance's discussion), the error reported here is not itself
floored by the reference.

Workflow (repeat per object)
-----------------------------
1. Aim the laser at one identifiable point on the object (a corner, a sticker,
   a printed mark -- anything you can click precisely in the image) and note
   the reading.
2. Capture a stereo pair with the rig pointed at the same scene:
       python scripts/capture_stereo.py --calib <calib.npz>
3. Run this script on that pair. A window opens showing the left rectified
   image with a live depth readout following the cursor -- move to the exact
   point the laser was aimed at, left-click to sample it, and confirm.
4. Enter the laser reading (metres) when prompted, or pass --laser-m to skip
   the prompt.

Each run appends one row to --log (created with a header if new) and reprints
the running table so accuracy across objects accumulates automatically. An
annotated snapshot (crosshair + both readings) is saved next to the log for
each measurement, usable directly as a qualitative figure.

Headless / reproducible use: pass --x and --y (pixel coordinates in the
*left.png* image, i.e. full capture resolution) to skip the interactive
window entirely, e.g. to resample the same point at a different --iters.

Usage:
    python scripts/eval_raft_laser.py \
        --ckpt checkpoints/raft_middlebury_ft/best.pth \
        --calib outputs/calibration_110mm/stereo_calib.npz \
        --left  outputs/capture_110mm_20260828_.../left.png \
        --right outputs/capture_110mm_20260828_.../right.png \
        --label "mug handle" --laser-m 1.482 \
        --log outputs/raft_laser_accuracy/session_log.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
_RAFT = _HERE.parent / "third_party" / "RAFT-Stereo"
sys.path.insert(0, str(_RAFT))
sys.path.insert(0, str(_RAFT / "core"))

LOG_FIELDS = ["timestamp", "label", "ckpt", "iters", "scale", "x_px", "y_px",
              "laser_m", "raft_m", "err_mm", "err_pct"]


def load_model(ckpt, device):
    from raft_stereo import RAFTStereo
    model = RAFTStereo(argparse.Namespace(
        hidden_dims=[128, 128, 128], corr_implementation="reg",
        corr_levels=4, corr_radius=4, context_norm="batch",
        mixed_precision=False, shared_backbone=True, n_downsample=3,
        n_gru_layers=2, slow_fast_gru=True))
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval()


def raft_disparity(model, L, R, iters, device):
    """Dense disparity in px at the resolution of L/R (positive = nearer)."""
    from utils.utils import InputPadder

    def to_t(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(device)
    t1, t2 = to_t(L), to_t(R)
    padder = InputPadder(t1.shape, divis_by=32)
    t1, t2 = padder.pad(t1, t2)
    with torch.no_grad():
        _, flow = model(t1, t2, iters=iters, test_mode=True)
    return -padder.unpad(flow).squeeze().cpu().numpy()


def sample_depth(depth_m, x, y, half=3):
    """Robust median depth in a small window; None if nothing valid there."""
    y0, y1 = max(0, y - half), min(depth_m.shape[0], y + half + 1)
    x0, x1 = max(0, x - half), min(depth_m.shape[1], x + half + 1)
    win = depth_m[y0:y1, x0:x1]
    valid = win[win > 0]
    return float(np.median(valid)) if valid.size else None


def turbo(d):
    v = d[d > 0]
    if v.size < 50:
        v = d.flatten()
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    n = np.clip((d - lo) / max(hi - lo, 1e-3), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def pick_point_interactively(left_bgr, depth_m):
    """Live depth-under-cursor readout as the mouse moves; left-click to
    arm a candidate point (yellow), ENTER confirms it (then shown green)."""
    state = {"hover": None, "picked": None}
    disp_vis = cv2.addWeighted(left_bgr, 0.55, turbo(depth_m), 0.45, 0)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_MOUSEMOVE:
            state["hover"] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            state["picked"] = (x, y)

    win = "move to the laser-aimed point, click, ENTER to confirm (ESC cancels)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        frame = disp_vis.copy()
        cv2.putText(frame, "move mouse to see live depth; left-click to arm a point",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if state["hover"] is not None and state["picked"] is None:
            x, y = state["hover"]
            z = sample_depth(depth_m, x, y)
            cv2.drawMarker(frame, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
            label = f"{z:.3f} m" if z is not None else "invalid here"
            cv2.putText(frame, label, (x + 15, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if state["picked"] is not None:
            x, y = state["picked"]
            z = sample_depth(depth_m, x, y)
            cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
            label = f"RAFT: {z:.3f} m" if z is not None else "RAFT: invalid here"
            cv2.putText(frame, label, (x + 15, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "ENTER to confirm, or click again to re-pick",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(win, frame)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # ESC
            cv2.destroyWindow(win)
            return None
        if k in (13, 10) and state["picked"] is not None:  # ENTER
            cv2.destroyWindow(win)
            return state["picked"]


def append_log(log_path: Path, row: dict):
    new = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def print_summary(log_path: Path):
    if not log_path.exists():
        return
    rows = list(csv.DictReader(open(log_path)))
    if not rows:
        return
    print("\n| label | laser (m) | RAFT (m) | error (mm) | error (%) |")
    print("|---|---:|---:|---:|---:|")
    errs = []
    for r in rows:
        print("| %s | %.3f | %.3f | %+.0f | %+.2f%% |" %
              (r["label"], float(r["laser_m"]), float(r["raft_m"]),
               float(r["err_mm"]), float(r["err_pct"])))
        errs.append(float(r["err_mm"]))
    rms = float(np.sqrt(np.mean(np.square(errs))))
    print(f"\n{len(rows)} measurement(s) logged. RMS error: {rms:.1f} mm")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--calib", required=True, help="stereo_calib.npz (for baseline/focal)")
    p.add_argument("--left", required=True, help="rectified left.png from capture_stereo.py")
    p.add_argument("--right", required=True, help="rectified right.png")
    p.add_argument("--iters", type=int, default=7, help="7 = live/deployed default")
    p.add_argument("--scale", type=float, default=0.5, help="inference downsample, matches deployment")
    p.add_argument("--label", default=None, help="what the object/point is, e.g. 'mug handle'")
    p.add_argument("--laser-m", type=float, default=None,
                   help="laser reading in metres; prompted for if omitted")
    p.add_argument("--x", type=int, default=None, help="skip GUI: pixel x in left.png (full res)")
    p.add_argument("--y", type=int, default=None, help="skip GUI: pixel y in left.png (full res)")
    p.add_argument("--log", type=Path, default=Path("outputs/raft_laser_accuracy/session_log.csv"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cal = np.load(args.calib)
    baseline_m = float(cal["baseline_mm"][0]) / 1000.0
    focal_px = float(cal["focal_px"][0])
    focal_eff = focal_px * args.scale

    L_full = cv2.imread(args.left)
    R_full = cv2.imread(args.right)
    if L_full is None or R_full is None:
        raise SystemExit("Could not read --left/--right images.")

    print(f"device: {device}   ckpt: {args.ckpt}   iters: {args.iters}   "
          f"baseline: {baseline_m*1000:.2f}mm   focal: {focal_px:.1f}px")
    model = load_model(args.ckpt, device)

    L = cv2.resize(L_full, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    R = cv2.resize(R_full, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    disp = raft_disparity(model, L, R, args.iters, device)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_m = np.where(disp > 0, baseline_m * focal_eff / disp, 0.0)

    if args.x is not None and args.y is not None:
        x_full, y_full = args.x, args.y
        x, y = int(round(x_full * args.scale)), int(round(y_full * args.scale))
    else:
        if not L.flags.writeable:
            L = L.copy()
        picked = pick_point_interactively(L, depth_m)
        if picked is None:
            raise SystemExit("Cancelled -- no point selected.")
        x, y = picked
        x_full, y_full = int(round(x / args.scale)), int(round(y / args.scale))

    raft_z = sample_depth(depth_m, x, y)
    if raft_z is None:
        raise SystemExit(f"No valid disparity in a window around ({x_full},{y_full}). "
                          f"Pick a different point (textured surface, not a specular "
                          f"highlight or a hole in the disparity map).")

    laser_m = args.laser_m
    if laser_m is None:
        laser_m = float(input(f"RAFT reads {raft_z:.3f} m at ({x_full},{y_full}). "
                               f"Laser reading (metres): ").strip())
    label = args.label or input("Label for this point (e.g. 'mug handle'): ").strip() or "unlabeled"

    err_mm = (raft_z - laser_m) * 1000.0
    err_pct = 100.0 * err_mm / (laser_m * 1000.0)
    print(f"\n{label}: laser={laser_m:.3f}m  RAFT={raft_z:.3f}m  "
          f"error={err_mm:+.0f}mm ({err_pct:+.2f}%)")

    row = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"), "label": label, "ckpt": args.ckpt,
        "iters": args.iters, "scale": args.scale, "x_px": x_full, "y_px": y_full,
        "laser_m": f"{laser_m:.4f}", "raft_m": f"{raft_z:.4f}",
        "err_mm": f"{err_mm:.1f}", "err_pct": f"{err_pct:.3f}",
    }
    append_log(args.log, row)

    snap = L_full.copy()
    cv2.drawMarker(snap, (x_full, y_full), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    cv2.putText(snap, f"{label}", (x_full + 18, y_full - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(snap, f"laser: {laser_m:.3f}m  RAFT: {raft_z:.3f}m  err: {err_mm:+.0f}mm",
                (x_full + 18, y_full - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    snap_path = args.log.parent / f"{row['timestamp']}_{label.replace(' ', '_')}.png"
    cv2.imwrite(str(snap_path), snap)
    print(f"Saved annotated snapshot -> {snap_path}")

    print_summary(args.log)


if __name__ == "__main__":
    main()

"""
RAFT-Stereo depth accuracy on the baseline-comparison station captures.

What this adds to eval_depth_accuracy.py
----------------------------------------
`eval_depth_accuracy.py` measures the *geometric floor*: checkerboard corners
differenced directly, no matcher involved. That is the best any model could do
on this hardware. This script measures what the deployed matcher actually
delivers on the same frames, so the two can be read against each other:

    geometric error   what the optics and calibration allow
    RAFT error        what the pipeline actually produces
    the gap           how much the matcher costs

Both are evaluated at the *same 54 corner locations* per board, so the
comparison is like for like -- no region-of-interest choice, no resampling
difference. RAFT's dense disparity is sampled bilinearly at those exact points.

Why this matters for the baseline question
------------------------------------------
The geometric result showed depth precision scaling as Z^2/(f*B), so a 280 mm
rig is ~2.6x more precise than a 110 mm one. But the deployed matcher's own
disparity error is roughly 10x the calibration noise (1.073 px vs ~0.12 px,
docs/jetson_deployment_results.md section 3). The open question is whether the
geometric advantage survives that. It should: a wider baseline divides *any*
disparity error, the matcher's included, by a larger f*B. This measures whether
it does in practice.

Inference matches deployment: realtime RAFT config at 640x480 input. Disparity
is a horizontal quantity, so it scales with width only -- a disparity of d at
640 wide is 3d at the 1920-wide capture resolution.

Usage:
    python scripts/eval_raft_stations.py --iters 4
    python scripts/eval_raft_stations.py --iters 7 --ckpt <other.pth>
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
_RAFT = _HERE.parent / "third_party" / "RAFT-Stereo"
sys.path.insert(0, str(_RAFT))
sys.path.insert(0, str(_RAFT / "core"))

PATTERN = (9, 6)
CHESS_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
REFINE_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# (label, station root, calibration). These are the same calibrations the
# geometric table uses, so the two are directly comparable: at-capture
# extrinsics where the rig drifted mid-session, session calibration otherwise.
RIGS = [
    ("110mm", "outputs/accuracy_110mm_v2", "outputs/calibration_110mm/stereo_calib.npz"),
    ("160mm", "outputs/accuracy_160mm", "outputs/calibration_160mm/stereo_calib_atcapture.npz"),
    ("280mm", "outputs/accuracy_280mm", "outputs/calibration_280mm/stereo_calib_atcapture.npz"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/raft_middlebury_ft/best.pth",
                   help="the deployed checkpoint (see scripts/export_raft_onnx.py)")
    p.add_argument("--iters", type=int, default=4,
                   help="GRU iterations. 4 is the recommended on-device config "
                        "(i4 @ 480x640); 7 is the accuracy-first one.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--rigs", default="110mm,160mm,280mm")
    p.add_argument("--root", default=None,
                   help="evaluate this station dir instead of the built-in RIGS")
    p.add_argument("--calib", default=None, help="calibration for --root")
    p.add_argument("--label", default="custom", help="label for --root")
    p.add_argument("--distances", default=None,
                   help="text file of '<station> <metres>' giving MEASURED "
                        "distances (e.g. laser). Without it the nominal "
                        "distance is parsed from the folder name.")
    p.add_argument("--per-capture", action="store_true",
                   help="print each capture's own medians before pooling. A "
                        "station pools all its captures into one median, so a "
                        "single diverged capture can move the station result "
                        "without being visible in the summary row.")
    return p.parse_args()


def load_model(ckpt, device):
    from raft_stereo import RAFTStereo
    import argparse as _a
    model = RAFTStereo(_a.Namespace(
        hidden_dims=[128, 128, 128], corr_implementation="reg",
        corr_levels=4, corr_radius=4, context_norm="batch",
        mixed_precision=False, shared_backbone=True, n_downsample=3,
        n_gru_layers=2, slow_fast_gru=True))
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval()


def raft_disparity(model, padder_cls, L, R, iters, device):
    """Dense disparity in px at the resolution of L/R (positive = nearer)."""
    def to_t(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(device)
    t1, t2 = to_t(L), to_t(R)
    padder = padder_cls(t1.shape, divis_by=32)
    t1, t2 = padder.pad(t1, t2)
    with torch.no_grad():
        _, flow = model(t1, t2, iters=iters, test_mode=True)
    # RAFT returns flow; disparity is its negation (see scripts/infer_raft.py).
    return -padder.unpad(flow).squeeze().cpu().numpy()


def sample_at(disp, pts):
    """Bilinear-sample a disparity map at floating-point pixel coordinates."""
    mx = pts[:, 0].astype(np.float32).reshape(-1, 1)
    my = pts[:, 1].astype(np.float32).reshape(-1, 1)
    return cv2.remap(disp.astype(np.float32), mx, my,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: %s   ckpt: %s   iters: %d   input: %dx%d\n"
          % (device, args.ckpt, args.iters, args.width, args.height))
    model = load_model(args.ckpt, device)
    from utils.utils import InputPadder

    rig_list = RIGS
    if args.root:
        if not args.calib:
            raise SystemExit("--root requires --calib")
        rig_list = [(args.label, args.root, args.calib)]
        args.rigs = args.label
    # measured distances override the nominal folder names when supplied
    overrides = {}
    if args.distances:
        for line in Path(args.distances).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                overrides[parts[0]] = float(parts[1])
        print("using measured distances from %s (%d stations)\n"
              % (args.distances, len(overrides)))

    wanted = [r.strip() for r in args.rigs.split(",")]
    summary = []

    for label, root, calpath in rig_list:
        if label not in wanted:
            continue
        cal = np.load(calpath)
        f = float(cal["focal_px"][0])
        B = float(cal["baseline_mm"][0])
        maps = (cal["map1L"], cal["map2L"], cal["map1R"], cal["map2R"])
        # Corners are detected on the RAW pixels and their COORDINATES
        # rectified, rather than detecting on the remapped image. Remapping
        # resamples, and at 5-6 m the board is only a few px per square, so
        # that blur costs detection outright -- it lost both far stations at
        # 110 mm. Rectifying 54 points is lossless. The remapped IMAGES are
        # still what RAFT is fed; only detection moves to the raw pixels.
        rect = (cal["K1"], cal["D1"], cal["R1"], cal["P1"],
                cal["K2"], cal["D2"], cal["R2"], cal["P2"])
        full_w = int(cal["image_size"][0])
        sx = args.width / float(full_w)      # disparity scales with width only

        print("=" * 74)
        print("%s   f=%.2f px  B=%.2f mm  (%s)" % (label, f, B, calpath))
        print("=" * 74)
        print("| station | tape | geometric Z | RAFT Z | geom err | RAFT err | "
              "RAFT disp err |")
        print("|---|---:|---:|---:|---:|---:|---:|")

        rows = []
        for st in sorted(Path(root).glob("z*m")):
            z_true = overrides.get(st.name, float(st.name[1:-1])) * 1000.0
            g_disp, r_disp = [], []
            for cap in sorted(st.glob("capture_*")):
                lr, rr = cap / "left_raw.png", cap / "right_raw.png"
                if not (lr.exists() and rr.exists()):
                    continue
                li, ri = cv2.imread(str(lr)), cv2.imread(str(rr))
                if li is None or ri is None:
                    continue
                # Rectify with the same calibration the geometric table used.
                rl = cv2.remap(li, maps[0], maps[1], cv2.INTER_LINEAR)
                rr_img = cv2.remap(ri, maps[2], maps[3], cv2.INTER_LINEAR)

                # Neither detection path dominates: RAW keeps full sharpness
                # but leaves lens distortion in, REMAPPED removes distortion but
                # resamples (which costs detection at 5-6 m, where a square is
                # only a few px). So try raw first, fall back to remapped --
                # the same strategy eval_depth_accuracy.py uses.
                K1c, D1c, R1c, P1c, K2c, D2c, R2c, P2c = rect
                pair = None
                for mode in ("raw", "remap"):
                    if mode == "raw":
                        gl = cv2.cvtColor(li, cv2.COLOR_BGR2GRAY)
                        gr = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)
                    else:
                        gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
                        gr = cv2.cvtColor(rr_img, cv2.COLOR_BGR2GRAY)
                    okl, cl = cv2.findChessboardCorners(gl, PATTERN, flags=CHESS_FLAGS)
                    okr, cr = cv2.findChessboardCorners(gr, PATTERN, flags=CHESS_FLAGS)
                    if not (okl and okr):
                        continue
                    sp = np.median(np.linalg.norm(
                        np.diff(cl.reshape(PATTERN[1], PATTERN[0], 2), axis=1), axis=2))
                    w = int(max(2, min(11, sp / 3.0)))
                    cl = cv2.cornerSubPix(gl, cl, (w, w), (-1, -1), REFINE_CRIT).reshape(-1, 2)
                    cr = cv2.cornerSubPix(gr, cr, (w, w), (-1, -1), REFINE_CRIT).reshape(-1, 2)
                    if mode == "raw":
                        # rectify the COORDINATES; lossless, unlike resampling
                        cl = cv2.undistortPoints(cl.reshape(-1, 1, 2), K1c, D1c,
                                                 R=R1c, P=P1c).reshape(-1, 2)
                        cr = cv2.undistortPoints(cr.reshape(-1, 1, 2), K2c, D2c,
                                                 R=R2c, P=P2c).reshape(-1, 2)
                    # Rectified correspondences share a row. Try both orderings
                    # and KEEP the better one, rather than reversing whenever dy
                    # exceeds a threshold -- an unverified flip can make the
                    # pairing worse instead of better.
                    best = None
                    for candidate in (cr, cr[::-1]):
                        d = float(np.median(np.abs(cl[:, 1] - candidate[:, 1])))
                        if best is None or d < best[0]:
                            best = (d, candidate)
                    row_dy, cr = best
                    if row_dy > 3.0:
                        continue
                    gd = cl[:, 0] - cr[:, 0]
                    if np.median(gd) <= 0:
                        continue
                    pair = (cl, cr, gd)
                    break
                if pair is None:
                    continue
                cl, cr, gd = pair

                # RAFT at deployment resolution, sampled at the same corners.
                Ls = cv2.resize(rl, (args.width, args.height), interpolation=cv2.INTER_AREA)
                Rs = cv2.resize(rr_img, (args.width, args.height), interpolation=cv2.INTER_AREA)
                d = raft_disparity(model, InputPadder, Ls, Rs, args.iters, device)
                pts = np.c_[cl[:, 0] * sx, cl[:, 1] * (args.height / float(cal["image_size"][1]))]
                rd = sample_at(d, pts) / sx     # back to full-resolution px

                g_disp.append(gd)
                r_disp.append(rd)
                if args.per_capture:
                    mg, mr = float(np.median(gd)), float(np.median(rd))
                    print("|    %s/%s | geo %.2f px | raft %.2f px | "
                          "err %.2f px | z_raft %.3f m |"
                          % (st.name, cap.name, mg, mr, abs(mr - mg),
                             (f * B / mr) / 1000.0 if mr > 0 else float("nan")))

            if not g_disp:
                print("| %s | -- | *no usable captures* | | | | |" % st.name)
                continue
            gd = np.concatenate(g_disp)
            rd = np.concatenate(r_disp)
            zg = f * B / np.median(gd)
            zr = f * B / np.median(rd)
            rows.append((z_true, zg, zr, float(np.median(np.abs(rd - gd)))))
            print("| %s | %.2f m | %.3f m | %.3f m | %+.0f mm | %+.0f mm | %.2f px |"
                  % (st.name, z_true / 1000, zg / 1000, zr / 1000,
                     zg - z_true, zr - z_true, rows[-1][3]))

        if rows:
            a = np.array(rows)
            rms_g = float(np.sqrt(np.mean((a[:, 1] - a[:, 0]) ** 2)))
            rms_r = float(np.sqrt(np.mean((a[:, 2] - a[:, 0]) ** 2)))
            mean_dd = float(np.mean(a[:, 3]))
            print("\nRMS vs tape: geometric %.1f mm, RAFT %.1f mm  (%.1fx worse)"
                  % (rms_g, rms_r, rms_r / rms_g if rms_g else float("nan")))
            print("mean RAFT disparity error vs the corner reference: %.2f px"
                  % mean_dd)
            summary.append((label, f * B, rms_g, rms_r, mean_dd))
        print()

    if len(summary) > 1:
        print("=" * 74)
        print("SUMMARY")
        print("=" * 74)
        print("| baseline | f*B | geometric RMS | RAFT RMS | RAFT disp err | "
              "RAFT RMS vs widest |")
        print("|---|---:|---:|---:|---:|---:|")
        widest = max(summary, key=lambda r: r[1])
        for label, fB, rg, rr, dd in summary:
            print("| %s | %.0f | %.1f mm | %.1f mm | %.2f px | %.2fx |"
                  % (label, fB, rg, rr, dd, rr / widest[3] if widest[3] else float("nan")))
        print("\nThe disparity error column is the matcher's own error and should "
              "be roughly BASELINE-INDEPENDENT -- it is a property of the network "
              "and the imagery, not the geometry. If it is, then the depth RMS "
              "differences across baselines come from f*B alone, exactly as in "
              "the geometric result.")


if __name__ == "__main__":
    main()

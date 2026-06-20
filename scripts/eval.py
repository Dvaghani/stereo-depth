"""
Evaluate a trained StereoUNet on KITTI 2015 (training split with GT) or
Middlebury 2014. Prints per-scene metrics and an aggregate summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.models import StereoUNet
from src.models.stereo_unet import StereoUNetConfig
from src.models.aanet import AANetWrapper
from src.datasets import KITTI2015Stereo, Middlebury2014Stereo, StereoTransform
from src.utils.metrics import compute_kitti_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--dataset", choices=["kitti", "middlebury"], default="kitti")
    p.add_argument("--data-root", required=True, type=str)
    p.add_argument("--max-disp", type=int, default=192)
    p.add_argument("--confidence-threshold", type=float, default=None)
    p.add_argument("--sweep-thresholds", type=float, nargs="+", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tf = StereoTransform(crop_size=None, color_jitter=0.0, training=False)
    if args.dataset == "kitti":
        ds = KITTI2015Stereo(args.data_root, split="training", transform=tf, return_path=True)
    else:
        ds = Middlebury2014Stereo(args.data_root, transform=tf, downsample=2, return_path=True)

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # Read config from checkpoint to auto-detect model type and uncertainty head.
    state = torch.load(args.ckpt, map_location=device)
    ckpt_cfg = state.get("config", {}) if isinstance(state, dict) else {}
    model_type = ckpt_cfg.get("model", "unet")
    predict_uncertainty = ckpt_cfg.get("predict_uncertainty", False)
    max_disp = ckpt_cfg.get("max_disp", args.max_disp)

    if model_type == "aanet":
        model = AANetWrapper(
            max_disp=max_disp,
            predict_uncertainty=predict_uncertainty,
        ).to(device).eval()
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        # Handle flat upstream weights (no "backbone." prefix)
        if not any(k.startswith("backbone.") for k in sd):
            sd = {"backbone." + k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print(f"Loaded AANetWrapper (uncertainty={predict_uncertainty}) from {args.ckpt}")
    else:
        cfg = StereoUNetConfig(
            max_disp=max_disp,
            feat_channels=ckpt_cfg.get("feat_channels", 64),
            unet_base_channels=ckpt_cfg.get("unet_base_channels", 32),
            use_cuda_extension=True,
            predict_uncertainty=predict_uncertainty,
        )
        model = StereoUNet(cfg).to(device).eval()
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"Loaded StereoUNet (uncertainty={predict_uncertainty}) from {args.ckpt}")

    has_unc = predict_uncertainty
    use_thresh = args.confidence_threshold is not None and has_unc
    sweep = args.sweep_thresholds if (args.sweep_thresholds and has_unc) else None
    if args.confidence_threshold is not None and not has_unc:
        print("WARNING: --confidence-threshold ignored: checkpoint has no uncertainty head.")
    if args.sweep_thresholds is not None and not has_unc:
        print("WARNING: --sweep-thresholds ignored: checkpoint has no uncertainty head.")

    agg = {"EPE": 0.0, "D1-all": 0.0, "bad-1": 0.0, "bad-2": 0.0, "bad-3": 0.0}
    agg_conf = {k: 0.0 for k in agg} if use_thresh else None
    coverage_sum = 0.0  # fraction of GT-valid pixels that are also confident
    n = 0
    n_conf = 0  # frames where the confident-masked metrics were non-empty

    # For the sweep, accumulate per-threshold aggregates in parallel.
    if sweep:
        sweep_agg = {t: {k: 0.0 for k in agg} for t in sweep}
        sweep_cov = {t: 0.0 for t in sweep}
        sweep_n = {t: 0 for t in sweep}

    with torch.no_grad():
        for batch in loader:
            L = batch["left"].to(device); R = batch["right"].to(device)
            D = batch["disparity"].to(device); V = batch["valid"].to(device)
            V = V * (D < float(cfg.max_disp)).float()
            out = model(L, R)
            pred = out["disparity"]
            m = compute_kitti_metrics(pred, D, V)
            line = f"{batch['path'][0]}  EPE={m['EPE']:.3f}  D1-all={m['D1-all']:.2f}%"

            if use_thresh or sweep:
                b = torch.exp(out["log_b"])  # (B, H, W) in pixels
                gt_valid_n = float(V.sum().item())

            if use_thresh:
                conf_mask = (b < args.confidence_threshold).float()
                V_conf = V * conf_mask
                conf_within_gt = float(V_conf.sum().item())
                cov = conf_within_gt / max(gt_valid_n, 1.0)
                coverage_sum += cov
                if conf_within_gt > 0:
                    m_conf = compute_kitti_metrics(pred, D, V_conf)
                    for k in agg_conf: agg_conf[k] += m_conf[k]
                    n_conf += 1
                    line += f"  | conf: EPE={m_conf['EPE']:.3f} D1={m_conf['D1-all']:.2f}% cov={100*cov:.1f}%"
                else:
                    line += f"  | conf: (no confident GT-valid pixels)"

            if sweep:
                for t in sweep:
                    V_t = V * (b < t).float()
                    n_t = float(V_t.sum().item())
                    cov_t = n_t / max(gt_valid_n, 1.0)
                    sweep_cov[t] += cov_t
                    if n_t > 0:
                        m_t = compute_kitti_metrics(pred, D, V_t)
                        for k in sweep_agg[t]: sweep_agg[t][k] += m_t[k]
                        sweep_n[t] += 1

            print(line)

            for k in agg: agg[k] += m[k]
            n += 1
    print("\n== Aggregate (all GT-valid pixels) ==")
    for k in agg:
        print(f"  {k}: {agg[k]/max(n,1):.3f}")
    if use_thresh:
        print(f"\n== Aggregate (confident GT-valid pixels, b<{args.confidence_threshold}) ==")
        for k in agg_conf:
            print(f"  {k}: {agg_conf[k]/max(n_conf,1):.3f}")
        print(f"  mean coverage of GT-valid pixels: {100*coverage_sum/max(n,1):.1f}%")

    if sweep:
        print("\n== Coverage sweep ==")
        print(f"  {'threshold':>11}  {'coverage':>9}  {'EPE':>7}  {'D1-all':>7}  {'bad-2':>7}  {'bad-3':>7}")
        for t in sorted(sweep):
            cov = 100 * sweep_cov[t] / max(n, 1)
            nn = max(sweep_n[t], 1)
            epe = sweep_agg[t]['EPE'] / nn
            d1 = sweep_agg[t]['D1-all'] / nn
            b2 = sweep_agg[t]['bad-2'] / nn
            b3 = sweep_agg[t]['bad-3'] / nn
            print(f"  b<{t:>7.2f} px  {cov:>7.1f}%  {epe:>7.3f}  {d1:>6.2f}%  {b2:>6.2f}%  {b3:>6.2f}%")
        # Also report the all-pixels row for comparison
        epe_all = agg['EPE']/max(n,1)
        d1_all = agg['D1-all']/max(n,1)
        b2_all = agg['bad-2']/max(n,1)
        b3_all = agg['bad-3']/max(n,1)
        print(f"  {'(no mask)':>11}  {100.0:>7.1f}%  {epe_all:>7.3f}  {d1_all:>6.2f}%  {b2_all:>6.2f}%  {b3_all:>6.2f}%")


if __name__ == "__main__":
    main()

"""
Run inference on a single stereo pair and save disparity + depth maps.

Example:
    python scripts/infer.py --ckpt checkpoints/best.pt \\
        --left left.png --right right.png \\
        --baseline 0.14 --focal 700 --out outputs/run1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.models import StereoUNet
from src.models.stereo_unet import StereoUNetConfig
from src.models.aanet import AANetWrapper
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.utils.io import save_disparity_visualization, disparity_to_depth, write_pfm
from src.utils.online_rectify import estimate_vertical_offset, apply_vertical_shift


def _load(path: str, normalize: bool = True, vshift: float = 0.0) -> torch.Tensor:
    """Load an image as a (1, 3, H, W) float tensor in [0,1].
    normalize=True applies ImageNet mean/std (for StereoUNet).
    normalize=False returns raw [0,1] (for AANetWrapper which normalises internally).
    vshift translates the image vertically (px) before tensorizing — used by the
    online rectification refinement to cancel a residual vertical offset.
    """
    img = np.asarray(Image.open(path).convert("RGB"))
    if abs(vshift) > 1e-3:
        img = apply_vertical_shift(img, vshift)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        t = (t - mean) / std
    return t


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--out", default="outputs/infer", help="Output directory.")
    p.add_argument("--max-disp", type=int, default=192)
    p.add_argument("--baseline", type=float, default=0.14,
                   help="Stereo baseline in meters (Holybro X500 medium: 0.14).")
    p.add_argument("--focal", type=float, default=700.0,
                   help="Focal length in pixels at FULL resolution (intrinsics-dependent). "
                        "Automatically rescaled by --scale so depth stays metric.")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Downsample factor applied to both images before inference "
                        "(e.g. 0.5 = half resolution). Lowering this brings close-range "
                        "disparities back under --max-disp AND speeds up inference — the "
                        "right knob for high-res / close-range (Middlebury, Brio) inputs.")
    p.add_argument("--online-align", action="store_true",
                   help="Estimate and cancel a residual VERTICAL rectification "
                        "offset from this pair before matching (corrects mount "
                        "flex / vibration on top of the static calibration).")
    p.add_argument("--align-scale", type=float, default=0.5,
                   help="Downscale for the online-align SIFT estimate (0.5 ≈ "
                        "±0.3px @ ~180ms; 1.0 = exact but slow).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load checkpoint and build model ──────────────────────────────────────
    state   = torch.load(args.ckpt, map_location=device)
    ckpt_cfg    = state.get("config", {}) if isinstance(state, dict) else {}
    model_type  = ckpt_cfg.get("model", "unet")
    predict_unc = ckpt_cfg.get("predict_uncertainty", False)
    max_disp    = ckpt_cfg.get("max_disp", args.max_disp)

    if model_type == "aanet":
        model = AANetWrapper(max_disp=max_disp, predict_uncertainty=predict_unc).to(device).eval()
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        if not any(k.startswith("backbone.") for k in sd):
            sd = {"backbone." + k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        # IMPORTANT: training (build_dataset → StereoTransform(normalize=True))
        # feeds AANet ImageNet-normalized images, and AANetWrapper.forward()
        # normalizes AGAIN internally. To match the training input distribution
        # we must ALSO ImageNet-normalize here (verified: EPE 0.50 vs 0.91px on
        # KITTI when fed raw [0,1]). Mismatched normalization was a contributor
        # to poor out-of-distribution (Middlebury / Brio) results.
        normalize_input = True
        # Pad to multiple of 64 for AANet's multi-scale pyramid
        pad_multiple = 64
        print(f"Model: AANetWrapper (uncertainty={predict_unc})")
    else:
        cfg = StereoUNetConfig(
            max_disp=max_disp,
            feat_channels=ckpt_cfg.get("feat_channels", 64),
            unet_base_channels=ckpt_cfg.get("unet_base_channels", 32),
            use_cuda_extension=True,
            predict_uncertainty=predict_unc,
        )
        model = StereoUNet(cfg).to(device).eval()
        model.load_state_dict(state["model"] if "model" in state else state)
        normalize_input = True
        pad_multiple = 16   # deepest U-Net stride
        print(f"Model: StereoUNet (uncertainty={predict_unc})")

    # ── Online rectification refinement (optional) ────────────────────────────
    # Estimate the residual vertical offset between the (already rectified) pair
    # and cancel it by shifting the RIGHT image. Corrects mount flex / vibration
    # that the static calibration cannot capture.
    vshift = 0.0
    if args.online_align:
        import cv2
        Lbgr = cv2.imread(args.left); Rbgr = cv2.imread(args.right)
        if Lbgr is None or Rbgr is None:
            raise SystemExit("online-align: could not read input images.")
        dy, n = estimate_vertical_offset(Lbgr, Rbgr, scale=args.align_scale)
        if dy is None:
            print(f"online-align: only {n} matches — skipping (textureless scene?)")
        else:
            vshift = dy
            print(f"online-align: residual vertical offset {dy:+.2f}px "
                  f"({n} matches) → shifting right image to cancel it")

    # ── Load and pad images ───────────────────────────────────────────────────
    L = _load(args.left,  normalize=normalize_input).unsqueeze(0).to(device)
    R = _load(args.right, normalize=normalize_input, vshift=vshift).unsqueeze(0).to(device)
    if L.shape != R.shape:
        raise SystemExit(f"Left/right shape mismatch: {L.shape} vs {R.shape}")

    # Downsample before inference. This both (a) brings close-range disparities
    # back under max_disp and (b) speeds up inference. Disparity is then measured
    # in downsampled pixels, so the effective focal scales by the same factor —
    # we account for that below when converting disparity → metric depth.
    if args.scale != 1.0:
        L = F.interpolate(L, scale_factor=args.scale, mode="bilinear", align_corners=False)
        R = F.interpolate(R, scale_factor=args.scale, mode="bilinear", align_corners=False)
        print(f"Downsampled to {tuple(L.shape[-2:])} (scale={args.scale})")
    focal_eff = args.focal * args.scale

    H, W = L.shape[-2:]
    pad_h = (pad_multiple - H % pad_multiple) % pad_multiple
    pad_w = (pad_multiple - W % pad_multiple) % pad_multiple
    L = F.pad(L, (0, pad_w, 0, pad_h), mode="reflect")
    R = F.pad(R, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        out  = model(L, R)
        disp = out["disparity"][:, :H, :W]

        # Entropy from cost volume (UNet only — AANet doesn't expose cost)
        cost = out.get("cost")
        if cost is not None:
            prob = F.softmax(cost.float(), dim=1)
            entropy_low  = -(prob * torch.log2(prob.clamp_min(1e-9))).sum(dim=1, keepdim=True)
            entropy_full = F.interpolate(entropy_low,
                                         size=(H + pad_h, W + pad_w),
                                         mode="bilinear", align_corners=False)
            entropy_np  = entropy_full[0, 0, :H, :W].cpu().numpy().astype(np.float32)
            max_entropy = math.log2(cost.shape[1])
        else:
            entropy_np  = None
            max_entropy = None

        # Laplace uncertainty (both models when head is present)
        log_b_full = out.get("log_b")
        if log_b_full is not None:
            # Clamp log_b to the same range used in laplace_nll_loss [-5, 5]
            # before exponentiation to avoid overflow on out-of-distribution pixels
            log_b_clamped = log_b_full[0, :H, :W].clamp(-5, 5)
            b_np = log_b_clamped.cpu().numpy().astype(np.float32)
            b_np = np.exp(b_np)   # now safely in [e^-5, e^5] = [0.007, 148] px
        else:
            b_np = None

    disp_np  = disp[0].cpu().numpy().astype(np.float32)
    depth_np = disparity_to_depth(disp_np, args.baseline, focal_eff).astype(np.float32)

    save_disparity_visualization(disp_np, out_dir / "disparity.png", max_disp=float(max_disp))
    write_pfm(out_dir / "disparity.pfm", disp_np)
    depth_mm = np.clip(depth_np * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(out_dir / "depth_mm.png")

    saved = ["disparity.png", "disparity.pfm", "depth_mm.png"]

    if entropy_np is not None:
        save_disparity_visualization(entropy_np, out_dir / "entropy.png", max_disp=max_entropy)
        saved.append("entropy.png")

    if b_np is not None:
        save_disparity_visualization(b_np, out_dir / "uncertainty_b.png", max_disp=10.0)
        mask = (b_np < 2.0).astype(np.float32)
        disp_masked = disp_np * mask
        save_disparity_visualization(disp_masked, out_dir / "disparity_confident.png",
                                     max_disp=float(max_disp))
        saved += ["uncertainty_b.png", "disparity_confident.png"]

    print(f"Saved {', '.join(saved)} → {out_dir}")
    print(f"  disparity: min={disp_np.min():.2f}  max={disp_np.max():.2f} px")
    print(f"  depth:     min={depth_np.min():.2f}  max={depth_np.max():.2f} m")
    if entropy_np is not None:
        print(f"  entropy:   min={entropy_np.min():.2f}  max={entropy_np.max():.2f} bits")
    if b_np is not None:
        frac_conf = float((b_np < 2.0).mean())
        print(f"  uncertainty b: min={b_np.min():.3f}  max={b_np.max():.3f} px  "
              f"(confident <2px: {100*frac_conf:.1f}% of pixels)")


if __name__ == "__main__":
    main()

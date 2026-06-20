"""
End-to-end smoke test for StereoUNet.

Run with:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.models import StereoUNet
from src.models.stereo_unet import StereoUNetConfig
from src.models.cost_volume import build_correlation_volume, CorrelationVolume
from src.utils.losses import multi_scale_loss, disparity_smooth_l1
from src.utils.metrics import compute_kitti_metrics


def main() -> int:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device}")

    # ---------------- 1. correlation parity ----------------
    fl = F.normalize(torch.randn(1, 16, 8, 32, device=device), dim=1)
    fr = F.normalize(torch.randn(1, 16, 8, 32, device=device), dim=1)
    ref = build_correlation_volume(fl, fr, max_disp=12)
    wrap = CorrelationVolume(max_disp=12)(fl, fr)
    assert ref.shape == wrap.shape == (1, 12, 8, 32), ref.shape
    assert torch.allclose(ref, wrap, atol=1e-5), \
        f"reference vs wrapper diff = {(ref - wrap).abs().max().item():.2e}"
    print(f"[smoke] correlation parity OK, shape={tuple(ref.shape)}")

    # ---------------- 2. forward / backward ----------------
    # Use a small max_disp so the test runs in a few seconds on CPU.
    cfg = StereoUNetConfig(
        max_disp=48, feat_channels=32, unet_base_channels=16, use_cuda_extension=False
    )
    model = StereoUNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] model params = {n_params:,}")

    B, H, W = 1, 64, 128
    L = torch.randn(B, 3, H, W, device=device, requires_grad=False)
    R = torch.randn(B, 3, H, W, device=device, requires_grad=False)

    # Synthetic ground truth: a horizontal ramp from 0 to 16 px.
    gt = torch.linspace(0, 16, W, device=device).expand(B, H, W).contiguous()
    valid = torch.ones_like(gt)

    out = model(L, R)
    assert out["disparity"].shape == (B, H, W), out["disparity"].shape
    assert out["disparity_low"].shape == (B, H // 4, W // 4)
    assert out["cost"].shape == (B, cfg.max_disp // 4, H // 4, W // 4)
    assert torch.isfinite(out["disparity"]).all()

    loss = multi_scale_loss(out["disparity"], out["disparity_low"], gt, valid)
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    print(f"[smoke] forward OK   loss = {loss.item():.4f}")

    loss.backward()
    n_with_grad = 0
    n_without_grad = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            n_without_grad += 1
            print(f"  [warn] no grad on {name}")
        elif p.grad.abs().sum().item() == 0:
            n_without_grad += 1
            print(f"  [warn] zero grad on {name}")
        else:
            n_with_grad += 1
    print(f"[smoke] backward OK  params_with_grad = {n_with_grad}, "
          f"params_without_grad = {n_without_grad}")
    assert n_without_grad == 0, "some learnable layers received no gradient"

    # ---------------- 3. metrics sanity ----------------
    pred = gt.clone()
    pred[..., :10] += 5.0  # deliberate 5-px error on left 10 cols
    m = compute_kitti_metrics(pred, gt, valid)
    print(f"[smoke] metrics on controlled error: {m}")
    # ~ 10 / 128 ≈ 7.8 % of pixels have a >3px AND >5% error.
    assert 5.0 < m["D1-all"] < 12.0, f"D1-all sanity failed: {m['D1-all']}"
    assert m["EPE"] > 0.0

    # ---------------- 4. predict_depth helper ----------------
    with torch.no_grad():
        depth = model.predict_depth(L, R, baseline_m=0.14, focal_px=700.0)
    assert depth.shape == (B, H, W)
    assert (depth > 0).all()
    print(f"[smoke] depth helper OK   depth range = "
          f"[{depth.min().item():.2f}, {depth.max().item():.2f}] m")

    print("\n[smoke] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

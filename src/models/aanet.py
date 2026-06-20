"""
AANetWrapper
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_AANET_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "aanet"
if str(_AANET_ROOT) not in sys.path:
    sys.path.insert(0, str(_AANET_ROOT))

from nets.aanet import AANet


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty head
# ─────────────────────────────────────────────────────────────────────────────

class _UncertaintyHead(nn.Module):
    """Per-pixel log(b) where b is the Laplace scale in pixel units.

    Same design as the head in unet_aggregator.py — 3×3 conv + BN + ReLU + 1×1 conv.
    Tapped from the last refinement module's dilated_blocks output (32 ch, full res)
    so it sees both the corrected disparity signal and the left-image context.
    """
    def __init__(self, in_channels: int = 32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)   # (B, 1, H, W)

# Wrapper


class AANetWrapper(nn.Module):
    """
    Args:
        max_disp:            Maximum disparity at full resolution (192 for KITTI).
        predict_uncertainty: Attach a Laplace uncertainty head and return log_b.
                             Use with --predict-uncertainty + --init-from in train.py.
    """

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, max_disp: int = 192, predict_uncertainty: bool = False):
        super().__init__()
        self.max_disp = max_disp
        self.predict_uncertainty = predict_uncertainty

        self.backbone = AANet(
            max_disp=max_disp,
            num_downsample=2,               # 2 refinement stages → full-res output
            feature_type="aanet",
            no_feature_mdconv=False,        # use deformable conv in feature extractor
            feature_pyramid_network=True,
            feature_similarity="correlation",
            aggregation_type="adaptive",
            num_scales=3,
            num_fusions=6,
            num_stage_blocks=1,
            num_deform_blocks=3,
            no_intermediate_supervision=False,
            refinement_type="stereodrnet",
            mdconv_dilation=2,
            deformable_groups=2,   # must match pretrained weights (default in upstream)
        )

        # Populated by the forward hook registered below (when uncertainty=True)
        self._refinement_features: torch.Tensor | None = None

        if predict_uncertainty:
            self.uncertainty_head = _UncertaintyHead(in_channels=32)
            self._register_refinement_hook()
        else:
            self.uncertainty_head = None

    # forward hook to capture refinement features

    def _register_refinement_hook(self) -> None:
        """Hook on the last refinement module's dilated_blocks.

        StereoDRNetRefinement.forward:
            out = self.dilated_blocks(concat2)   # (B, 32, H, W)  ← we want this
            residual_disp = self.final_conv(out)

        The hook fires after dilated_blocks and stores its output tensor.
        """
        last_refine = self.backbone.refinement[-1]   # StereoDRNetRefinement

        def _hook(_module: nn.Module, _inp, output: torch.Tensor) -> None:
            self._refinement_features = output       # (B, 32, H, W)

        last_refine.dilated_blocks.register_forward_hook(_hook)

    # ── normalisation ────────────────────────────────────────────────────────

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._MEAN.to(device=x.device, dtype=x.dtype)
        std  = self._STD.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> dict:
        """
        Args:
            left, right : (B, 3, H, W) float tensors in [0, 1].

        Returns dict:
            disparity      (B, H, W)       final disparity in pixels
            disparity_low  (B, H/4, W/4)   stride-4 aux-loss signal
            log_b          (B, H, W)       only if predict_uncertainty=True
            log_b_low      (B, H/4, W/4)  only if predict_uncertainty=True
        """
        H, W = left.shape[-2:]

        left_n  = self._normalize(left)
        right_n = self._normalize(right)
        pyramid = self.backbone(left_n, right_n)

        disp_full = pyramid[-1]
        disp_low = F.interpolate(
            pyramid[-2].unsqueeze(1),
            size=(H // 4, W // 4),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1) / 4.0

        result: dict[str, torch.Tensor] = {
            "disparity":     disp_full,
            "disparity_low": disp_low,
        }

        # uncertainty head
        if self.uncertainty_head is not None and self._refinement_features is not None:
            feats = self._refinement_features          # (B, 32, H, W)
            log_b_full = self.uncertainty_head(feats)  # (B, 1, H, W)

            log_b_low = F.interpolate(
                log_b_full,
                size=(H // 4, W // 4),
                mode="bilinear",
                align_corners=False,
            )

            log_b_full = log_b_full + math.log(4.0)

            result["log_b"]     = log_b_full.squeeze(1)  # (B, H, W)
            result["log_b_low"] = log_b_low.squeeze(1)   # (B, H/4, W/4)

        return result

    # Convenience

    @torch.no_grad()
    def predict_depth(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        baseline_m: float,
        focal_px: float,
        min_disp: float = 1e-3,
    ) -> torch.Tensor:
        """Depth (metres):  depth = (baseline × focal) / disparity"""
        disp = self.forward(left, right)["disparity"].clamp_min(min_disp)
        return (baseline_m * focal_px) / disp


if __name__ == "__main__":
    print("=== AANetWrapper smoke test ===")
    for uncertainty in (False, True):
        model = AANetWrapper(max_disp=192, predict_uncertainty=uncertainty)
        model.eval()
        L = torch.randn(1, 3, 256, 512)
        R = torch.randn(1, 3, 256, 512)
        with torch.no_grad():
            out = model(L, R)
        print(f"\npredict_uncertainty={uncertainty}")
        for k, v in out.items():
            print(f"  {k:15s}  shape={tuple(v.shape)}  range=[{v.min():.2f}, {v.max():.2f}]")
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total:,}")

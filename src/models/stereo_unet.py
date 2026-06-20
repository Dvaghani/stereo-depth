"""
End-to-end StereoUNet model.

Pipeline (Module A from the thesis concept):

    Left, Right images (B, 3, H, W)
        |
        v
    Shared 2D CNN feature extractor  ->  fL, fR : (B, C, H/4, W/4)
        |
        v
    Correlation cost volume  ->  cost : (B, D, H/4, W/4)        [D = max_disp/4]
        |
        v
    2D U-Net aggregator      ->  cost' : (B, D, H/4, W/4)
        |
        v
    Soft-argmin regression   ->  disp_q : (B, 1, H/4, W/4)
        |
        v
    Upsample x4 + scale x4   ->  disp   : (B, 1, H,   W)         [pixels]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import FeatureExtractor
from .cost_volume import CorrelationVolume, build_correlation_volume
from .unet_aggregator import UNetAggregator


@dataclass
class StereoUNetConfig:
    max_disp: int = 192          # max disparity at full resolution (KITTI: 192)
    feat_channels: int = 64
    unet_base_channels: int = 32
    feature_stride: int = 4
    use_cuda_extension: bool = True
    # If True, the U-Net aggregator gains a head that predicts per-pixel
    # log(b) where b is the Laplace scale (uncertainty in pixels). Trained
    # with Laplace NLL; at inference, exp(log_b) can be thresholded to mask
    # out unreliable predictions (sky, occlusions, reflective surfaces).
    predict_uncertainty: bool = False


class StereoUNet(nn.Module):
    def __init__(self, cfg: StereoUNetConfig | None = None):
        super().__init__()
        cfg = cfg or StereoUNetConfig()
        if cfg.max_disp % cfg.feature_stride != 0:
            raise ValueError(
                f"max_disp ({cfg.max_disp}) must be divisible by feature_stride ({cfg.feature_stride})"
            )
        self.cfg = cfg
        self.max_disp = cfg.max_disp
        self.feature_stride = cfg.feature_stride
        self.disp_levels = cfg.max_disp // cfg.feature_stride  # disparity bins at stride 4

        self.feature_extractor = FeatureExtractor(out_channels=cfg.feat_channels)
        if cfg.use_cuda_extension:
            self.correlation = CorrelationVolume(max_disp=self.disp_levels)
        else:
            self.correlation = None  # use pure PyTorch path
        self.aggregator = UNetAggregator(
            max_disp=self.disp_levels,
            base_channels=cfg.unet_base_channels,
            predict_uncertainty=cfg.predict_uncertainty,
        )

        # Pre-computed disparity index buffer for soft-argmin
        self.register_buffer(
            "disp_index",
            torch.arange(self.disp_levels, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )

    # ----------------------------- forward ----------------------------------
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            left, right: (B, 3, H, W) float tensors in [0, 1] (or normalized).
        Returns:
            dict with keys:
                "disparity": (B, H, W) predicted disparity in image pixels.
                "disparity_low": (B, H/4, W/4) low-res disparity (for aux loss).
                "cost": (B, D, H/4, W/4) aggregated cost volume (debug/inspect).
                "log_b": (B, H, W) per-pixel log-uncertainty (Laplace scale),
                         in *pixel* units. Only present if the config has
                         ``predict_uncertainty=True``.
                "log_b_low": (B, H/4, W/4) low-res log_b for aux loss. Same
                         caveat as above.
        """
        if left.shape != right.shape:
            raise ValueError("Left/right must have identical shapes")

        # 1) Features at stride 4 ---------------------------------------------
        fL = self.feature_extractor(left)
        fR = self.feature_extractor(right)

        # 2) Cost volume ------------------------------------------------------
        if self.correlation is not None:
            cost = self.correlation(fL, fR)
        else:
            cost = build_correlation_volume(fL, fR, self.disp_levels)

        # 3) 2D U-Net aggregation --------------------------------------------
        agg = self.aggregator(cost)
        cost = agg["cost"]
        log_b_low = agg["log_b"]  # None or (B, 1, H/4, W/4)

        # 4) Soft-argmin disparity regression --------------------------------
        # cost: (B, D, H/4, W/4). We want a probability distribution over D.
        # Higher correlation = better match, so we softmax over D.
        # IMPORTANT: force fp32 here. In AMP/fp16, the residual added by the
        # U-Net aggregator can push individual cost values past fp16's max
        # (~65504); exp(cost) then becomes inf and the softmax produces NaN,
        # poisoning training irrecoverably.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            cost_fp32 = cost.float()
            prob = F.softmax(cost_fp32, dim=1)
            disp_low = (prob * self.disp_index).sum(dim=1, keepdim=True)  # (B,1,H/4,W/4)

        # 5) Upsample + scale -------------------------------------------------
        H, W = left.shape[-2:]
        disp_full = F.interpolate(
            disp_low, size=(H, W), mode="bilinear", align_corners=False
        ) * float(self.feature_stride)
        disp_full = disp_full.squeeze(1)  # (B, H, W)

        result = {
            "disparity": disp_full,
            "disparity_low": disp_low.squeeze(1),
            "cost": cost,
        }

        if log_b_low is not None:
            # log_b is in PIXEL units. Disparity is upsampled and scaled by
            # feature_stride (a pixel at stride 4 corresponds to 4 px of disp);
            # the uncertainty therefore also scales by feature_stride. We add
            # log(feature_stride) to log_b after upsampling.
            import math as _math
            log_b_full = F.interpolate(
                log_b_low, size=(H, W), mode="bilinear", align_corners=False
            ) + _math.log(float(self.feature_stride))
            result["log_b"] = log_b_full.squeeze(1)
            result["log_b_low"] = log_b_low.squeeze(1)

        return result

    # ----------------------------- helpers ----------------------------------
    @torch.no_grad()
    def predict_depth(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        baseline_m: float,
        focal_px: float,
        min_disp: float = 1e-3,
    ) -> torch.Tensor:
        """Convenience: predict depth (meters) from a stereo pair.

        depth = (baseline * focal) / disparity_pixels
        """
        disp = self.forward(left, right)["disparity"].clamp_min(min_disp)
        return (baseline_m * focal_px) / disp


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = StereoUNetConfig(max_disp=192, feat_channels=64, unet_base_channels=32,
                           use_cuda_extension=False)
    net = StereoUNet(cfg)
    L = torch.randn(1, 3, 256, 512)
    R = torch.randn(1, 3, 256, 512)
    out = net(L, R)
    print("disparity:", out["disparity"].shape, "range:", out["disparity"].min().item(),
          out["disparity"].max().item())
    print("disparity_low:", out["disparity_low"].shape)
    print("cost:", out["cost"].shape)
    print("Trainable params:", count_params(net))

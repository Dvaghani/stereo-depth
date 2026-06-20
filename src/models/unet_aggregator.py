"""
2D U-Net for cost-volume aggregation.

The cost volume from ``cost_volume.py`` has shape ``(B, D, H, W)`` — we treat
the disparity dimension ``D`` as input *channels* and run a standard 2D U-Net
to aggregate context across the full frame, smoothing the per-pixel matching
scores. This is the key cost-saving choice in the thesis: a 2D U-Net instead
of the 3D regularization used by FoundationStereo / IGEV-Stereo / PSMNet,
which would not fit on the Jetson Nano's 4 GB VRAM.

The output is the *refined* cost volume, same shape ``(B, D, H, W)``. A
soft-argmin head downstream converts it to a disparity map.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class Down(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _double_conv(in_c, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample + concat with skip + double conv. Uses bilinear upsampling
    rather than transposed conv for slightly better Jetson throughput and to
    avoid checkerboard artifacts in disparity space."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _double_conv(in_c + skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd input sizes — pad x to skip's spatial size.
        if x.shape[-2:] != skip.shape[-2:]:
            dy = skip.size(-2) - x.size(-2)
            dx = skip.size(-1) - x.size(-1)
            x = F.pad(x, [0, dx, 0, dy])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetAggregator(nn.Module):
    """
    Args:
        max_disp: number of disparity channels D (input channels).
        base_channels: width of the first encoder stage. The U-Net follows
            the classic doubling pattern: base, 2*base, 4*base, 8*base.
        predict_uncertainty: if True, build a small uncertainty head that
            outputs per-pixel log(b) where b is the Laplace scale parameter.
            Adds ~10k params at base_channels=32.

    Input:  (B, D, H, W) raw correlation volume.

    Output: a dict with keys
        "cost":      (B, D, H, W) aggregated cost volume (residual added).
        "log_b":     (B, 1, H, W) per-pixel log(uncertainty), OR None when
                     ``predict_uncertainty`` is False.

    Returning a dict (instead of a bare tensor) is mildly less ergonomic but
    keeps the API stable as we add new outputs later (e.g. semantic mask).
    """

    def __init__(self, max_disp: int, base_channels: int = 32,
                 predict_uncertainty: bool = False):
        super().__init__()
        self.max_disp = max_disp
        self.predict_uncertainty = predict_uncertainty
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.inc = _double_conv(max_disp, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)

        # Bottleneck — slightly wider receptive field via dilation
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        self.up1 = Up(c4, c3, c3)
        self.up2 = Up(c3, c2, c2)
        self.up3 = Up(c2, c1, c1)

        # Output head: 1x1 conv back to disparity channels (residual to the
        # raw correlation, which improves training stability).
        self.outc = nn.Conv2d(c1, max_disp, kernel_size=1)

        # Optional uncertainty head. Tapped from the same decoder features
        # the cost head uses, so it sees both matching evidence and the
        # semantic context the U-Net has built up. Predicts log(b) — log
        # is used for numerical stability; b must stay positive so we
        # exponentiate at the loss site, not here.
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, 1, kernel_size=1),
            )
        else:
            self.uncertainty_head = None

    def forward(self, cost: torch.Tensor) -> dict:
        x1 = self.inc(cost)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.bottleneck(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        residual = self.outc(x)
        out = {"cost": cost + residual, "log_b": None}
        if self.uncertainty_head is not None:
            out["log_b"] = self.uncertainty_head(x)  # (B, 1, H, W) at stride 4
        return out


if __name__ == "__main__":
    net = UNetAggregator(max_disp=48, base_channels=32, predict_uncertainty=True)
    dummy = torch.randn(1, 48, 64, 128)
    out = net(dummy)
    print("UNetAggregator cost:", out["cost"].shape,
          "log_b:", None if out["log_b"] is None else out["log_b"].shape)
    print("Params:", sum(p.numel() for p in net.parameters()))

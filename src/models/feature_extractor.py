"""
Shared 2D CNN feature extractor (siamese) for stereo matching.

Produces stride-4 feature maps used to build the cost volume. Kept intentionally
lightweight so it can run on a Jetson Nano (128 CUDA cores, 4 GB VRAM).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_c: int, out_c: int, k: int = 3, s: int = 1, d: int = 1) -> nn.Sequential:
    pad = ((k - 1) * d) // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=pad, dilation=d, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    """Pre-activation residual block."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv_bn_relu(in_c, out_c, k=3, s=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return F.relu(out + identity, inplace=True)


class FeatureExtractor(nn.Module):
    """
    Siamese feature extractor.

    Input:  (B, 3, H, W) image tensor in [0, 1] (per-channel normalized externally)
    Output: (B, C_out, H/4, W/4) feature map

    The same module instance is called twice with shared weights — once on the
    left image and once on the right image — to keep the disparity computation
    consistent.
    """

    def __init__(self, out_channels: int = 64):
        super().__init__()
        # Stem: stride 2
        self.stem = nn.Sequential(
            conv_bn_relu(3, 32, k=3, s=2),  # H/2
            conv_bn_relu(32, 32, k=3, s=1),
            conv_bn_relu(32, 32, k=3, s=1),
        )
        # Stage 1: stride 4
        self.layer1 = nn.Sequential(
            BasicBlock(32, 48, stride=2),  # H/4
            BasicBlock(48, 48),
            BasicBlock(48, 48),
        )
        # Stage 2: keep stride 4 but expand channels with dilation for context
        self.layer2 = nn.Sequential(
            BasicBlock(48, 64),
            BasicBlock(64, 64),
        )
        self.project = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.project(x)
        # L2-normalize features along channel axis: makes dot-product correlation
        # behave like a cosine similarity, which improves matching stability.
        x = F.normalize(x, p=2, dim=1)
        return x


if __name__ == "__main__":
    # Quick sanity check
    net = FeatureExtractor(out_channels=64)
    dummy = torch.randn(2, 3, 256, 512)
    feats = net(dummy)
    print("FeatureExtractor output:", feats.shape)
    print("Params:", sum(p.numel() for p in net.parameters()))

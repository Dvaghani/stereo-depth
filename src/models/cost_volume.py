"""
Cost-volume construction.

Implements two equivalent code paths:

1. ``build_correlation_volume`` — a pure PyTorch fallback that works anywhere
   (CPU or GPU, no compilation required). Used by default and for unit tests.

2. ``CorrelationVolume`` module — tries to load the compiled CUDA/C++ extension
   ``stereo_corr_cuda`` (see csrc/). If unavailable it falls back to the
   PyTorch implementation. This lets the network train on a workstation and
   later be deployed on the Jetson Nano with the optimized kernel.

The volume is "compressed": we use a 1-channel dot-product correlation
(``sum over feature channels``) rather than a feature-concatenation 4D volume,
which keeps memory at O(D * H/4 * W/4) instead of O(D * C * H/4 * W/4).
"""
from __future__ import annotations

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_correlation_volume(
    feat_left: torch.Tensor,
    feat_right: torch.Tensor,
    max_disp: int,
) -> torch.Tensor:
    """
    Pure-PyTorch reference implementation.

    Args:
        feat_left, feat_right: (B, C, H, W) feature maps from the siamese
            feature extractor. Assumed L2-normalized along C so the dot
            product is a cosine similarity in [-1, 1].
        max_disp: maximum disparity in feature-map pixels (i.e. at stride 4
            this corresponds to ``4 * max_disp`` pixels in the input image).

    Returns:
        cost: (B, D, H, W) tensor. ``cost[:, d]`` is the correlation between
            ``feat_left`` and ``feat_right`` shifted by ``d`` pixels to the
            right (i.e. the right view shifted left).
    """
    if feat_left.shape != feat_right.shape:
        raise ValueError("Left/right feature maps must have identical shapes.")
    B, C, H, W = feat_left.shape
    D = int(max_disp)
    if D < 1:
        raise ValueError("max_disp must be >= 1")

    # Allocate volume on the same device/dtype as the features.
    cost = feat_left.new_zeros((B, D, H, W))

    # d=0 is the trivial case: no shift.
    cost[:, 0] = (feat_left * feat_right).sum(dim=1)

    for d in range(1, D):
        # Shift right features by d pixels to the right (zero-pad on the left).
        # For each left pixel at column x, this looks at right pixel x - d.
        cost[:, d, :, d:] = (
            feat_left[:, :, :, d:] * feat_right[:, :, :, :-d]
        ).sum(dim=1)
        # Columns [0, d) have no valid match — they stay at 0.

    return cost


class CorrelationVolume(nn.Module):
    """
    Wrapper module that prefers the compiled CUDA kernel when available.

    On first instantiation we try ``import stereo_corr_cuda``. If the import
    fails (extension not built, or running on CPU only), we fall back to the
    PyTorch implementation and emit a warning once.
    """

    _warned: bool = False

    def __init__(self, max_disp: int):
        super().__init__()
        self.max_disp = int(max_disp)
        self._cuda_ext = None
        try:
            import stereo_corr_cuda  # type: ignore

            self._cuda_ext = stereo_corr_cuda
        except Exception as exc:  # ImportError or OSError on load
            if not CorrelationVolume._warned:
                warnings.warn(
                    f"stereo_corr_cuda extension not loaded ({exc!r}); "
                    "falling back to the PyTorch correlation. Build the "
                    "extension with `python csrc/setup.py build_ext --inplace` "
                    "for full speed on Jetson.",
                    RuntimeWarning,
                )
                CorrelationVolume._warned = True

    def forward(self, fl: torch.Tensor, fr: torch.Tensor) -> torch.Tensor:
        if self._cuda_ext is not None and fl.is_cuda and fl.dtype == torch.float32:
            # Pre-allocate output and call the kernel.
            B, _, H, W = fl.shape
            out = fl.new_empty((B, self.max_disp, H, W))
            self._cuda_ext.forward(fl.contiguous(), fr.contiguous(), out, self.max_disp)
            return out
        return build_correlation_volume(fl, fr, self.max_disp)


if __name__ == "__main__":
    fl = torch.randn(1, 32, 64, 128)
    fr = torch.randn(1, 32, 64, 128)
    fl = F.normalize(fl, dim=1)
    fr = F.normalize(fr, dim=1)
    vol = build_correlation_volume(fl, fr, max_disp=48)
    print("Cost volume shape:", vol.shape)
    print("Cost volume range:", vol.min().item(), vol.max().item())

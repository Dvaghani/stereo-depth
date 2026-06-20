"""
Loss functions for stereo disparity training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def disparity_smooth_l1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Masked smooth-L1 loss on disparity.

    Args:
        pred: (B, H, W) predicted disparity in pixels.
        gt:   (B, H, W) ground-truth disparity in pixels (0 = invalid).
        valid: (B, H, W) binary mask (1 = use this pixel).
        beta: smooth-L1 transition point in pixels.

    Returns:
        Scalar loss tensor (mean over valid pixels). Returns 0 if no valid
        pixels exist in the batch, so empty batches don't crash training.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    mask = (valid > 0.5).float()
    n = mask.sum().clamp_min(1.0)
    diff = (pred - gt).abs()
    loss_elem = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return (loss_elem * mask).sum() / n


def laplace_nll_loss(
    pred: torch.Tensor,
    log_b: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    log_b_min: float = -5.0,
    log_b_max: float = 5.0,
) -> torch.Tensor:
    """Negative log-likelihood under a Laplace observation model.

        p(gt | pred, b) = (1 / 2b) * exp(-|gt - pred| / b)
        -log p = |gt - pred| / b + log(b) + log(2)

    The log(2) is a constant and dropped. We parameterize the network to
    output ``log_b`` so b > 0 is enforced automatically and the |.|/b term
    can't blow up at b -> 0.

    Args:
        pred:  (B, H, W) predicted disparity in pixels.
        log_b: (B, H, W) predicted log-uncertainty (Laplace scale, in pixels,
               in log-space). Must be the same shape as ``pred``.
        gt:    (B, H, W) GT disparity in pixels.
        valid: (B, H, W) binary mask.
        log_b_min, log_b_max: clamp log_b to a sane range to prevent runaway
               growth. log_b_min=-5 => b_min=e^-5 ~= 0.007 px (very confident);
               log_b_max=5 => b_max=e^5 ~= 148 px (we admit total ignorance).

    Returns:
        Scalar loss (mean over valid pixels).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if pred.shape != log_b.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs log_b {log_b.shape}")
    mask = (valid > 0.5).float()
    n = mask.sum().clamp_min(1.0)
    log_b = log_b.clamp(min=log_b_min, max=log_b_max)
    inv_b = torch.exp(-log_b)
    diff = (pred - gt).abs()
    loss_elem = diff * inv_b + log_b
    return (loss_elem * mask).sum() / n


def multi_scale_loss(
    pred_full: torch.Tensor,
    pred_low: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    weight_low: float = 0.3,
    feature_stride: int = 4,
    log_b_full: torch.Tensor | None = None,
    log_b_low: torch.Tensor | None = None,
) -> torch.Tensor:
    """Total loss = main(full) + weight_low * main(low_res).

    Auxiliary low-res loss helps train the cost-volume aggregator more directly.

    If ``log_b_full`` is provided, the loss switches from smooth-L1 to Laplace
    NLL. ``log_b_low`` (the same quantity at the low-res grid, BEFORE the
    +log(stride) shift) should also be provided so the auxiliary loss can be
    Laplace too. If only ``log_b_full`` is provided, we fall back to smooth-L1
    for the aux term.
    """
    # Downsample GT and valid mask to the low-res grid (shared by both paths).
    gt_low = F.avg_pool2d(gt.unsqueeze(1), kernel_size=feature_stride).squeeze(1) / feature_stride
    valid_low = F.max_pool2d(valid.unsqueeze(1), kernel_size=feature_stride).squeeze(1)

    if log_b_full is None:
        main = disparity_smooth_l1(pred_full, gt, valid)
        aux = disparity_smooth_l1(pred_low, gt_low, valid_low)
        return main + weight_low * aux

    # Laplace NLL path
    main = laplace_nll_loss(pred_full, log_b_full, gt, valid)
    if log_b_low is not None:
        aux = laplace_nll_loss(pred_low, log_b_low, gt_low, valid_low)
    else:
        aux = disparity_smooth_l1(pred_low, gt_low, valid_low)
    return main + weight_low * aux

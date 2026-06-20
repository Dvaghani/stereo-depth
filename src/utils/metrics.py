"""
KITTI / Middlebury evaluation metrics.

D1-all is the official KITTI 2015 metric: percentage of pixels with absolute
disparity error > 3 px AND relative error > 5 %.

EPE (End-Point Error) is the mean absolute disparity error.

Bad-N is the fraction of pixels with absolute disparity error > N.
"""
from __future__ import annotations

from typing import Dict

import torch


def end_point_error(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid > 0.5
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (pred[mask] - gt[mask]).abs().mean()


def bad_pixel_ratio(
    pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, threshold: float
) -> torch.Tensor:
    mask = valid > 0.5
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    err = (pred[mask] - gt[mask]).abs()
    return (err > threshold).float().mean()


def d1_all(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """KITTI D1-all: |err| > 3px AND |err| / gt > 0.05."""
    mask = valid > 0.5
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    err = (pred[mask] - gt[mask]).abs()
    gt_v = gt[mask].clamp_min(1e-3)
    bad = (err > 3.0) & ((err / gt_v) > 0.05)
    return bad.float().mean()


def compute_kitti_metrics(
    pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor
) -> Dict[str, float]:
    """Return a dict of standard stereo metrics. All values are Python floats."""
    return {
        "EPE": end_point_error(pred, gt, valid).item(),
        "D1-all": d1_all(pred, gt, valid).item() * 100.0,  # in %
        "bad-1": bad_pixel_ratio(pred, gt, valid, 1.0).item() * 100.0,
        "bad-2": bad_pixel_ratio(pred, gt, valid, 2.0).item() * 100.0,
        "bad-3": bad_pixel_ratio(pred, gt, valid, 3.0).item() * 100.0,
    }

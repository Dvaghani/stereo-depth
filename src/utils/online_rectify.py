"""
Online stereo rectification refinement: estimate and remove the residual
VERTICAL offset between an already-rectified left/right pair, live.

Why this exists
---------------
A metric stereo calibration fixes ONE relative camera pose. In practice the rig
flexes — hand pressure on a 3D-printed mount, thermal creep, and (critically for
a drone) rotor vibration and airframe flex. That introduces a small, slowly
time-varying VERTICAL misalignment on top of the static calibration. A
horizontal stereo matcher (AANet) assumes correspondences lie on the same
scanline, so even ~2 px of residual vertical offset biases every match.

This is NOT a replacement for calibration (it cannot recover metric scale,
distortion, or horizontal geometry). It is a thin refinement layer: measure the
leftover vertical shift from the live frames and cancel it before matching.

Design
------
* Estimate with SIFT on a DOWNSCALED pair (half-res ≈ ±0.3 px accuracy, ~180 ms)
  — accurate and robust, unlike ORB which drifts on low-texture indoor scenes.
* The flex is sub-Hz, so we don't estimate every frame: `OnlineVerticalAligner`
  re-estimates every N frames and exponentially smooths, rejecting estimates
  with too few matches or implausibly large jumps. Amortized cost is negligible.
* Correction is a sub-pixel vertical translation of the right image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


def estimate_vertical_offset(
    rectL: np.ndarray,
    rectR: np.ndarray,
    scale: float = 0.5,
    nfeat: int = 1500,
    max_disp: float = 400.0,
    min_matches: int = 20,
    ratio: float = 0.8,
) -> Tuple[Optional[float], int]:
    """Robust median vertical residual (yL - yR) between a rectified pair.

    Returns (offset_px, n_matches). offset is None if too few matches.
    `offset` is in FULL-resolution pixels regardless of `scale`.
    """
    if rectL.ndim == 3:
        Lg = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        Rg = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    else:
        Lg, Rg = rectL, rectR
    if scale != 1.0:
        Lg = cv2.resize(Lg, None, fx=scale, fy=scale)
        Rg = cv2.resize(Rg, None, fx=scale, fy=scale)

    sift = cv2.SIFT_create(nfeat)
    k1, d1 = sift.detectAndCompute(Lg, None)
    k2, d2 = sift.detectAndCompute(Rg, None)
    if d1 is None or d2 is None:
        return None, 0
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)

    dy = []
    md = max_disp * scale
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            x1, y1 = k1[m.queryIdx].pt
            x2, y2 = k2[m.trainIdx].pt
            if 0 < (x1 - x2) < md:
                dy.append((y1 - y2) / scale)
    if len(dy) < min_matches:
        return None, len(dy)

    dy = np.asarray(dy)
    med = np.median(dy)
    mad = np.median(np.abs(dy - med))
    dy = dy[np.abs(dy - med) < 4 * mad + 1e-6]
    return float(np.median(dy)), int(len(dy))


def apply_vertical_shift(img: np.ndarray, offset: float) -> np.ndarray:
    """Translate `img` vertically so a feature at row y moves to y + offset.

    To cancel a measured residual dy = yL - yR, shift the RIGHT image by
    `offset = dy` (then yR + dy aligns with yL). Sub-pixel via bilinear warp.
    """
    if abs(offset) < 1e-3:
        return img
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, offset]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


@dataclass
class OnlineVerticalAligner:
    """Stateful per-stream vertical aligner for a live rectified feed.

    Usage (per frame):
        aligner = OnlineVerticalAligner(period=15)
        ...
        rectR_corrected = aligner.process(rectL, rectR)

    It re-estimates every `period` frames, exponentially smooths, and rejects
    bad estimates (too few matches, or a jump larger than `max_jump` px from the
    smoothed value — a transient mismatch rather than real flex).
    """
    period: int = 15
    scale: float = 0.5
    nfeat: int = 1500
    ema: float = 0.5           # smoothing weight for a new estimate
    max_jump: float = 4.0      # px; reject estimates farther than this from state
    min_matches: int = 20
    offset: float = 0.0        # current smoothed vertical offset (full-res px)
    initialized: bool = False
    _count: int = field(default=0, repr=False)
    last_n: int = field(default=0, repr=False)

    def update_estimate(self, rectL: np.ndarray, rectR: np.ndarray) -> Optional[float]:
        """Force a re-estimate now; update the smoothed offset. Returns raw est."""
        est, n = estimate_vertical_offset(
            rectL, rectR, scale=self.scale, nfeat=self.nfeat,
            min_matches=self.min_matches)
        self.last_n = n
        if est is None:
            return None
        if not self.initialized:
            self.offset = est
            self.initialized = True
        elif abs(est - self.offset) <= self.max_jump:
            self.offset = (1 - self.ema) * self.offset + self.ema * est
        # else: outlier, keep previous smoothed offset
        return est

    def process(self, rectL: np.ndarray, rectR: np.ndarray) -> np.ndarray:
        """Return a vertically-corrected copy of rectR for this frame."""
        if self._count % self.period == 0:
            self.update_estimate(rectL, rectR)
        self._count += 1
        return apply_vertical_shift(rectR, self.offset)

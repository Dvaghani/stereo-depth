"""
Stereo-aware transforms.

A few key constraints:
- Spatial transforms (crop, flip, scale) must be applied identically to
  ``left``, ``right``, and ``disparity`` — otherwise the geometric meaning
  of disparity breaks.
- Horizontal flip is NOT safe for stereo pairs unless you also swap left/right
  AND negate disparity. We only allow vertical-free horizontal-free crops here.
- Color jitter is applied independently to L and R (asymmetric augmentation)
  which improves robustness, following common practice (RAFT-Stereo, PSMNet).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import functional as TF


# ImageNet stats — used because most pretrained backbones expect them.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> float32 CHW in [0, 1]."""
    if img.ndim == 2:
        img = img[..., None]
    # .copy() ensures the buffer is writable, silencing torch's repeated
    # "given NumPy array is not writable" UserWarning in DataLoader workers.
    t = torch.from_numpy(np.ascontiguousarray(img).copy()).permute(2, 0, 1).contiguous()
    return t.float() / 255.0


@dataclass
class StereoTransform:
    """Configurable stereo transform pipeline.

    Args:
        crop_size: (H, W). If ``None``, returns the full image (validation).
        color_jitter: brightness/contrast/saturation/hue jitter applied
            independently to L and R.
        normalize: apply ImageNet mean/std normalization.
    """

    crop_size: Tuple[int, int] | None = (256, 512)
    color_jitter: float = 0.4
    normalize: bool = True
    training: bool = True

    def __call__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        disparity: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L = _to_tensor(left)
        R = _to_tensor(right)
        D = torch.from_numpy(np.ascontiguousarray(disparity)).float()
        if D.ndim == 3:
            D = D.squeeze(-1)

        # Random crop (training) or center crop (validation)
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            H, W = L.shape[-2:]
            if H < crop_h or W < crop_w:
                # Pad reflectively so we can always satisfy the crop.
                pad_h = max(0, crop_h - H)
                pad_w = max(0, crop_w - W)
                L = TF.pad(L, [0, 0, pad_w, pad_h], padding_mode="reflect")
                R = TF.pad(R, [0, 0, pad_w, pad_h], padding_mode="reflect")
                D = torch.nn.functional.pad(D.unsqueeze(0).unsqueeze(0),
                                            (0, pad_w, 0, pad_h)).squeeze()
                H, W = L.shape[-2:]
            if self.training:
                y = random.randint(0, H - crop_h)
                x = random.randint(0, W - crop_w)
            else:
                y = (H - crop_h) // 2
                x = (W - crop_w) // 2
            L = L[:, y : y + crop_h, x : x + crop_w]
            R = R[:, y : y + crop_h, x : x + crop_w]
            D = D[y : y + crop_h, x : x + crop_w]

        # Asymmetric color jitter (training only).
        # Applied independently to L and R to simulate real-world camera
        # variation (exposure, white-balance, sensor response) — follows
        # RAFT-Stereo / PSMNet practice. Saturation + hue jitter are added
        # on top of brightness/contrast to close the Middlebury→Brio domain
        # gap (Brio has different color response than calibrated lab cameras).
        if self.training and self.color_jitter > 0:
            cj = self.color_jitter
            for img in (L, R):
                pass  # applied below per-image
            def _jitter(img):
                img = TF.adjust_brightness(img, 1.0 + random.uniform(-cj, cj))
                img = TF.adjust_contrast(img,   1.0 + random.uniform(-cj, cj))
                img = TF.adjust_saturation(img, 1.0 + random.uniform(-cj * 0.8, cj * 0.8))
                img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))
                # Random gamma [0.7, 1.5] — simulates exposure/sensor variation
                gamma = random.uniform(0.7, 1.5)
                img = img.clamp(1e-6, 1.0).pow(gamma)
                return img.clamp(0, 1)
            L = _jitter(L)
            R = _jitter(R)

        if self.normalize:
            L = TF.normalize(L, IMAGENET_MEAN, IMAGENET_STD)
            R = TF.normalize(R, IMAGENET_MEAN, IMAGENET_STD)

        return L, R, D

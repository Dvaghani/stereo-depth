"""
Middlebury 2014 stereo dataset.

Expected directory layout (the "training" set with ground truth):

    root/
        Adirondack/
            im0.png         # left
            im1.png         # right
            disp0.pfm       # ground-truth disparity (PFM, inf = invalid)
            calib.txt       # camera intrinsics (optional, not parsed here)
        Backpack/
            ...
        ...

Some Middlebury images are very large (≥ 2k px) — by default we downsample
on load to keep memory reasonable. Pass ``downsample=1`` to disable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transforms import StereoTransform
from ..utils.io import read_pfm


class Middlebury2014Stereo(Dataset):
    """
    Args:
        root: dataset directory containing scene subdirectories.
        variant: which calibration variant to load.
            "imperfect" — practical calibration with residual rectification
                error (1-3 px typical). Closer to real cameras like the
                Logitech Brio used in this thesis. RECOMMENDED for deployment-
                oriented training.
            "perfect" — ideal pixel-perfect rectification. Used by the
                Middlebury leaderboard convention for benchmark numbers.
            "both" — load both variants of every scene (doubles dataset size).
        transform, downsample, return_path: as before.
    """

    VALID_VARIANTS = ("imperfect", "perfect", "both")

    def __init__(
        self,
        root: str | Path,
        transform: Optional[StereoTransform] = None,
        downsample: int = 2,
        return_path: bool = False,
        variant: str = "imperfect",
    ):
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {self.VALID_VARIANTS}, got {variant!r}"
            )
        self.root = Path(root)
        self.downsample = max(1, int(downsample))
        self.variant = variant

        # Directory naming: '<Scene>-perfect' and '<Scene>-imperfect'.
        all_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if variant == "both":
            scenes = all_dirs
        else:
            suffix = f"-{variant}"
            scenes = [p for p in all_dirs if p.name.endswith(suffix)]

        self.samples = []
        for scene in scenes:
            l = scene / "im0.png"
            r = scene / "im1.png"
            d = scene / "disp0.pfm"
            if l.exists() and r.exists() and d.exists():
                self.samples.append((l, r, d))
        if not self.samples:
            raise FileNotFoundError(
                f"No Middlebury scenes (variant={variant}) with "
                f"im0/im1/disp0 found under {self.root}"
            )
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _downsample(img: np.ndarray, factor: int) -> np.ndarray:
        if factor == 1:
            return img

        # PIL handles RGB and L modes correctly; disparity is float so we use
        # numpy slicing (decimation) instead — bilinear would corrupt disp values.
        if img.dtype == np.float32:
            return img[::factor, ::factor]
        H, W = img.shape[:2]
        new_size = (W // factor, H // factor)
        mode = "RGB" if img.ndim == 3 else "L"
        return np.asarray(Image.fromarray(img, mode=mode).resize(new_size, Image.BILINEAR))

    def __getitem__(self, idx: int):
        lp, rp, dp = self.samples[idx]
        left = np.asarray(Image.open(lp).convert("RGB"))
        right = np.asarray(Image.open(rp).convert("RGB"))
        disparity, _ = read_pfm(dp)
        # Middlebury marks invalid as inf
        disparity = np.where(np.isfinite(disparity), disparity, 0.0).astype(np.float32)

        if self.downsample > 1:
            left = self._downsample(left, self.downsample)
            right = self._downsample(right, self.downsample)
            disparity = self._downsample(disparity, self.downsample) / float(self.downsample)

        if self.transform is not None:
            L, R, D = self.transform(left, right, disparity)
        else:
            from .transforms import _to_tensor
            import torch
            L = _to_tensor(left)
            R = _to_tensor(right)
            D = torch.from_numpy(disparity).float()

        sample = {"left": L, "right": R, "disparity": D, "valid": (D > 0).float()}
        if self.return_path:
            sample["path"] = str(lp.parent)
        return sample

"""
SceneFlow FlyingThings3D subset stereo dataset.

Expected directory layout (after extracting the official archives):

    root/
        train/
            image_clean/
                left/   0000000.png  0000001.png  ...
                right/  0000000.png  ...
            disparity/
                left/   0000000.pfm  0000001.pfm  ...
        val/
            image_clean/
                left/   ...
                right/  ...
            disparity/
                left/   ...

Pass split='train' or split='val'.  The dataset has 21 818 train and
4 248 val samples.  Only the LEFT disparity is loaded (right is discarded).

Disparities above max_disp are marked invalid (valid mask = 0).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transforms import StereoTransform
from ..utils.io import read_pfm


class SceneFlowStereo(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[StereoTransform] = None,
        max_disp: float = 192.0,
        return_path: bool = False,
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.root = Path(root)
        self.max_disp = max_disp
        self.transform = transform
        self.return_path = return_path

        img_dir  = self.root / split / "image_clean"
        disp_dir = self.root / split / "disparity"
        left_imgs  = sorted((img_dir  / "left").glob("*.png"))
        if not left_imgs:
            raise FileNotFoundError(
                f"No PNG files found under {img_dir / 'left'}. "
                f"Check that root points to FlyingThings3D_subset/ and "
                f"split='{split}' is correct."
            )

        self.samples = []
        for lp in left_imgs:
            rp = img_dir  / "right" / lp.name
            dp = disp_dir / "left"  / (lp.stem + ".pfm")
            if rp.exists() and dp.exists():
                self.samples.append((lp, rp, dp))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        lp, rp, dp = self.samples[idx]
        left  = np.asarray(Image.open(lp).convert("RGB"))
        right = np.asarray(Image.open(rp).convert("RGB"))
        disparity, _ = read_pfm(dp)
        disparity = np.abs(disparity.astype(np.float32))  # FlyingThings3D stores as negative floats
        disparity = np.where(
            (disparity > 0) & (disparity < self.max_disp), disparity, 0.0
        )

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
            sample["path"] = str(lp)
        return sample

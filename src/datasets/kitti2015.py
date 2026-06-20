"""
KITTI 2015 stereo dataset.

Expected directory layout (the official KITTI stereo 2015 archive):

    root/
        training/
            image_2/    # left  (000000_10.png, 000000_11.png, ...)
            image_3/    # right
            disp_occ_0/ # ground-truth disparity (16-bit PNG, /256.0)
        testing/
            image_2/
            image_3/

Disparity values of 0 indicate "invalid / no ground truth" and must be
masked out during loss/metric computation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transforms import StereoTransform
from ..utils.io import read_kitti_disparity_png


class KITTI2015Stereo(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "training",
        transform: Optional[StereoTransform] = None,
        return_path: bool = False,
    ):
        self.root = Path(root)
        if split not in {"training", "testing"}:
            raise ValueError("split must be 'training' or 'testing'")
        self.split = split
        self.has_gt = split == "training"

        left_dir = self.root / split / "image_2"
        right_dir = self.root / split / "image_3"
        # Only frame "_10" has ground truth in KITTI 2015 (it is the reference frame).
        left_files = sorted(p for p in left_dir.glob("*_10.png"))
        if not left_files:
            raise FileNotFoundError(
                f"No '*_10.png' files in {left_dir}. Check the KITTI layout."
            )
        self.samples = []
        for lp in left_files:
            rp = right_dir / lp.name
            dp = (self.root / split / "disp_occ_0" / lp.name) if self.has_gt else None
            if not rp.exists():
                continue
            if self.has_gt and not dp.exists():
                continue
            self.samples.append((lp, rp, dp))

        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        lp, rp, dp = self.samples[idx]
        left = np.asarray(Image.open(lp).convert("RGB"))
        right = np.asarray(Image.open(rp).convert("RGB"))
        if dp is not None:
            disparity = read_kitti_disparity_png(dp)
        else:
            disparity = np.zeros(left.shape[:2], dtype=np.float32)

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

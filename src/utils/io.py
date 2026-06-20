"""
I/O helpers: PFM read/write, disparity ↔ depth, disparity visualization.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------- PFM (Middlebury) ------------------------------
def read_pfm(path: str | Path) -> Tuple[np.ndarray, float]:
    """Read a Middlebury / Sintel-style PFM file.

    Returns:
        data: (H, W) or (H, W, 3) float32 array. Invalid pixels in Middlebury
            ground truth are stored as ``inf`` — callers should mask them out.
        scale: PFM scale (negative => little-endian, positive => big-endian).
    """
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        if header == b"PF":
            color = True
        elif header == b"Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {path}")

        dim_line = f.readline().decode("ascii").strip()
        while dim_line.startswith("#"):
            dim_line = f.readline().decode("ascii").strip()
        m = re.match(r"^(\d+)\s+(\d+)\s*$", dim_line)
        if not m:
            raise ValueError(f"Malformed PFM dimensions: {dim_line!r}")
        width, height = int(m.group(1)), int(m.group(2))

        scale_line = f.readline().decode("ascii").strip()
        scale = float(scale_line)
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        n_channels = 3 if color else 1
        data = np.fromfile(f, dtype=endian + "f", count=width * height * n_channels)
        data = data.reshape((height, width, n_channels)) if color else data.reshape((height, width))
        data = np.flipud(data)  # PFM stores bottom-to-top
        return data.astype(np.float32), scale


def write_pfm(path: str | Path, data: np.ndarray, scale: float = 1.0) -> None:
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    if data.ndim == 2:
        color = False
    elif data.ndim == 3 and data.shape[2] == 3:
        color = True
    else:
        raise ValueError("PFM data must be HxW or HxWx3")
    with open(path, "wb") as f:
        f.write(b"PF\n" if color else b"Pf\n")
        f.write(f"{data.shape[1]} {data.shape[0]}\n".encode("ascii"))
        endian = data.dtype.byteorder
        if endian == "<" or (endian == "=" and struct.pack("=I", 1) == b"\x01\x00\x00\x00"):
            scale_signed = -abs(scale)
        else:
            scale_signed = abs(scale)
        f.write(f"{scale_signed}\n".encode("ascii"))
        np.flipud(data).tofile(f)


# ---------------------------- KITTI disparity -------------------------------
def read_kitti_disparity_png(path: str | Path) -> np.ndarray:
    """KITTI stores disparity in 16-bit PNG: value/256.0 in pixels, 0=invalid."""
    from PIL import Image
    arr = np.asarray(Image.open(path), dtype=np.float32)
    disp = arr / 256.0
    return disp  # invalid pixels are 0.0


# ---------------------------- disparity ↔ depth -----------------------------
def disparity_to_depth(
    disparity: np.ndarray | "torch.Tensor",
    baseline_m: float,
    focal_px: float,
    min_disp: float = 1e-3,
):
    """depth = baseline * focal / disparity. Returns same backend as input."""
    try:
        import torch
        if isinstance(disparity, torch.Tensor):
            return (baseline_m * focal_px) / disparity.clamp_min(min_disp)
    except ImportError:
        pass
    disp = np.clip(disparity, min_disp, None)
    return (baseline_m * focal_px) / disp


# ---------------------------- visualization ---------------------------------
def save_disparity_visualization(
    disparity: np.ndarray,
    out_path: str | Path,
    max_disp: float | None = None,
    cmap: str = "magma",
) -> None:
    from PIL import Image

    arr = np.asarray(disparity, dtype=np.float32)
    if max_disp is None:
        max_disp = float(np.nanpercentile(arr[arr > 0], 99)) if np.any(arr > 0) else 1.0
    norm = np.clip(arr / max(max_disp, 1e-6), 0.0, 1.0)

    try:
        import matplotlib.cm as cm
        rgba = (cm.get_cmap(cmap)(norm) * 255).astype(np.uint8)
        Image.fromarray(rgba).save(out_path)
    except Exception:
        # No matplotlib — fall back to grayscale.
        Image.fromarray((norm * 255).astype(np.uint8)).save(out_path)

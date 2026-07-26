"""ONNX/TensorRT-friendly replacement for RAFT-Stereo's bilinear_sampler.

The correlation volume is (N, C, 1, W) and the sampled y is always 0, so
grid_sample degenerates to 1D linear interpolation along x. Implemented with
gather + lerp, which export at opset 12 and are natively supported by
TensorRT 8.2 (unlike GridSample, which needs TRT >= 8.5).

Matches F.grid_sample(..., mode='bilinear', align_corners=True) with the
default padding_mode='zeros': out-of-bounds corners contribute nothing.
"""
import torch


def bilinear_sampler_1d(img, coords, mode="bilinear", mask=False):
    # img:    (N, C, 1, W)
    # coords: (N, Hout, Wout, 2) in pixel units; coords[..., 1] is always 0
    N, C, H, W = img.shape
    assert H == 1, f"1D sampler expects height 1, got {H}"

    x = coords[..., 0]                                   # (N, Hout, Wout)
    Hout, Wout = x.shape[1], x.shape[2]

    x0 = torch.floor(x)
    x1 = x0 + 1.0
    w1 = x - x0                                          # lerp weight toward x1
    w0 = 1.0 - w1

    # zeros padding: a corner outside [0, W-1] contributes 0
    valid0 = ((x0 >= 0) & (x0 <= W - 1)).to(img.dtype)
    valid1 = ((x1 >= 0) & (x1 <= W - 1)).to(img.dtype)

    x0c = x0.clamp(0, W - 1).to(torch.int64)
    x1c = x1.clamp(0, W - 1).to(torch.int64)

    flat = img.reshape(N, C, W)                          # drop the height-1 axis
    n_out = Hout * Wout
    # gather wants indices broadcast across channels
    i0 = x0c.reshape(N, 1, n_out).expand(N, C, n_out)
    i1 = x1c.reshape(N, 1, n_out).expand(N, C, n_out)

    g0 = torch.gather(flat, 2, i0)
    g1 = torch.gather(flat, 2, i1)

    ww0 = (w0 * valid0).reshape(N, 1, n_out)
    ww1 = (w1 * valid1).reshape(N, 1, n_out)

    out = g0 * ww0 + g1 * ww1
    out = out.reshape(N, C, Hout, Wout)

    if mask:
        m = ((x >= 0) & (x <= W - 1)).to(img.dtype).reshape(N, 1, Hout, Wout)
        return out, m
    return out

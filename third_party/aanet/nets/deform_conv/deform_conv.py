"""
Deformable convolution — torchvision backend.

The original AANet code used a hand-rolled CUDA extension (deform_conv_cuda)
that no longer builds on CUDA 13 / Python 3.14.  torchvision ≥ 0.9 ships
torchvision.ops.deform_conv2d which is the same operation, already compiled
against the installed CUDA toolkit.  This file is a drop-in replacement: every
class and function that the rest of AANet imports is preserved with an identical
API, but the implementation delegates to torchvision.
"""
import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _single
import torchvision.ops as tvops


# ---------------------------------------------------------------------------
# Functional helpers (kept for backward-compat — nets/deform.py uses them)
# ---------------------------------------------------------------------------

def deform_conv(input, offset, weight, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1):
    """Standard (non-modulated) deformable conv via torchvision."""
    # torchvision.ops.deform_conv2d is the *modulated* variant; calling it
    # with mask=None is equivalent to mask=ones, which degenerates to the
    # non-modulated case.
    return tvops.deform_conv2d(
        input, offset, weight,
        bias=None,
        stride=_pair(stride),
        padding=_pair(padding),
        dilation=_pair(dilation),
        mask=None,
    )


def modulated_deform_conv(input, offset, mask, weight, bias=None,
                           stride=1, padding=0, dilation=1,
                           groups=1, deformable_groups=1):
    """Modulated (v2) deformable conv via torchvision."""
    return tvops.deform_conv2d(
        input, offset, weight,
        bias=bias,
        stride=_pair(stride),
        padding=_pair(padding),
        dilation=_pair(dilation),
        mask=mask,
    )


# ---------------------------------------------------------------------------
# nn.Module wrappers — identical API to the original
# ---------------------------------------------------------------------------

class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=False):
        super().__init__()
        assert not bias, "DeformConv does not support bias"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        return deform_conv(x, offset, self.weight,
                           self.stride, self.padding, self.dilation,
                           self.groups, self.deformable_groups)


class DeformConvPack(DeformConv):
    """Self-contained deformable conv that learns its own offsets."""
    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight,
                           self.stride, self.padding, self.dilation,
                           self.groups, self.deformable_groups)


class ModulatedDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv):
    """Self-contained modulated deformable conv that learns its own offsets."""
    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)

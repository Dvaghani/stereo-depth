"""
Uncertainty-aware RAFT-Stereo.

Extends RAFT-Stereo (realtime config) with a per-pixel Laplace uncertainty
head, ported from the thesis AANet design: a small conv head on the finest
GRU hidden state predicts log(b), the scale of a Laplace distribution over
the disparity error. Trained in a second phase with the backbone frozen,
using the NLL loss  |d - d*| / b + log(2b)  — identical to the AANet recipe.

Requires third_party/RAFT-Stereo on sys.path (see load_raft_uncertainty).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_RAFT_DIR = Path(__file__).resolve().parents[2] / "third_party" / "RAFT-Stereo"


def _import_raft():
    sys.path.insert(0, str(_RAFT_DIR))
    sys.path.insert(0, str(_RAFT_DIR / "core"))
    from raft_stereo import RAFTStereo, autocast          # noqa
    from core.corr import (CorrBlock1D, PytorchAlternateCorrBlock1D,  # noqa
                           CorrBlockFast1D, AlternateCorrBlock)
    return RAFTStereo, autocast, {
        "reg": CorrBlock1D, "alt": PytorchAlternateCorrBlock1D,
        "reg_cuda": CorrBlockFast1D, "alt_cuda": AlternateCorrBlock}


def realtime_args(mixed_precision: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        hidden_dims=[128, 128, 128], corr_implementation="reg",
        corr_levels=4, corr_radius=4, context_norm="batch",
        mixed_precision=mixed_precision,
        shared_backbone=True, n_downsample=3, n_gru_layers=2,
        slow_fast_gru=True)


def build_raft_uncertainty(args: argparse.Namespace | None = None):
    """Factory — returns a RAFTStereoUncertainty instance (class is created
    dynamically because the base class lives in third_party)."""
    RAFTStereo, autocast, corr_blocks = _import_raft()
    args = args or realtime_args()

    class RAFTStereoUncertainty(RAFTStereo):
        def __init__(self, a):
            super().__init__(a)
            hd = a.hidden_dims[0]        # finest GRU hidden state channels
            self.unc_head = nn.Sequential(
                nn.Conv2d(hd, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 3, padding=1),
            )

        def freeze_base(self):
            """Freeze everything except the uncertainty head (phase-2 training)."""
            for p in self.parameters():
                p.requires_grad = False
            for p in self.unc_head.parameters():
                p.requires_grad = True

        def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
            """Same as RAFTStereo.forward but additionally returns log_b
            (full resolution, clamped to [-5, 5])."""
            image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
            image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

            with autocast(enabled=self.args.mixed_precision):
                if self.args.shared_backbone:
                    *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0),
                                              dual_inp=True, num_layers=self.args.n_gru_layers)
                    fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
                else:
                    cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                    fmap1, fmap2 = self.fnet([image1, image2])
                net_list = [torch.tanh(x[0]) for x in cnet_list]
                inp_list = [torch.relu(x[1]) for x in cnet_list]
                inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1))
                            for i, conv in zip(inp_list, self.context_zqr_convs)]

            corr_block = corr_blocks[self.args.corr_implementation]
            if self.args.corr_implementation in ("reg", "alt"):
                fmap1, fmap2 = fmap1.float(), fmap2.float()
            corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius,
                                 num_levels=self.args.corr_levels)

            coords0, coords1 = self.initialize_flow(net_list[0])
            if flow_init is not None:
                coords1 = coords1 + flow_init

            flow_predictions = []
            flow_up = None
            for itr in range(iters):
                coords1 = coords1.detach()
                corr = corr_fn(coords1)
                flow = coords1 - coords0
                with autocast(enabled=self.args.mixed_precision):
                    if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:
                        net_list = self.update_block(net_list, inp_list, iter32=True,
                                                     iter16=False, iter08=False, update=False)
                    if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:
                        net_list = self.update_block(net_list, inp_list,
                                                     iter32=self.args.n_gru_layers == 3,
                                                     iter16=True, iter08=False, update=False)
                    net_list, up_mask, delta_flow = self.update_block(
                        net_list, inp_list, corr, flow,
                        iter32=self.args.n_gru_layers == 3,
                        iter16=self.args.n_gru_layers >= 2)

                delta_flow[:, 1] = 0.0
                coords1 = coords1 + delta_flow

                if test_mode and itr < iters - 1:
                    continue
                if up_mask is None:
                    flow_up = F.interpolate(8 * (coords1 - coords0),
                                            scale_factor=8, mode="bilinear")
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                flow_up = flow_up[:, :1]
                flow_predictions.append(flow_up)

            # ── uncertainty from the final hidden state ──────────────────────
            log_b_low = self.unc_head(net_list[0].float())
            factor = 2 ** self.args.n_downsample
            log_b = F.interpolate(log_b_low, scale_factor=factor,
                                  mode="bilinear", align_corners=False)
            log_b = log_b.clamp(-5.0, 5.0)

            if test_mode:
                return coords1 - coords0, flow_up, log_b
            return flow_predictions, log_b

    return RAFTStereoUncertainty(args)


def load_raft_uncertainty(ckpt: str | Path, device: str = "cuda",
                          mixed_precision: bool = False):
    """Load a checkpoint into the uncertainty model. Works both with plain
    RAFT checkpoints (head randomly initialized) and with checkpoints that
    already contain the trained head."""
    model = build_raft_uncertainty(realtime_args(mixed_precision))
    sd = torch.load(ckpt, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("unc_head")]
    if missing or unexpected:
        print(f"load_raft_uncertainty: missing={missing} unexpected={unexpected}")
    return model.to(device).eval()

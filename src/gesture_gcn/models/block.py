"""Unified Spatio-Temporal Gesture Block combining DA-GC, MS-TCN, and DRE."""

from __future__ import annotations

import torch
from torch import nn

from gesture_gcn.models.dagc import DualStreamDAGC
from gesture_gcn.models.dre import DynamicRangeEmbedding
from gesture_gcn.models.mstcn import MultiScaleTCN, StandardTCN, _group_norm


class UnifiedGestureBlock(nn.Module):
    """Core building block of Plan A:
    DualStreamDAGC -> BN -> GELU -> (Optional DRE Gating) -> MS-TCN -> Residual -> GELU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int,
        natural_adj: torch.Tensor,
        num_groups: int = 4,
        use_aru: bool = True,
        use_dru: bool = True,
        use_dre: bool = True,
        use_mstcn: bool = True,
        mstcn_dilations: list[int] | tuple[int, ...] = (1, 2, 3, 4),
    ) -> None:
        super().__init__()
        self.spatial_gcn = DualStreamDAGC(
            in_channels=in_channels,
            out_channels=out_channels,
            num_joints=num_joints,
            natural_adj=natural_adj,
            num_groups=num_groups,
            use_aru=use_aru,
            use_dru=use_dru,
        )
        self.spatial_bn = _group_norm(out_channels)
        self.act = nn.GELU()

        self.dre = DynamicRangeEmbedding(out_channels) if use_dre else None

        self.temporal_conv: nn.Module = (
            MultiScaleTCN(out_channels, dilations=mstcn_dilations)
            if use_mstcn
            else StandardTCN(out_channels)
        )

        if in_channels != out_channels:
            self.residual: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                _group_norm(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, dynamic_range: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        h = self.act(self.spatial_bn(self.spatial_gcn(x)))
        if self.dre is not None:
            h = self.dre(h, dynamic_range)
        h = self.temporal_conv(h)
        return self.act(h + res)

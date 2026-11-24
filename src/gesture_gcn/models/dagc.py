"""Dual-Stream Difference-Attention Graph Convolution (DA-GC + Grouped Static Topology)."""

from __future__ import annotations

import torch
from torch import nn

from gesture_gcn.models.topology import (
    AttentionRelationUnit,
    DifferenceRelationUnit,
    GroupedTopologyAggregation,
)


class DualStreamDAGC(nn.Module):
    """Parallel Channel-wise and Temporal-wise dynamic non-shared graph convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int,
        natural_adj: torch.Tensor,
        num_groups: int = 4,
        use_aru: bool = True,
        use_dru: bool = True,
    ) -> None:
        super().__init__()
        self.use_aru = use_aru
        self.use_dru = use_dru

        # Shared high-level feature projection
        self.feature_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # Channel stream topology units
        self.ch_aru = AttentionRelationUnit(out_channels, mode="channel") if use_aru else None
        self.ch_dru = DifferenceRelationUnit(out_channels, mode="channel") if use_dru else None
        self.ch_tam = GroupedTopologyAggregation(
            num_joints=num_joints,
            natural_adj=natural_adj,
            num_groups=num_groups,
            mode="channel",
            use_aru=use_aru,
            use_dru=use_dru,
        )

        # Temporal stream topology units
        self.tm_aru = AttentionRelationUnit(out_channels, mode="temporal") if use_aru else None
        self.tm_dru = DifferenceRelationUnit(out_channels, mode="temporal") if use_dru else None
        self.tm_tam = GroupedTopologyAggregation(
            num_joints=num_joints,
            natural_adj=natural_adj,
            num_groups=num_groups,
            mode="temporal",
            use_aru=use_aru,
            use_dru=use_dru,
        )

        # Channel-Temporal Fusion (CTF) learnable weights
        self.eps = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C_in, T, N)

        Returns:
            (B, C_out, T, N)
        """
        feat = self.feature_proj(x)  # (B, C_out, T, N)

        # 1. Channel-wise non-shared topology S: (B, C_out, N, N)
        ch_diff = self.ch_dru(feat) if self.ch_dru is not None else None
        ch_attn = self.ch_aru(feat) if self.ch_aru is not None else None
        s_adj = self.ch_tam(ch_diff, ch_attn)
        z_s = torch.einsum("bctn,bcnm->bctm", feat, s_adj)

        # 2. Temporal-wise non-shared topology H: (B, T, N, N)
        tm_diff = self.tm_dru(feat) if self.tm_dru is not None else None
        tm_attn = self.tm_aru(feat) if self.tm_aru is not None else None
        h_adj = self.tm_tam(tm_diff, tm_attn)
        z_h = torch.einsum("bctn,btnm->bctm", feat, h_adj)

        # 3. Channel-Temporal Fusion (CTF)
        return self.eps * z_s + self.gamma * z_h

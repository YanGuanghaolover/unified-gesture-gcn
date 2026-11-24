"""Temporal Convolution modules: Multi-Scale TCN (DSTSA-GCN) and Standard TCN (DA-GCN)."""

from __future__ import annotations

import torch
from torch import nn


def _group_norm(channels: int) -> nn.GroupNorm:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class MultiScaleTCN(nn.Module):
    """Multi-Scale Temporal Convolution (MS-TCN) from DSTSA-GCN.

    Combines dilated temporal convolution branches (capturing varying dynamic speeds),
    a temporal MaxPool branch, and a 1x1 shortcut branch (preserving static frame features).
    """

    def __init__(
        self,
        channels: int,
        dilations: list[int] | tuple[int, ...] = (1, 2, 3, 4),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        num_branches = len(dilations) + 2  # dilated branches + maxpool + shortcut
        base_ch = channels // num_branches
        remainder = channels - base_ch * num_branches
        branch_channels = [base_ch + (1 if i < remainder else 0) for i in range(num_branches)]

        self.branches = nn.ModuleList()

        # 1. Multi-scale dilated temporal convolution branches
        for idx, dilation in enumerate(dilations):
            b_ch = branch_channels[idx]
            pad = (kernel_size - 1) * dilation // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, b_ch, kernel_size=1, bias=False),
                    _group_norm(b_ch),
                    nn.GELU(),
                    nn.Conv2d(
                        b_ch,
                        b_ch,
                        kernel_size=(kernel_size, 1),
                        padding=(pad, 0),
                        dilation=(dilation, 1),
                        bias=False,
                    ),
                )
            )

        # 2. Temporal MaxPool branch
        pool_ch = branch_channels[-2]
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(channels, pool_ch, kernel_size=1, bias=False),
                _group_norm(pool_ch),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            )
        )

        # 3. Shortcut branch (crucial for static hand gestures)
        short_ch = branch_channels[-1]
        self.branches.append(nn.Conv2d(channels, short_ch, kernel_size=1, bias=False))

        self.bn = _group_norm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.bn(out)


class StandardTCN(nn.Module):
    """Standard 9x1 Temporal Convolution from DA-GCN (kept for ablation comparison)."""

    def __init__(self, channels: int, kernel_size: int = 9) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0), bias=False),
            _group_norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

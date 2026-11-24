"""Unified Static & Dynamic Hand Gesture Recognition Network (Plan A)."""

from __future__ import annotations

import torch
from torch import nn

from gesture_gcn.config import ModelConfig
from gesture_gcn.data.skeleton import HandSkeletonSpec
from gesture_gcn.models.block import UnifiedGestureBlock


class UnifiedGestureGCN(nn.Module):
    """Lightweight 4-layer unified skeleton gesture recognition network.

    Architecture (Plan A):
      - Input: Joint + Bone early fusion (6, T, N) + Dynamic Range (3, N)
      - Backbone: 4 x UnifiedGestureBlock (DA-GC + Grouped Static Topology + DRE + MS-TCN)
      - Head: Global Average Pooling + Dropout + Linear Classifier
    """

    def __init__(
        self,
        cfg: ModelConfig,
        spec: HandSkeletonSpec,
        in_channels: int = 6,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.spec = spec
        natural_adj = spec.build_adjacency(self_loops=True, normalize=True)

        self.input_bn = nn.GroupNorm(1, in_channels)

        blocks: list[UnifiedGestureBlock] = []
        prev_ch = in_channels
        for out_ch in cfg.channels:
            blocks.append(
                UnifiedGestureBlock(
                    in_channels=prev_ch,
                    out_channels=out_ch,
                    num_joints=spec.num_joints,
                    natural_adj=natural_adj,
                    num_groups=cfg.num_groups,
                    use_aru=cfg.use_aru,
                    use_dru=cfg.use_dru,
                    use_dre=cfg.use_dre,
                    use_mstcn=cfg.use_mstcn,
                    mstcn_dilations=cfg.mstcn_dilations,
                )
            )
            prev_ch = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.classifier = nn.Linear(prev_ch, cfg.num_classes)

    def forward(self, x: torch.Tensor, dynamic_range: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Preprocessed skeleton features of shape (B, C_in, T, N).
            dynamic_range: Per-joint temporal range of shape (B, 3, N).

        Returns:
            Class logits of shape (B, num_classes).
        """
        h = self.input_bn(x)
        for block in self.blocks:
            h = block(h, dynamic_range)
        # Global Average Pooling over (T, N)
        pooled = h.mean(dim=(2, 3))
        return self.classifier(self.dropout(pooled))

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

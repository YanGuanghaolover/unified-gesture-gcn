"""Dynamic Range Embedding (DRE) module inspired by SKETCH (Doodle to Detect).

Preserves absolute per-joint motion magnitudes lost during window normalization,
enabling explicit disambiguation between static poses and dynamic gestures.
"""

from __future__ import annotations

import torch
from torch import nn


class DynamicRangeEmbedding(nn.Module):
    """Projects per-joint temporal dynamic range (B, 3, N) into a motion-aware feature gate."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(16, channels // reduction)
        self.net = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, channels * 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, dynamic_range: torch.Tensor) -> torch.Tensor:
        """Modulate spatio-temporal features using per-joint dynamic range.

        Args:
            x: Feature tensor of shape (B, C, T, N).
            dynamic_range: Per-joint temporal range of shape (B, 3, N).

        Returns:
            Modulated feature tensor of shape (B, C, T, N).
        """
        # Use log1p to compress large dynamic ranges smoothly while keeping 0 -> 0 for static poses
        dr_encoded = self.net(torch.log1p(dynamic_range))  # (B, 2C, N)
        scale, shift = dr_encoded.chunk(2, dim=1)          # each (B, C, N)
        gate = torch.sigmoid(scale).unsqueeze(2)           # (B, C, 1, N)
        bias = shift.unsqueeze(2)                          # (B, C, 1, N)
        return x * (0.5 + gate) + 0.1 * bias

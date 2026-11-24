"""Dynamic topology modeling units: ARU, DRU (DA-GCN) + Grouped Static Topology (DSTSA-GCN)."""

from __future__ import annotations

import math

import torch
from torch import nn


class AttentionRelationUnit(nn.Module):
    """Attention Relation Unit (ARU) from DA-GCN.

    Computes global pairwise node attention along either the channel or temporal axis.
    """

    def __init__(self, in_channels: int, mode: str) -> None:
        super().__init__()
        if mode not in ("channel", "temporal"):
            raise ValueError(f"Unsupported ARU mode: {mode}")
        self.mode = mode
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention-based topology.

        Args:
            x: Input tensor of shape (B, C, T, N).

        Returns:
            If mode == "channel":  (B, C, N, N)
            If mode == "temporal": (B, T, N, N)
        """
        _, channels, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)

        if self.mode == "channel":
            scores = torch.einsum("bctn,bctm->bcnm", q, k) / math.sqrt(max(seq_len, 1))
        else:
            scores = torch.einsum("bctn,bctm->btnm", q, k) / math.sqrt(max(channels, 1))
        return torch.softmax(scores, dim=-1)


class DifferenceRelationUnit(nn.Module):
    """Difference Relation Unit (DRU) from DA-GCN.

    Compresses the orthogonal axis via mean pooling, computes pairwise node differences,
    and applies GELU non-linearity.
    """

    def __init__(self, in_channels: int, mode: str) -> None:
        super().__init__()
        if mode not in ("channel", "temporal"):
            raise ValueError(f"Unsupported DRU mode: {mode}")
        self.mode = mode
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute difference-based topology.

        Args:
            x: Input tensor of shape (B, C, T, N).

        Returns:
            If mode == "channel":  (B, C, N, N)
            If mode == "temporal": (B, T, N, N)
        """
        feat = self.proj(x)
        if self.mode == "channel":
            pooled = feat.mean(dim=2)  # (B, C, N)
        else:
            pooled = feat.mean(dim=1)  # (B, T, N)
        diff = pooled.unsqueeze(-1) - pooled.unsqueeze(-2)  # (B, D, N, N)
        return self.act(diff)


class GroupedTopologyAggregation(nn.Module):
    """Topology Aggregation Module (TAM) enhanced with DSTSA-GCN's grouped static topologies.

    Fuses:
      - Difference topology D (optional)
      - Attention topology A_att (optional)
      - Grouped learnable static topology A_static of shape (K, N, N) seeded with natural
        adjacency + random multi-scale groups to eliminate local bias.
    """

    def __init__(
        self,
        num_joints: int,
        natural_adj: torch.Tensor,
        num_groups: int = 4,
        mode: str = "channel",
        use_aru: bool = True,
        use_dru: bool = True,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.num_groups = num_groups
        self.mode = mode
        self.use_aru = use_aru
        self.use_dru = use_dru

        # Learnable fusion scalars
        self.alpha_d = nn.Parameter(torch.tensor(0.5))
        self.alpha_a = nn.Parameter(torch.tensor(0.5))

        # Grouped static topology (K, N, N): group 0 initialized from natural skeleton,
        # remaining groups randomly initialized (DSTSA-GCN) to capture distant fingertip relations.
        static_init = torch.randn(num_groups, num_joints, num_joints) * 0.05
        static_init[0] = natural_adj.clone()
        self.static_adj = nn.Parameter(static_init)
        self.static_mask = nn.Parameter(torch.ones(num_groups, num_joints, num_joints))

        self.act = nn.GELU()

    def forward(
        self,
        diff_topo: torch.Tensor | None,
        attn_topo: torch.Tensor | None,
    ) -> torch.Tensor:
        """Aggregate dynamic and grouped static topologies.

        Args:
            diff_topo: (B, D, N, N) or None
            attn_topo: (B, D, N, N) or None

        Returns:
            Aggregated non-shared adjacency matrix of shape (B, D, N, N).
        """
        ref = diff_topo if diff_topo is not None else attn_topo
        if ref is None:
            raise ValueError("At least one of diff_topo or attn_topo must be provided.")

        batch_size, dim, n_joints, _ = ref.shape
        fused = torch.zeros_like(ref)

        if self.use_dru and diff_topo is not None:
            fused = fused + self.alpha_d * diff_topo
        if self.use_aru and attn_topo is not None:
            fused = fused + self.alpha_a * attn_topo

        # Broadcast grouped static topology (K, N, N) -> (1, D, N, N)
        masked_static = self.static_adj * self.static_mask  # (K, N, N)
        if self.mode == "channel" and dim % self.num_groups == 0:
            repeats = dim // self.num_groups
            static_expanded = masked_static.repeat_interleave(repeats, dim=0).unsqueeze(0)
        else:
            # For temporal mode (arbitrary sequence length T), aggregate grouped prior smoothly
            static_expanded = masked_static.mean(dim=0, keepdim=True).unsqueeze(0).expand(
                batch_size, dim, n_joints, n_joints
            )

        return self.act(fused + static_expanded)

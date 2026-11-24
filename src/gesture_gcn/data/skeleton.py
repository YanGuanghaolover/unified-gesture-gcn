"""Hand skeleton topology specifications and adjacency matrix builders."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HandSkeletonSpec:
    """Immutable specification of a hand skeleton graph."""

    name: str
    num_joints: int
    root_joint: int
    parents: tuple[int, ...]  # parent[i] == i for the root joint

    def build_adjacency(self, self_loops: bool = True, normalize: bool = True) -> torch.Tensor:
        """Build undirected natural adjacency matrix A of shape (N, N)."""
        n = self.num_joints
        adj = torch.zeros((n, n), dtype=torch.float32)
        for child, parent in enumerate(self.parents):
            if child != parent:
                adj[child, parent] = 1.0
                adj[parent, child] = 1.0
        if self_loops:
            adj.fill_diagonal_(1.0)
        if normalize:
            deg = adj.sum(dim=-1).clamp(min=1e-6)
            deg_inv_sqrt = deg.pow(-0.5)
            adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        return adj


# MediaPipe Hands / FPHA 21-joint standard topology:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)
_MEDIAPIPE_21_PARENTS: tuple[int, ...] = (
    0,  # 0: wrist (root)
    0, 1, 2, 3,      # thumb
    0, 5, 6, 7,      # index
    0, 9, 10, 11,    # middle
    0, 13, 14, 15,   # ring
    0, 17, 18, 19,   # pinky
)

# SHREC'17 / DHG-14/28 22-joint standard topology (0-indexed):
# 0: Wrist, 1: Palm
# 2-5: Thumb, 6-9: Index, 10-13: Middle, 14-17: Ring, 18-21: Pinky
_SHREC_22_PARENTS: tuple[int, ...] = (
    0,  # 0: wrist (root)
    0,  # 1: palm
    0, 2, 3, 4,      # thumb
    1, 6, 7, 8,      # index
    1, 10, 11, 12,   # middle
    1, 14, 15, 16,   # ring
    1, 18, 19, 20,   # pinky
)

_SKELETON_REGISTRY: dict[str, HandSkeletonSpec] = {
    "mediapipe_21": HandSkeletonSpec(
        name="mediapipe_21",
        num_joints=21,
        root_joint=0,
        parents=_MEDIAPIPE_21_PARENTS,
    ),
    "shrec_22": HandSkeletonSpec(
        name="shrec_22",
        num_joints=22,
        root_joint=0,
        parents=_SHREC_22_PARENTS,
    ),
}


def register_skeleton_spec(spec: HandSkeletonSpec) -> None:
    """Register a custom skeleton topology for extensibility."""
    if len(spec.parents) != spec.num_joints:
        raise ValueError("Length of parents must equal num_joints.")
    _SKELETON_REGISTRY[spec.name] = spec


def get_skeleton_spec(name: str) -> HandSkeletonSpec:
    """Retrieve a registered skeleton specification by name."""
    if name not in _SKELETON_REGISTRY:
        available = ", ".join(sorted(_SKELETON_REGISTRY))
        raise KeyError(f"Unknown skeleton_type '{name}'. Available: {available}")
    return _SKELETON_REGISTRY[name]

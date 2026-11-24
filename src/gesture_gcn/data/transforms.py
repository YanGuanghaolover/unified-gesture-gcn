"""Skeleton preprocessing: safe window normalization, DRE extraction, and Joint+Bone fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gesture_gcn.data.skeleton import HandSkeletonSpec


@dataclass
class GestureSample:
    """Structured model input returned by datasets and preprocessors."""

    features: torch.Tensor       # (C_in, T, N), where C_in = 6 if bone fusion else 3
    dynamic_range: torch.Tensor  # (3, N), unnormalized temporal motion range per joint
    label: int


class SkeletonPreprocessor:
    """Converts raw 3D skeleton sequences (3, T, N) into model-ready tensors.

    Combines:
    1. Dynamic Range Extraction (inspired by SKETCH): records true temporal motion range
       per joint along (x, y, z) BEFORE normalization, cleanly separating static vs. dynamic.
    2. Safe Per-Window Isotropic Normalization: centers around the window's root trajectory
       mean and divides by max(window_spatial_span, scale_floor) so static poses never
       amplify sensor jitter.
    3. Joint + Bone Early Fusion: concatenates normalized 3D coordinates with parent-child
       bone vectors along the channel dimension for strong static pose geometry.
    """

    def __init__(
        self,
        spec: HandSkeletonSpec,
        use_bone_fusion: bool = True,
        scale_floor: float = 0.05,
    ) -> None:
        self.spec = spec
        self.use_bone_fusion = use_bone_fusion
        self.scale_floor = scale_floor
        self._parent_idx = torch.tensor(spec.parents, dtype=torch.long)

    @property
    def out_channels(self) -> int:
        return 6 if self.use_bone_fusion else 3

    def __call__(self, raw_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a single raw sequence of shape (3, T, N).

        Returns:
            features: (C_in, T, N)
            dynamic_range: (3, N)
        """
        if raw_seq.ndim != 3 or raw_seq.shape[0] != 3 or raw_seq.shape[2] != self.spec.num_joints:
            raise ValueError(
                f"Expected raw_seq shape (3, T, {self.spec.num_joints}), got {tuple(raw_seq.shape)}"
            )

        # 1. Temporal dynamic range per joint (3, N) before normalization
        t_max = raw_seq.max(dim=1).values
        t_min = raw_seq.min(dim=1).values
        dynamic_range = (t_max - t_min).clamp(min=0.0)

        # 2. Safe per-window isotropic normalization
        root_center = raw_seq[:, :, self.spec.root_joint : self.spec.root_joint + 1].mean(
            dim=1, keepdim=True
        )
        centered = raw_seq - root_center

        # Use overall spatial span across the window with a floor to avoid static jitter blow-up
        spatial_span = (centered.amax(dim=(1, 2)) - centered.amin(dim=(1, 2))).max()
        scale = torch.clamp(spatial_span, min=self.scale_floor)
        norm_joints = centered / scale

        if not self.use_bone_fusion:
            return norm_joints, dynamic_range

        # 3. Bone vectors: child - parent (3, T, N)
        parent_coords = norm_joints.index_select(dim=2, index=self._parent_idx.to(raw_seq.device))
        bones = norm_joints - parent_coords
        features = torch.cat([norm_joints, bones], dim=0)
        return features, dynamic_range

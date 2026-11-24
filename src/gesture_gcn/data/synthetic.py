"""Synthetic static + dynamic hand gesture dataset for minimal runnable verification."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset

from gesture_gcn.config import DataConfig
from gesture_gcn.data.skeleton import HandSkeletonSpec, get_skeleton_spec
from gesture_gcn.data.transforms import SkeletonPreprocessor

GESTURE_LABELS: tuple[str, ...] = (
    "static_open_palm",
    "static_fist",
    "static_ok_pinch",
    "dynamic_swipe_right",
    "dynamic_grab_close",
    "dynamic_circle_wave",
)


def _base_open_hand(num_joints: int) -> torch.Tensor:
    """Create a canonical 3D open-hand pose of shape (3, N)."""
    pose = torch.zeros((3, num_joints), dtype=torch.float32)
    if num_joints == 21:
        finger_starts = [1, 5, 9, 13, 17]
        x_offsets = [-0.28, -0.14, 0.0, 0.14, 0.26]
    else:
        # 22 joints: index 1 is palm
        pose[:, 1] = torch.tensor([0.0, 0.18, 0.0])
        finger_starts = [2, 6, 10, 14, 18]
        x_offsets = [-0.28, -0.14, 0.0, 0.14, 0.26]

    for x_off, start in zip(x_offsets, finger_starts):
        for seg in range(4):
            idx = start + seg
            y = 0.20 + 0.14 * (seg + 1)
            pose[0, idx] = x_off * (0.6 + 0.15 * seg)
            pose[1, idx] = y
            pose[2, idx] = 0.0
    return pose


def _curl_fingers(pose: torch.Tensor, finger_indices: list[int], curl: float) -> torch.Tensor:
    """Curl specified finger chains (0..4) by curl factor in [0, 1]."""
    out = pose.clone()
    num_joints = pose.shape[1]
    starts = [1, 5, 9, 13, 17] if num_joints == 21 else [2, 6, 10, 14, 18]
    for f_id in finger_indices:
        start = starts[f_id]
        base = out[:, start].clone()
        for seg in range(1, 4):
            idx = start + seg
            rel = out[:, idx] - base
            # Curl brings fingertips inward toward palm (lower y, positive z)
            out[1, idx] = base[1] + rel[1] * (1.0 - 0.85 * curl)
            out[2, idx] = base[2] + 0.22 * curl * (seg / 3.0)
    return out


class SyntheticGestureDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Generates 3 static + 3 dynamic hand gesture classes with realistic noise & jitter."""

    def __init__(
        self,
        spec: HandSkeletonSpec,
        preprocessor: SkeletonPreprocessor,
        seq_len: int = 48,
        num_samples_per_class: int = 40,
        seed: int = 42,
    ) -> None:
        self.spec = spec
        self.preprocessor = preprocessor
        self.seq_len = seq_len
        self.samples: list[tuple[torch.Tensor, int]] = []

        gen = torch.Generator().manual_seed(seed)
        base = _base_open_hand(spec.num_joints)
        t_grid = torch.linspace(0.0, 1.0, seq_len)

        for cls_idx in range(len(GESTURE_LABELS)):
            for _ in range(num_samples_per_class):
                seq = self._generate_sequence(base, cls_idx, t_grid, gen)
                self.samples.append((seq, cls_idx))

    def _generate_sequence(
        self,
        base: torch.Tensor,
        cls_idx: int,
        t_grid: torch.Tensor,
        gen: torch.Generator,
    ) -> torch.Tensor:
        t_len = t_grid.shape[0]
        # Random hand scale and 3D translation to test view/scale invariance
        hand_scale = 0.8 + 0.4 * torch.rand((), generator=gen).item()
        offset = torch.randn((3, 1, 1), generator=gen) * 0.3

        if cls_idx == 0:
            # Static: open palm
            seq = base.unsqueeze(1).repeat(1, t_len, 1)
        elif cls_idx == 1:
            # Static: closed fist (all 5 fingers curled)
            fist = _curl_fingers(base, [0, 1, 2, 3, 4], curl=0.95)
            seq = fist.unsqueeze(1).repeat(1, t_len, 1)
        elif cls_idx == 2:
            # Static: OK pinch (thumb + index curled toward each other, others open)
            ok = _curl_fingers(base, [0, 1], curl=0.75)
            # Bring thumb tip and index tip close
            thumb_tip = 4 if self.spec.num_joints == 21 else 5
            index_tip = 8 if self.spec.num_joints == 21 else 9
            mid = 0.5 * (ok[:, thumb_tip] + ok[:, index_tip])
            ok[:, thumb_tip] = mid + torch.tensor([-0.01, 0.0, 0.0])
            ok[:, index_tip] = mid + torch.tensor([0.01, 0.0, 0.0])
            seq = ok.unsqueeze(1).repeat(1, t_len, 1)
        elif cls_idx == 3:
            # Dynamic: swipe right
            seq = base.unsqueeze(1).repeat(1, t_len, 1)
            shift_x = (0.9 * t_grid).view(1, t_len, 1)
            seq[0:1] = seq[0:1] + shift_x
        elif cls_idx == 4:
            # Dynamic: grab (open palm gradually curls into fist)
            frames = [_curl_fingers(base, [0, 1, 2, 3, 4], curl=float(c)) for c in t_grid]
            seq = torch.stack(frames, dim=1)
        else:
            # Dynamic: circular wave in XY plane
            seq = base.unsqueeze(1).repeat(1, t_len, 1)
            angle = 2.0 * math.pi * t_grid
            seq[0:1] = seq[0:1] + (0.35 * torch.cos(angle)).view(1, t_len, 1)
            seq[1:2] = seq[1:2] + (0.35 * torch.sin(angle)).view(1, t_len, 1)

        # Apply hand scale, global offset, and realistic sensor jitter
        jitter = torch.randn(seq.shape, generator=gen) * 0.012
        return seq * hand_scale + offset + jitter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        raw_seq, label = self.samples[index]
        features, dynamic_range = self.preprocessor(raw_seq)
        return features, dynamic_range, label


def build_synthetic_dataloaders(
    cfg: DataConfig,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, HandSkeletonSpec, int]:
    """Build train and validation DataLoaders for the synthetic static+dynamic benchmark."""
    spec = get_skeleton_spec(cfg.skeleton_type)
    preprocessor = SkeletonPreprocessor(
        spec=spec,
        use_bone_fusion=cfg.use_bone_fusion,
        scale_floor=cfg.norm_scale_floor,
    )
    val_per_cls = max(1, int(round(cfg.num_samples_per_class * cfg.val_ratio)))
    train_per_cls = max(1, cfg.num_samples_per_class - val_per_cls)

    train_ds = SyntheticGestureDataset(
        spec=spec,
        preprocessor=preprocessor,
        seq_len=cfg.seq_len,
        num_samples_per_class=train_per_cls,
        seed=seed,
    )
    val_ds = SyntheticGestureDataset(
        spec=spec,
        preprocessor=preprocessor,
        seq_len=cfg.seq_len,
        num_samples_per_class=val_per_cls,
        seed=seed + 10_000,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, spec, preprocessor.out_channels

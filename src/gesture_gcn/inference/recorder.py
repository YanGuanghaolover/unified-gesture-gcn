"""Interactive real-hand gesture clip store & fast on-device UnifiedGestureGCN trainer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from gesture_gcn.config import DataConfig, ModelConfig, TrainConfig
from gesture_gcn.data.skeleton import HandSkeletonSpec
from gesture_gcn.data.transforms import SkeletonPreprocessor
from gesture_gcn.engine.trainer import Trainer
from gesture_gcn.models.network import UnifiedGestureGCN


def _augment_raw_sequence(seq: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Apply lightweight 3D rotation, scale, and jitter augmentation to a (3, T, N) clip."""
    # Random Z-axis (in-plane) rotation within [-15, 15] degrees
    angle = (torch.rand((), generator=gen).item() * 2.0 - 1.0) * math.radians(15.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    # seq: (3, T, N) -> apply rot along dim 0
    rotated = torch.einsum("ij,jtn->itn", rot, seq)
    scale = 0.85 + 0.30 * torch.rand((), generator=gen).item()
    jitter = torch.randn(seq.shape, generator=gen) * 0.005
    return rotated * scale + jitter


class _RecordedClipDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(
        self,
        clips: list[tuple[torch.Tensor, int]],
        preprocessor: SkeletonPreprocessor,
        augment_factor: int = 8,
        seed: int = 42,
    ) -> None:
        self.preprocessor = preprocessor
        self.items: list[tuple[torch.Tensor, int]] = []
        gen = torch.Generator().manual_seed(seed)
        for seq, label in clips:
            self.items.append((seq, label))
            for _ in range(augment_factor):
                self.items.append((_augment_raw_sequence(seq, gen), label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        seq, label = self.items[idx]
        feat, dr = self.preprocessor(seq)
        return feat, dr, label


class CustomGestureStore:
    """Persists real user-recorded 3D skeleton clips and trains UnifiedGestureGCN on them."""

    def __init__(self, store_path: str | Path, seq_len: int = 32, num_joints: int = 21) -> None:
        self.store_path = Path(store_path)
        self.seq_len = seq_len
        self.num_joints = num_joints
        self._clips_by_label: dict[str, list[list[list[list[float]]]]] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            self._clips_by_label = data.get("clips", {})

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seq_len": self.seq_len,
            "num_joints": self.num_joints,
            "clips": self._clips_by_label,
        }
        self.store_path.write_text(json.dumps(payload), encoding="utf-8")

    def add_clip(self, label: str, frames: list[list[list[float]]]) -> int:
        """Add a recorded clip of shape (T, N, 3) and resample linearly to self.seq_len."""
        label = label.strip()
        if not label:
            raise ValueError("Label cannot be empty.")
        tensor = torch.as_tensor(frames, dtype=torch.float32)
        if tensor.ndim != 3 or tensor.shape[1:] != (self.num_joints, 3):
            raise ValueError(
                f"Expected frames shape (T, {self.num_joints}, 3), got {tuple(tensor.shape)}"
            )
        # Convert (T, N, 3) -> (3, T, N) and resample to fixed seq_len
        seq = tensor.permute(2, 0, 1).contiguous()
        if seq.shape[1] != self.seq_len:
            seq = torch.nn.functional.interpolate(
                seq.unsqueeze(0),
                size=(self.seq_len, self.num_joints),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        self._clips_by_label.setdefault(label, []).append(seq.tolist())
        self._save()
        return len(self._clips_by_label[label])

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._clips_by_label.items()}

    def clear(self) -> None:
        self._clips_by_label.clear()
        if self.store_path.exists():
            self.store_path.unlink()

    def train_model(
        self,
        base_model_cfg: ModelConfig,
        data_cfg: DataConfig,
        spec: HandSkeletonSpec,
        device: torch.device,
        epochs: int = 18,
    ) -> tuple[UnifiedGestureGCN, list[str], float]:
        """Train a fresh UnifiedGestureGCN on all recorded real-hand clips."""
        labels = sorted(k for k, v in self._clips_by_label.items() if len(v) > 0)
        if len(labels) < 2:
            raise ValueError("请至少录制 2 个不同手势类别的样本后再训练。")

        clips: list[tuple[torch.Tensor, int]] = []
        for idx, label in enumerate(labels):
            for raw_list in self._clips_by_label[label]:
                clips.append((torch.tensor(raw_list, dtype=torch.float32), idx))

        preprocessor = SkeletonPreprocessor(
            spec=spec,
            use_bone_fusion=data_cfg.use_bone_fusion,
            scale_floor=data_cfg.norm_scale_floor,
        )
        dataset = _RecordedClipDataset(clips, preprocessor, augment_factor=10)
        loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)

        cfg_dict = base_model_cfg.model_dump()
        cfg_dict["num_classes"] = len(labels)
        model_cfg = ModelConfig.model_validate(cfg_dict)
        model = UnifiedGestureGCN(cfg=model_cfg, spec=spec, in_channels=preprocessor.out_channels)

        train_cfg = TrainConfig(epochs=epochs, lr=4e-3, weight_decay=5e-4, log_interval=6)
        trainer = Trainer(model=model, cfg=train_cfg, device=device)
        history = trainer.fit(loader, loader)
        final_acc = history[-1].train_acc
        return model, labels, final_acc

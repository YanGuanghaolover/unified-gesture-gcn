"""Strongly-typed configuration schema for the Unified Gesture GCN project."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    skeleton_type: Literal["mediapipe_21", "shrec_22"] = "mediapipe_21"
    seq_len: int = Field(default=48, ge=8)
    use_bone_fusion: bool = True
    norm_scale_floor: float = Field(default=0.05, gt=0.0)
    num_samples_per_class: int = Field(default=40, ge=4)
    val_ratio: float = Field(default=0.25, gt=0.0, lt=1.0)
    batch_size: int = Field(default=16, ge=1)


class ModelConfig(BaseModel):
    num_classes: int = Field(default=6, ge=2)
    channels: list[int] = Field(default_factory=lambda: [64, 64, 64, 128], min_length=1)
    num_groups: int = Field(default=4, ge=1)
    use_aru: bool = True
    use_dru: bool = True
    use_dre: bool = True
    use_mstcn: bool = True
    mstcn_dilations: list[int] = Field(default_factory=lambda: [1, 2, 3, 4], min_length=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_divisibility(self) -> ModelConfig:
        for ch in self.channels:
            if ch % self.num_groups != 0:
                raise ValueError(
                    f"Channel width {ch} must be divisible by num_groups ({self.num_groups})."
                )
        if not self.use_aru and not self.use_dru:
            raise ValueError("At least one of use_aru or use_dru should be enabled.")
        return self


class TrainConfig(BaseModel):
    epochs: int = Field(default=12, ge=1)
    lr: float = Field(default=3e-3, gt=0.0)
    weight_decay: float = Field(default=5e-4, ge=0.0)
    log_interval: int = Field(default=2, ge=1)


class ExperimentConfig(BaseModel):
    experiment_name: str = "unified_gesture_demo"
    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

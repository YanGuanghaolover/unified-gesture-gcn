"""End-to-end smoke test: training converges on synthetic static+dynamic gestures."""

from __future__ import annotations

import torch

from gesture_gcn.config import DataConfig, ModelConfig, TrainConfig
from gesture_gcn.data.synthetic import build_synthetic_dataloaders
from gesture_gcn.engine.trainer import Trainer
from gesture_gcn.models.network import UnifiedGestureGCN


def test_end_to_end_training_convergence() -> None:
    torch.manual_seed(0)
    data_cfg = DataConfig(seq_len=24, num_samples_per_class=12, val_ratio=0.25, batch_size=12)
    model_cfg = ModelConfig(num_classes=6, channels=[32, 32, 32, 64], num_groups=4)
    train_cfg = TrainConfig(epochs=6, lr=5e-3, log_interval=6)

    train_loader, val_loader, spec, in_ch = build_synthetic_dataloaders(data_cfg, seed=0)
    model = UnifiedGestureGCN(cfg=model_cfg, spec=spec, in_channels=in_ch)
    trainer = Trainer(model=model, cfg=train_cfg, device=torch.device("cpu"))

    history = trainer.fit(train_loader, val_loader)
    assert history[-1].val_acc >= 0.85

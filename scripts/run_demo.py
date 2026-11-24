"""CLI entry point to run the minimal Plan A static+dynamic gesture recognition demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gesture_gcn.config import ExperimentConfig
from gesture_gcn.data.synthetic import GESTURE_LABELS, build_synthetic_dataloaders
from gesture_gcn.engine.trainer import Trainer, benchmark_inference, resolve_device
from gesture_gcn.models.network import UnifiedGestureGCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Unified Static+Dynamic Hand Gesture GCN (Plan A) demo."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
        help="Path to YAML experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    print(f"=== Experiment: {cfg.experiment_name} | Device: {device} ===")
    train_loader, val_loader, spec, in_channels = build_synthetic_dataloaders(
        cfg.data, seed=cfg.seed
    )

    model = UnifiedGestureGCN(cfg=cfg.model, spec=spec, in_channels=in_channels)
    num_params = model.count_parameters()
    print(
        f"Skeleton: {spec.name} ({spec.num_joints} joints) | "
        f"Input channels: {in_channels} | Trainable params: {num_params / 1e6:.3f}M"
    )
    print(f"Classes ({len(GESTURE_LABELS)}): {', '.join(GESTURE_LABELS)}")

    trainer = Trainer(model=model, cfg=cfg.train, device=device)
    history = trainer.fit(train_loader, val_loader)

    latency_ms = benchmark_inference(
        model=model,
        in_channels=in_channels,
        seq_len=cfg.data.seq_len,
        num_joints=spec.num_joints,
        device=device,
    )
    best_val_acc = max(m.val_acc for m in history)
    print(
        f"\nDone! Best Val Acc: {best_val_acc * 100:.2f}% | "
        f"Single-sample Latency: {latency_ms:.2f} ms ({1000.0 / max(latency_ms, 1e-6):.1f} FPS)"
    )


if __name__ == "__main__":
    main()

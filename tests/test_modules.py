"""Unit tests for Plan A model modules and ablation switches."""

from __future__ import annotations

import pytest
import torch

from gesture_gcn.config import ModelConfig
from gesture_gcn.data.skeleton import get_skeleton_spec
from gesture_gcn.models.network import UnifiedGestureGCN


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # Full Plan A
        {"use_dre": False},
        {"use_mstcn": False},
        {"use_aru": False, "use_dru": True},
        {"use_aru": True, "use_dru": False},
    ],
)
def test_model_forward_and_backward_ablations(overrides: dict) -> None:
    cfg = ModelConfig(num_classes=6, channels=[32, 32, 32, 64], num_groups=4, **overrides)
    spec = get_skeleton_spec("mediapipe_21")
    model = UnifiedGestureGCN(cfg=cfg, spec=spec, in_channels=6)

    x = torch.randn(2, 6, 24, 21)
    dr = torch.rand(2, 3, 21)
    logits = model(x, dr)

    assert logits.shape == (2, 6)
    loss = logits.sum()
    loss.backward()

    # Ensure gradients flow cleanly without NaNs
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"

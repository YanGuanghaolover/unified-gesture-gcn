"""Unit tests for skeleton topology registry and preprocessing transforms."""

from __future__ import annotations

import pytest
import torch

from gesture_gcn.data.skeleton import get_skeleton_spec
from gesture_gcn.data.transforms import SkeletonPreprocessor


@pytest.mark.parametrize("skeleton_type,expected_n", [("mediapipe_21", 21), ("shrec_22", 22)])
def test_adjacency_properties(skeleton_type: str, expected_n: int) -> None:
    spec = get_skeleton_spec(skeleton_type)
    adj = spec.build_adjacency(self_loops=True, normalize=True)
    assert adj.shape == (expected_n, expected_n)
    assert torch.allclose(adj, adj.T, atol=1e-5)
    assert torch.all(adj.diag() > 0)


def test_preprocessor_distinguishes_static_and_dynamic() -> None:
    spec = get_skeleton_spec("mediapipe_21")
    pre = SkeletonPreprocessor(spec, use_bone_fusion=True, scale_floor=0.05)

    # Static sequence with tiny jitter
    base = torch.randn(3, 1, 21)
    static_seq = base.repeat(1, 32, 1) + torch.randn(3, 32, 21) * 0.001
    # Dynamic sequence with large translation
    t_shift = torch.linspace(0.0, 1.0, 32).view(1, 32, 1)
    dynamic_seq = base.repeat(1, 32, 1)
    dynamic_seq[0:1] = dynamic_seq[0:1] + t_shift

    static_feat, static_dr = pre(static_seq)
    dyn_feat, dyn_dr = pre(dynamic_seq)

    assert static_feat.shape == (6, 32, 21)
    assert dyn_feat.shape == (6, 32, 21)
    assert static_dr.shape == (3, 21)
    assert dyn_dr.mean() > static_dr.mean() * 10.0
    assert torch.isfinite(static_feat).all()
    assert torch.isfinite(dyn_feat).all()

"""Tests for real-time inference buffer, recognizer checkpoint I/O, and FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from fastapi.testclient import TestClient

from gesture_gcn.config import DataConfig, ModelConfig
from gesture_gcn.data.skeleton import get_skeleton_spec
from gesture_gcn.data.transforms import SkeletonPreprocessor
from gesture_gcn.inference.buffer import SlidingWindowBuffer
from gesture_gcn.inference.recognizer import GestureRecognizer
from gesture_gcn.models.network import UnifiedGestureGCN
from gesture_gcn.web.app import create_app


def test_sliding_window_buffer_and_checkpoint(tmp_path: Path) -> None:
    buf = SlidingWindowBuffer(seq_len=16, num_joints=21)
    for _ in range(5):
        buf.push(torch.randn(21, 3))
    seq = buf.get_sequence()
    assert seq.shape == (3, 16, 21)

    spec = get_skeleton_spec("mediapipe_21")
    data_cfg = DataConfig(seq_len=16)
    model_cfg = ModelConfig(num_classes=4, channels=[32, 32], num_groups=4)
    pre = SkeletonPreprocessor(spec, use_bone_fusion=True)
    model = UnifiedGestureGCN(cfg=model_cfg, spec=spec, in_channels=pre.out_channels)
    labels = ["open", "fist", "swipe", "grab"]

    rec = GestureRecognizer(model, pre, labels, torch.device("cpu"))
    pred = rec.predict(seq)
    assert pred.label in labels
    assert 0.0 <= pred.confidence <= 1.0

    ckpt = rec.save_checkpoint(tmp_path / "test.pt", data_cfg)
    loaded, loaded_data_cfg = GestureRecognizer.from_checkpoint(ckpt, torch.device("cpu"))
    assert loaded.labels == labels
    assert loaded_data_cfg.seq_len == 16


def test_web_app_status_and_websocket(tmp_path: Path) -> None:
    app = create_app(
        checkpoint_path=tmp_path / "ckpt.pt",
        custom_store_path=tmp_path / "custom.json",
    )
    client = TestClient(app)

    res = client.get("/api/status")
    assert res.status_code == 200
    body = res.json()
    assert body["num_joints"] == 21
    assert len(body["labels"]) == 6

    with client.websocket_connect("/ws/predict") as ws:
        frame = [[0.5, 0.5, 0.0] for _ in range(21)]
        ws.send_json({"landmarks": frame})
        out = ws.receive_json()
        assert "label" in out
        assert "motion_energy" in out
        assert "latency_ms" in out

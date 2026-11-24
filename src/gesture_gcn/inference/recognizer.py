"""Real-time gesture recognizer with checkpoint I/O, EMA smoothing, and DRE motion reporting."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from gesture_gcn.config import DataConfig, ModelConfig
from gesture_gcn.data.skeleton import HandSkeletonSpec, get_skeleton_spec
from gesture_gcn.data.transforms import SkeletonPreprocessor
from gesture_gcn.models.network import UnifiedGestureGCN


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    motion_energy: float
    is_dynamic: bool
    latency_ms: float


class GestureRecognizer:
    """Thread-safe-friendly inference wrapper around UnifiedGestureGCN."""

    def __init__(
        self,
        model: UnifiedGestureGCN,
        preprocessor: SkeletonPreprocessor,
        labels: list[str],
        device: torch.device,
        ema_alpha: float = 0.45,
        dynamic_threshold: float = 0.08,
    ) -> None:
        self.model = model.to(device).eval()
        self.preprocessor = preprocessor
        self.labels = list(labels)
        self.device = device
        self.ema_alpha = ema_alpha
        self.dynamic_threshold = dynamic_threshold
        self._ema_probs: torch.Tensor | None = None

    def reset_smoothing(self) -> None:
        self._ema_probs = None

    @torch.no_grad()
    def predict(self, raw_seq: torch.Tensor) -> PredictionResult:
        """Run inference on a raw skeleton sequence of shape (3, T, N)."""
        start = time.perf_counter()
        features, dynamic_range = self.preprocessor(raw_seq)
        motion_energy = float(dynamic_range.mean().item())

        x = features.unsqueeze(0).to(self.device)
        dr = dynamic_range.unsqueeze(0).to(self.device)

        logits = self.model(x, dr)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()

        if self._ema_probs is None or self._ema_probs.shape != probs.shape:
            self._ema_probs = probs
        else:
            self._ema_probs = self.ema_alpha * probs + (1.0 - self.ema_alpha) * self._ema_probs

        conf, idx = self._ema_probs.max(dim=0)
        latency_ms = (time.perf_counter() - start) * 1000.0

        prob_map = {
            self.labels[i]: round(float(self._ema_probs[i].item()), 4)
            for i in range(len(self.labels))
        }
        return PredictionResult(
            label=self.labels[int(idx.item())],
            confidence=float(conf.item()),
            probabilities=prob_map,
            motion_energy=round(motion_energy, 4),
            is_dynamic=motion_energy >= self.dynamic_threshold,
            latency_ms=round(latency_ms, 2),
        )

    def save_checkpoint(self, path: str | Path, data_cfg: DataConfig) -> Path:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "model_cfg": self.model.cfg.model_dump(),
            "data_cfg": data_cfg.model_dump(),
            "labels": self.labels,
            "in_channels": self.preprocessor.out_channels,
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device,
    ) -> tuple[GestureRecognizer, DataConfig]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        model_cfg = ModelConfig.model_validate(payload["model_cfg"])
        data_cfg = DataConfig.model_validate(payload["data_cfg"])
        labels: list[str] = list(payload["labels"])
        in_channels: int = int(payload["in_channels"])

        spec: HandSkeletonSpec = get_skeleton_spec(data_cfg.skeleton_type)
        preprocessor = SkeletonPreprocessor(
            spec=spec,
            use_bone_fusion=data_cfg.use_bone_fusion,
            scale_floor=data_cfg.norm_scale_floor,
        )
        model = UnifiedGestureGCN(cfg=model_cfg, spec=spec, in_channels=in_channels)
        model.load_state_dict(payload["model_state"])
        recognizer = cls(
            model=model,
            preprocessor=preprocessor,
            labels=labels,
            device=device,
        )
        return recognizer, data_cfg

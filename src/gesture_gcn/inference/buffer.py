"""Fixed-length sliding window buffer for real-time 3D hand skeleton streams."""

from __future__ import annotations

from collections import deque

import torch


class SlidingWindowBuffer:
    """Maintains the latest `seq_len` frames of 3D landmarks shaped (3, N)."""

    def __init__(self, seq_len: int, num_joints: int = 21) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2.")
        self.seq_len = seq_len
        self.num_joints = num_joints
        self._frames: deque[torch.Tensor] = deque(maxlen=seq_len)

    def push(self, landmarks: list[list[float]] | torch.Tensor) -> None:
        """Append one frame of shape (N, 3) or (3, N)."""
        tensor = torch.as_tensor(landmarks, dtype=torch.float32)
        if tensor.shape == (self.num_joints, 3):
            tensor = tensor.transpose(0, 1).contiguous()
        elif tensor.shape != (3, self.num_joints):
            raise ValueError(
                f"Expected landmarks shape ({self.num_joints}, 3) or (3, {self.num_joints}), "
                f"got {tuple(tensor.shape)}"
            )
        self._frames.append(tensor)

    def __len__(self) -> int:
        return len(self._frames)

    def is_ready(self, min_frames: int | None = None) -> bool:
        target = self.seq_len if min_frames is None else min(min_frames, self.seq_len)
        return len(self._frames) >= target

    def get_sequence(self) -> torch.Tensor:
        """Return a (3, seq_len, N) tensor, padding the front with the earliest frame if needed."""
        if not self._frames:
            raise RuntimeError("SlidingWindowBuffer is empty.")
        frames = list(self._frames)
        if len(frames) < self.seq_len:
            pad = [frames[0]] * (self.seq_len - len(frames))
            frames = pad + frames
        return torch.stack(frames, dim=1)  # (3, T, N)

    def clear(self) -> None:
        self._frames.clear()

"""Launch the live UnifiedGestureGCN Web Studio (FastAPI + WebSocket + MediaPipe Hands JS)."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from gesture_gcn.web.app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live webcam hand gesture recognition Web Studio."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(config_path=args.config)
    print(f"Starting Unified Gesture GCN Web Studio at http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

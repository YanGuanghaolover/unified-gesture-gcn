"""Download @mediapipe/hands static assets locally for zero-CDN Web Studio operation."""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path

ASSET_URLS: tuple[str, ...] = (
    "https://registry.npmmirror.com/@mediapipe/hands/-/hands-0.4.1675469240.tgz",
    "https://registry.npmjs.org/@mediapipe/hands/-/hands-0.4.1675469240.tgz",
)


def download_mediapipe_hands(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if (target_dir / "hands.js").exists() and (target_dir / "hand_landmark_lite.tflite").exists():
        print(f"MediaPipe Hands assets already present in {target_dir}")
        return

    raw_bytes: bytes | None = None
    last_err: Exception | None = None
    for url in ASSET_URLS:
        try:
            print(f"Downloading MediaPipe Hands assets from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "gesture-gcn-setup/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw_bytes = resp.read()
            break
        except Exception as exc:
            last_err = exc
            print(f"  Failed ({exc}), trying next mirror...")

    if raw_bytes is None:
        raise RuntimeError(f"Failed to download MediaPipe Hands assets: {last_err}")

    with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) <= 1:
                continue
            rel_path = Path(*parts[1:])
            dest = target_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is not None:
                dest.write_bytes(extracted.read())

    print(f"Successfully extracted MediaPipe Hands assets to {target_dir}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    target = project_root / "src" / "gesture_gcn" / "web" / "static" / "mediapipe" / "hands"
    download_mediapipe_hands(target)


if __name__ == "__main__":
    main()

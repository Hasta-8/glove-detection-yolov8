"""Download the fine-tuned detector from the GitHub release and verify its SHA-256.

uv run python tools/download_model.py            # -> models/best_50epc.pt
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

URL = "https://github.com/Hasta-8/glove-detection-yolov8/releases/download/model-v1/best_50epc.pt"
SHA256 = "cbceb1a10dc68f2377b4f351a4309771147c48598f138620bea045cd30b93be8"
DEST = Path(__file__).resolve().parent.parent / "models" / "best_50epc.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if DEST.exists() and sha256(DEST) == SHA256:
        print(f"already present and verified: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=DEST.parent, delete=False) as tmp:
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL) as r:
            while chunk := r.read(1 << 20):
                tmp.write(chunk)
    got = sha256(Path(tmp.name))
    if got != SHA256:
        Path(tmp.name).unlink()
        print(f"checksum mismatch: expected {SHA256}, got {got}", file=sys.stderr)
        return 1
    Path(tmp.name).replace(DEST)
    print(f"saved and verified: {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

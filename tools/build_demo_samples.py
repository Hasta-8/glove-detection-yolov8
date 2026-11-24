"""Build the demo's samples from input_images/: a resized copy of each photo and the
API's response for it (through the real /v1/detect, in-process).

    uv run python tools/build_demo_samples.py

"What to look at" notes are written from the results (they describe what the model
did on each photo, including its failures).
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402

SRC = ROOT / "input_images"
OUT = ROOT / "demo" / "samples"
NOTES = json.loads((SRC / "notes.json").read_text()) if (SRC / "notes.json").exists() else {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    credits = json.loads((SRC / "credits.json").read_text())
    with TestClient(create_app(get_settings())) as client:
        while client.get("/ready").status_code != 200:
            time.sleep(0.3)
        manifest = []
        for c in credits:
            sid = Path(c["file"]).stem
            img = Image.open(SRC / c["file"]).convert("RGB")
            img.thumbnail((1280, 1280))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=88)
            (OUT / f"{sid}.jpg").write_bytes(buf.getvalue())
            r = client.post(
                "/v1/detect", files={"file": (f"{sid}.jpg", buf.getvalue(), "image/jpeg")}
            )
            r.raise_for_status()
            body = r.json() | {"request_id": "precomputed"}
            (OUT / f"{sid}.json").write_text(json.dumps(body, indent=1))
            note = NOTES.get(sid, {})
            manifest.append(
                {
                    "id": sid,
                    "title": note.get("title", sid.replace("-", " ")),
                    "look_at": note.get("look_at", ""),
                    "image": f"samples/{sid}.jpg",
                    "result": f"samples/{sid}.json",
                    "credit": {
                        k: c[k] for k in ("title", "page", "author", "licence", "licence_url")
                    },
                }
            )
            print(sid, [(d["label"], d["confidence"]) for d in body["detections"]])
        (OUT / "samples.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

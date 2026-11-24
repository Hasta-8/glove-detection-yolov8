import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app.config import Settings


def jpeg(w=320, h=240) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (w, h), (200, 200, 200))
    img.paste(Image.new("RGB", (40, 40), (255, 0, 0)), (10, 20))  # a red square, top left
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


class FakeDetector:
    def __init__(self):
        self.seen = []

    def detect(self, image, conf_threshold=None):
        self.seen.append((image.shape, image[40, 30].tolist(), conf_threshold))
        return [
            {"label": "bare_hand", "confidence": 0.8, "bbox": [10, 20, 50, 60]},
            {"label": "gloved_hand", "confidence": 0.6, "bbox": [100, 100, 150, 160]},
        ]


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, max_upload_mb=1, busy_timeout_s=0.2, demo_dir=tmp_path)


@pytest.fixture
def make_client(settings):
    def make(det=None, **overrides):
        s = settings.model_copy(update=overrides)
        return TestClient(main.create_app(s, det or FakeDetector()), raise_server_exceptions=False)

    return make


def post(client, data, **params):
    return client.post("/v1/detect", params=params, files={"file": ("x.jpg", data, "image/jpeg")})


def test_detect_returns_the_contract_and_bgr_pixels(make_client):
    det = FakeDetector()
    r = post(make_client(det), jpeg())
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["counts"] == {"gloved_hand": 1, "bare_hand": 1}
    assert b["detections"][0] == {"label": "bare_hand", "confidence": 0.8, "bbox": [10, 20, 50, 60]}
    assert b["image"]["width"] == 320 and b["confidence_threshold"] == 0.25
    shape, pixel, conf = det.seen[0]
    assert shape == (240, 320, 3) and conf is None
    assert pixel[2] > 200 and pixel[0] < 60  # red square arrives in OpenCV's BGR order


def test_confidence_override(make_client):
    det = FakeDetector()
    r = post(make_client(det), jpeg(), confidence=0.6)
    assert r.json()["confidence_threshold"] == 0.6 and det.seen[0][2] == 0.6
    assert post(make_client(), jpeg(), confidence=2).status_code == 422


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (b"nope", 415, "unsupported_type"),
        (b"", 422, "empty_file"),
        (jpeg(16, 16), 422, "too_small"),
    ],
)
def test_bad_uploads(make_client, data, status, code):
    r = post(make_client(), data)
    assert (r.status_code, r.json()["error"]) == (status, code)


def test_oversized_upload(make_client):
    r = post(make_client(), b"\xff" * (1024 * 1024 + 1))
    assert (r.status_code, r.json()["error"]) == (413, "too_large")


def test_ready_is_503_until_loaded(settings, monkeypatch):
    gate = threading.Event()

    def slow(_s):
        gate.wait(5)
        return FakeDetector()

    monkeypatch.setattr(main, "load_detector", slow)
    with TestClient(main.create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert post(client, jpeg()).json()["error"] == "not_ready"
        gate.set()
        for _ in range(50):
            if client.get("/ready").status_code == 200:
                break
            time.sleep(0.05)
        assert client.get("/ready").json()["status"] == "ready"


def test_errors_hide_internals(make_client):
    class Broken:
        def detect(self, image, conf_threshold=None):
            raise RuntimeError("/secret/path")

    r = post(make_client(Broken()), jpeg())
    assert r.status_code == 500 and "/secret" not in r.text

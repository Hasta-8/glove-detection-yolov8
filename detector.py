"""HandDetector: the fine-tuned YOLOv8s model, returning plain-dict detections.

The image array goes straight to the model. (The January version wrote each image to
a temporary JPEG first, meaning to match the `yolo predict` CLI; experiment V3 showed
the re-compression changed the boxes on 83% of validation images, while passing the
array gives exactly the CLI's result on all 882.)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "best_50epc.pt"
DEFAULT_CONFIDENCE = 0.25  # Ultralytics' default; the threshold the evaluation used (V4)
IMAGE_SIZE = 640  # the training size
SCRATCH = str(Path(tempfile.gettempdir()) / "glove-detection")


def to_detections(cls, conf, xyxy, names: dict, width: int, height: int) -> list[dict]:
    """Model outputs → [{label, confidence, bbox}], most confident first.

    Boxes are integer pixels [left, top, right, bottom], rounded and clipped to the image.
    """
    dets = []
    for c, s, (x1, y1, x2, y2) in zip(cls, conf, xyxy, strict=True):
        box = [
            int(np.clip(round(x1), 0, width)),
            int(np.clip(round(y1), 0, height)),
            int(np.clip(round(x2), 0, width)),
            int(np.clip(round(y2), 0, height)),
        ]
        dets.append({"label": names[int(c)], "confidence": round(float(s), 3), "bbox": box})
    return sorted(dets, key=lambda d: -d["confidence"])


class HandDetector:
    """Gloved vs bare hand detection with the fine-tuned YOLOv8s."""

    def __init__(
        self, conf_threshold: float = DEFAULT_CONFIDENCE, model_path: str | Path = DEFAULT_MODEL
    ) -> None:
        from ultralytics import YOLO

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}; run `uv run python tools/download_model.py`."
            )
        self.conf_threshold = conf_threshold
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names

    def detect(self, image: np.ndarray, conf_threshold: float | None = None) -> list[dict]:
        """Detect hands in a BGR image (as `cv2.imread` returns it)."""
        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        # Ultralytics creates an (empty) output folder per predictor even without saving;
        # keep it out of the working directory instead of deleting a `runs/` folder there.
        r = self.model.predict(
            image, conf=conf, imgsz=IMAGE_SIZE, verbose=False, save=False, project=SCRATCH
        )[0]
        b = r.boxes
        h, w = image.shape[:2]
        return to_detections(
            b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist(), self.class_names, w, h
        )

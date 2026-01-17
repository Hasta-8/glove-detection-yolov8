"""
detector.py

Encapsulates YOLOv8-based hand detection logic.
Responsibilities:
- Load trained YOLO model
- Run inference on input images
- Return detections in a clean, model-agnostic format
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import tempfile
import os


class HandDetector:
    """
    Wrapper class around YOLOv8 for gloved vs ungloved hand detection.
    """

    def __init__(self, conf_threshold=0.25, model_path="models/best_50epc.pt"):
        """
        Initializes the detector.

        Args:
            conf_threshold (float): Minimum confidence score for detections.
            model_path (str): Path to trained YOLO model weights.
        """
        self.conf_threshold = conf_threshold

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load YOLO model
        self.model = YOLO(str(model_path))

        # Class ID → label mapping
        self.class_names = self.model.names

    def detect(self, image):
        """
        Runs YOLOv8 inference on a single image.

        Important design choice:
        - Image is written to disk temporarily to force the same preprocessing
          pipeline as the YOLO CLI (letterboxing, resizing, etc.)

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            list: List of detection dictionaries with keys:
                  - label
                  - confidence
                  - bbox [x1, y1, x2, y2]
        """

        # Save image temporarily to mimic YOLO CLI behavior
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, image)

        # Run YOLO inference
        results = self.model.predict(
            source=temp_path,
            conf=self.conf_threshold,
            imgsz=640,
            verbose=False
        )

        detections = []

        # Parse YOLO results
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.class_names[cls_id]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })

        # Remove temporary file
        os.remove(temp_path)

        return detections

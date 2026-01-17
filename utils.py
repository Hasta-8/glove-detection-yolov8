"""
utils.py

Utility functions for:
- Drawing bounding boxes and labels on images
- Saving detection results to JSON logs
"""

import json
from pathlib import Path
import cv2


def draw_detections(image, detections):
    """
    Draws bounding boxes and labels on an image.

    Args:
        image (np.ndarray): Original image in BGR format.
        detections (list): List of detection dictionaries with keys:
                           - label
                           - confidence
                           - bbox [x1, y1, x2, y2]

    Returns:
        np.ndarray: Annotated image.
    """
    h, w = image.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        # Clip bounding box coordinates to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 🎨 Class-based color coding
        # Red for unsafe (bare hand), Green for safe (gloved hand)
        if "bare" in label:
            color = (0, 0, 255)      # Red (BGR)
        else:
            color = (0, 255, 0)      # Green (BGR)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label text
        cv2.putText(
            image,
            f"{label}: {conf:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return image


def save_json_log(filename, detections, log_dir: Path):
    """
    Saves detection results for a single image to a JSON file.

    Args:
        filename (str): Original image filename.
        detections (list): List of detection dictionaries.
        log_dir (Path): Directory where JSON logs should be saved.
    """
    log_data = {
        "filename": filename,
        "detections": detections
    }

    # Create JSON filename based on image name
    log_path = log_dir / f"{Path(filename).stem}.json"

    # Write JSON log with indentation for readability
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

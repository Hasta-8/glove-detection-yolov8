"""
detection_script.py

CLI-driven script to perform gloved vs ungloved hand detection on a folder of images.
Responsibilities:
- Parse command-line arguments
- Load images from input directory
- Run detection using HandDetector
- Save annotated images and JSON logs
"""

import argparse
from pathlib import Path

import cv2

from detector import DEFAULT_CONFIDENCE, HandDetector
from utils import draw_detections, save_json_log

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    """
    Parses command-line arguments for the detection script.

    Returns:
        argparse.Namespace: Parsed arguments including input path,
                            output directory, and confidence threshold.
    """
    parser = argparse.ArgumentParser(description="Gloved vs Ungloved Hand Detection")

    parser.add_argument("--input", required=True, help="Path to input image folder")

    parser.add_argument(
        "--output", default="output", help="Root directory to save outputs (images and JSON logs)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Confidence threshold for detections (default 0.25)",
    )

    return parser.parse_args()


def main():
    """
    Main execution function.

    Workflow:
    1. Parse CLI arguments
    2. Create output directories
    3. Load YOLO-based HandDetector
    4. Iterate over images and run detection
    5. Save annotated images and JSON logs
    """
    args = parse_args()

    # Resolve input and output paths
    input_dir = Path(args.input)
    output_root = Path(args.output)

    images_dir = output_root / "images"
    logs_dir = output_root / "logs"

    # Create output directories if they don't exist
    images_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detection model
    detector = HandDetector(conf_threshold=args.confidence)

    # Collect supported image formats
    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    if not image_paths:
        print("[WARN] No images found in input directory.")
        return

    # Process images one by one
    for img_path in image_paths:
        image = cv2.imread(str(img_path))

        # Skip unreadable images
        if image is None:
            print(f"[WARN] Could not read {img_path.name}, skipping.")
            continue

        # Run hand detection
        detections = detector.detect(image)

        # Draw bounding boxes and labels
        annotated = draw_detections(image, detections)

        # Save annotated image
        cv2.imwrite(str(images_dir / img_path.name), annotated)

        # Save detection results as JSON
        save_json_log(img_path.name, detections, logs_dir)

    print(f"[INFO] Processing complete. Results saved to: {output_root}")


if __name__ == "__main__":
    main()

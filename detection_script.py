"""
detection_script.py

CLI-driven script to perform gloved vs ungloved hand detection on a folder of images.
Responsibilities:
- Parse command-line arguments
- Load images from input directory
- Run detection using HandDetector
- Save annotated images and JSON logs
- Clean up Ultralytics side-effects
"""

import argparse
from pathlib import Path
import cv2
import shutil

from detector import HandDetector
from utils import save_json_log, draw_detections


def parse_args():
    """
    Parses command-line arguments for the detection script.

    Returns:
        argparse.Namespace: Parsed arguments including input path,
                            output directory, and confidence threshold.
    """
    parser = argparse.ArgumentParser(
        description="Gloved vs Ungloved Hand Detection"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input image folder"
    )

    parser.add_argument(
        "--output",
        default="output",
        help="Root directory to save outputs (images and JSON logs)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for detections"
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
    6. Clean up Ultralytics auto-generated folders
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
    image_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

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

    # 🔹 Clean up YOLO auto-created "runs/" directory
    # This directory is created internally by Ultralytics during inference
    if Path("runs").exists():
        shutil.rmtree("runs")


if __name__ == "__main__":
    main()

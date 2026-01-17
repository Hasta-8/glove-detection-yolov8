# Gloved vs Ungloved Hand Detection

## Overview

This project implements an object detection pipeline to identify whether workers are wearing gloves in factory images. The system detects and classifies hands into two categories:

- `gloved_hand`
- `bare_hand`

The pipeline is designed to be CLI-driven, modular, and suitable for real-world deployment scenarios such as factory CCTV snapshots or camera frames.

---

## Dataset

**Dataset Name:** Personal Protective Equipment (PPE) Dataset  
**Source:** [Roboflow Universe](https://universe.roboflow.com/)

### Dataset Details

- Original dataset contains ~12,000 images
- Images depict factory workers wearing various protective equipment
- Original dataset has 12 PPE-related classes

### Dataset Filtering & Relabeling

For this project:

- Only glove and no-glove related classes were retained
- All other PPE classes were removed
- The retained classes were renamed to:
  - `gloved_hand`
  - `bare_hand`

This filtering ensured the dataset directly aligned with the task of glove compliance detection while keeping the training data realistic and industrial in nature.

---

## Model Used

| Parameter | Value |
|-----------|-------|
| **Model Architecture** | YOLOv8s |
| **Framework** | Ultralytics YOLOv8 |
| **Task Type** | Object Detection |

YOLOv8s was chosen because it offers a strong balance between accuracy and inference speed, making it suitable for real-time or near real-time factory environments.

---

## Training & Fine-Tuning Details

The YOLOv8s model was fine-tuned on the filtered PPE dataset using the following setup:

- **Training epochs:** 50
- **Image size:** 640 × 640
- **Task:** Detection of `gloved_hand` and `bare_hand`
- **Augmentation:** Default YOLOv8 augmentations (including mosaic and scaling)

### Saved Model Checkpoints

During training, the best-performing checkpoints were saved at multiple stages:

- `best_10epc.pt` → Best model after 10 epochs
- `best_20epc.pt` → Best model after 20 epochs
- `best_50epc.pt` → Best model after 50 epochs

All three model weights are available.

### Selecting Which Model to Use

The model used for inference can be easily changed in `detector.py`.

**Location:**
```python
class HandDetector:
    def __init__(self, conf_threshold=0.25, model_path="models/best_20epc.pt"):
```

**How to switch models:**

To use a different checkpoint, simply change the `model_path`:

```python
model_path="models/best_10epc.pt"
```

or

```python
model_path="models/best_50epc.pt"
```

No other code changes are required.

---

## Preprocessing & Inference Notes

- Input images are processed using the same preprocessing path as the YOLO CLI
- Images are temporarily written to disk before inference to ensure:
  - Consistent letterboxing
  - Correct resizing
  - Identical behavior to `yolo detect predict`
- Bounding boxes are:
  - **Green** for `gloved_hand`
  - **Red** for `bare_hand`
- YOLO's auto-created `runs/` directory is cleaned up after inference to keep the workspace tidy

---

## Results

### ✅ What Worked Well

- Fine-tuning on a filtered industrial PPE dataset produced strong results
- YOLOv8s performed reliably on validation images
- Using the same inference path as the YOLO CLI significantly improved local inference quality
- Modular pipeline design made debugging and iteration easier

### ⚠️ What Didn't Work / Limitations

- Performance drops on images that differ significantly from the training distribution (domain shift)
- Small, distant hands in wide-angle factory images are harder to detect

**This can be improved further with:**
- Additional factory-specific training data
- Image tiling or multi-scale inference

---

## How to Run the Script

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Project Structure

```
Part_1_Glove_Detection/
├── detection_script.py
├── detector.py
├── utils.py
├── models/
│   ├── best_10epc.pt
│   ├── best_20epc.pt
│   └── best_50epc.pt
├── input_images/
├── output/
└── requirements.txt
```

### 3. Run Inference

```bash
python detection_script.py \
  --input input_images \
  --output results \
  --confidence 0.25
```

### 4. Outputs

```
results/
├── images/   # Annotated images
└── logs/     # JSON detection logs
```

Each JSON file contains detections in the following format:

```json
{
  "filename": "image1.jpg",
  "detections": [
    {
      "label": "gloved_hand",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2]
    },
    {
      "label": "bare_hand",
      "confidence": 0.85,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

**Output Format Details:**
- `filename`: Name of the processed image
- `detections`: Array of detected objects
  - `label`: Class label (`gloved_hand` or `bare_hand`)
  - `confidence`: Detection confidence score (0-1)
  - `bbox`: Bounding box coordinates `[x1, y1, x2, y2]` where (x1, y1) is top-left and (x2, y2) is bottom-right

---

## Summary

This project demonstrates:

- Fine-tuning a modern object detector on real-world industrial data
- Building a clean, CLI-driven detection pipeline
- Handling practical deployment issues such as preprocessing consistency and output management

The solution is modular, extensible, and aligned with real-world computer vision practices.

---

## License

This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2025 Debadri Chatterjee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Note:** This project uses YOLOv8 from Ultralytics, which is licensed under AGPL-3.0 for open source use.

## Contact

**Debadri Chatterjee**

📧 Email: [debadrichatterjee1997@gmail.com](mailto:debadrichatterjee1997@gmail.com)  
💼 LinkedIn: [linkedin.com/in/debadri-chatterjee](https://www.linkedin.com/in/debadri-chatterjee/)

---

⭐ If you found this project helpful, please consider giving it a star!
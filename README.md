# Glove detection

Detects hands in work photos and classifies each as **`gloved_hand`** or
**`bare_hand`**, for glove-compliance checks. A YOLOv8s model fine-tuned on an
industrial PPE dataset; a command-line tool writes annotated images and JSON logs.
CPU is enough.

> **Status:** complete. The model is published with a checksum, its accuracy is
> measured with the data leakage stated, and there is an API, Docker and a demo page.
> **On scenes it hasn't seen it finds about half of the bare hands, and on ordinary
> photos unlike its factory-camera training data it fails outright**, so treat it as a
> demonstration, not a safety system.

**▶ [Try the online demo](https://hasta-8.github.io/glove-detection-yolov8/)** (precomputed results, runs in the browser)

## Results (measured)

Validation split of the training data (882 images, 2,532 hands), confidence 0.001
for mAP (Ultralytics' default), CPU re-run of the training-time validation.

| Images | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **No near-copy in training (422 images)** | **0.810** | **0.653** | **0.700** | **0.319** |
| All validation images (882) | 0.855 | 0.806 | 0.846 | 0.419 |

| Class, images with no near-copy in training | Precision | Recall | mAP50 |
|---|---|---|---|
| `gloved_hand` | 0.839 | 0.795 | 0.835 |
| `bare_hand` | 0.781 | **0.511** | 0.565 |

- **Why two rows:** 52 % of the validation images are near-copies of training images,
  mostly frames from the same fixed CCTV cameras taken moments apart (found with a
  perceptual hash, checked by eye; `eval/leakage.py`). On those the model has in
  effect seen the answer. The first row leaves them out and is the better estimate
  for new scenes; even it shares camera views with training, so a new site will likely
  do worse.
- **No separate test set:** the source dataset's test split has no glove labels, and
  this validation split also picked the best training epoch, so all numbers lean
  optimistic.

### What goes wrong

Each annotated hand was matched (IoU ≥ 0.5, confidence ≥ 0.25) and counted as found,
found with the wrong class, or missed (`eval/errors.py`):

| Validation images with no near-copy in training | Small hands (< 32 px) | Medium (32–96 px) | Large (> 96 px) | All |
|---|---|---|---|---|
| `gloved_hand` found | 1 of 18 | 0.89 (of 192) | 12 of 12 | 0.82 |
| `bare_hand` found | 5 of 37 | **0.53 (of 553)** | 0.96 (of 52) | 0.54 |

- **It misses hands; it rarely confuses them.** Of 2,532 hands in the whole validation
  set, only 7 were found with the wrong class. Almost every error is a hand not detected.
- **Size decides:** large hands are found about 95 % of the time, small ones 6–25 %.
  Most bare hands in the data are medium-sized, and those are where recall collapses
  on unseen scenes.
- **False alarms** are mostly "bare hand" boxes on background (174 on the 422 images).

### Outside its training data

On eight ordinary work photos from Wikimedia Commons (close-ups, outdoor, a lab; the
demo samples), it made **no correct detection** of about 12 visible hands. Its five
boxes landed on ear defenders, a tool pouch, a worker's head and a basket. The
training images are overhead factory CCTV frames; anything else is a different
problem (an observation on 8 photos, not a benchmark).

## How it works

```
image (any size)
  → YOLOv8s (letterboxed to 640×640), 2 classes: gloved_hand, bare_hand
  → boxes with confidence ≥ 0.25 (the default) → annotated image + JSON log
```

Training (in `yolov8_ppe_dataset_fineTuning.ipynb`, run on Colab, January 2026):
YOLOv8s pretrained on COCO, fine-tuned 50 epochs at 640×640, batch 16, Ultralytics
8.3.252.

**Data:** the Roboflow Universe project `personal-protective-equipment/ppes-kaxsi`,
version 8 (~12,000 images, 12 PPE classes), filtered to its glove and no-glove
classes and relabelled `gloved_hand` / `bare_hand`: 6,984 training and 882 validation
images. The dataset is not redistributed here and the published numbers are
aggregates only.

## Quick start

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). CPU only.

```bash
uv sync                                    # CPU-only torch, versions from uv.lock
uv run python tools/download_model.py      # the model from the GitHub release, checksum verified
uv run python detection_script.py --input input_images --output results    # --confidence 0.25 by default
```

```
results/
├── images/   annotated copies (green = gloved, red = bare)
└── logs/     one JSON file per image
```

```json
{
  "filename": "red-work-glove.jpg",
  "detections": [
    {"label": "gloved_hand", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}
  ]
}
```

`bbox` is integer pixels `[left, top, right, bottom]` in the input image.

## API and demo

```bash
uv run python run.py      # http://127.0.0.1:8004: demo page; API docs at /docs
```

`POST /v1/detect` (multipart `file`, optional `?confidence=`) returns the detections (most confident first),
counts per class, image size and timings; `/health` and `/ready` (503 until the model is loaded). Uploads are
checked from their bytes (JPEG/PNG/WebP, size and pixel limits) and errors have one shape,
`{"error", "detail", "request_id"}`. The demo page shows the sample photos below with precomputed
results, runs them live, and takes uploads. **Docker:** `docker compose up --build` (the model folder is
mounted read-only); ready in about 20 s.

### Sample photos

| Photo | What the model did | Credit |
|---|---|---|
| [Jackhammer, outdoors (gloved)](input_images/jackhammer-gloves.jpg) | Both "gloved hand" boxes are on the ear defenders and the "bare hand" box is on the tool pouch; the yellow-gloved hands themselves are missed. Outdoor, eye-level photos are unlike the overhead factory cameras it was trained on. | [Hk6rsf](https://commons.wikimedia.org/wiki/File:Colombia_Jackhammer_01.jpg), Public domain |
| [Red work glove, close-up](input_images/red-work-glove.jpg) | The only box is a "bare hand" on the worker's head; the large red glove is missed. Close-ups are far from the training data, where hands are small and seen from above. | [Shixart1985](https://commons.wikimedia.org/wiki/File:Man_works_with_glass_in_a_factory_setting_wearing_gloves_and_focusing_on_his_task.jpg), CC BY 2.0 |
| [Handling a metal sheet (gloved)](input_images/metal-sheet-gloves.jpg) | Nothing detected, though a red-gloved hand is in view. | [Shixart1985](https://commons.wikimedia.org/wiki/File:Worker_handling_a_metal_sheet_in_a_manufacturing_facility.jpg), CC BY 2.0 |
| [Welding (gloved)](input_images/welding-gloves.jpg) | Nothing detected: thick welding gloves seen close up never appear in the training data. | [Prowelder87](https://commons.wikimedia.org/wiki/File:Tig_Welding_WP9.jpg), CC BY-SA 4.0 |
| [Lab, nitrile gloves](input_images/lab-nitrile-gloves.jpg) | Nothing detected on two clearly gloved hands. | [M Joko Apriyo Putro](https://commons.wikimedia.org/wiki/File:Arranging_capsules.jpg), CC BY 4.0 |
| [Machine operator (bare hands)](input_images/machine-operator-bare-hands.jpg) | Nothing detected: the bare hands holding a phone are missed. Missed bare hands are this model's main failure, even on its own validation data. | [Shixart1985](https://commons.wikimedia.org/wiki/File:A_worker_operates_a_machine_while_checking_a_mobile_device_in_a_workshop_setting.jpg), CC BY 2.0 |
| [Salt workers, distant (bare hands)](input_images/salt-worker-distant.jpg) | One "gloved hand" box, on the basket. The workers' hands are tiny; small hands are found 6-25% of the time even on training-like data. | [Dey.sandip](https://commons.wikimedia.org/wiki/File:Salt_workers_of_Marakkanam.jpg), CC BY-SA 4.0 |
| [Tools, no hands](input_images/tools-no-hands.jpg) | Correctly empty: no hands, no boxes. | [Edward Middleton](https://commons.wikimedia.org/wiki/File:A_mechanic_at_work_2.jpg), CC BY-SA 4.0 |

## Model

| | |
|---|---|
| File | `models/best_50epc.pt` (22.5 MB), the training run's best checkpoint |
| Download | [release `model-v1`](https://github.com/Hasta-8/glove-detection-yolov8/releases/tag/model-v1) via `tools/download_model.py` |
| SHA-256 | `cbceb1a10dc68f2377b4f351a4309771147c48598f138620bea045cd30b93be8` |
| Licence | AGPL-3.0 (fine-tuned from Ultralytics' pretrained YOLOv8s) |

The 10- and 20-epoch checkpoints from the same run are not published; the 50-epoch
one is the only supported model.

## Performance (measured)

On a 4-core Intel i5-10210U laptop CPU (4 threads), image size 640:

| | |
|---|---|
| Per image, warm (Docker, 1280-px photos) | about 0.15 s; the first request about 2 s |
| Per image, the 882 validation images (one at a time) | median 0.37 s (measured while other work ran) |
| Model load | about 20 s in Docker (cold), a few seconds locally |
| Memory | about 0.5 GB at rest |

Passing the image to the model as an array gives exactly the boxes the `yolo predict`
command gives on all 882 validation images. The January code wrote each image to a
temporary JPEG first "for CLI parity"; that re-compression changed the boxes on 83 % of
them, so it was removed (`eval/inference_paths.py`).

## Limitations

- **Bare hands on unseen scenes:** recall about 0.51 on validation images with no
  near-copy in training. Misses are the main failure (see error analysis).
- **Domain:** trained mostly on a few fixed factory CCTV views; other industries,
  camera angles and lighting are unmeasured.
- **Single images only.** There is no video tracking, so there is no notion of the
  same worker across frames and no compliance-over-time logic.
- **No separate test set** (see Results).
- **Ordinary photos fail:** close-up, eye-level or outdoor photos are outside the
  training distribution (see above).

## Repository

```
detection_script.py   CLI: a folder of images → annotated images + JSON logs
detector.py           HandDetector: loads the model, returns detections
utils.py              drawing and JSON logging
app/, services/       FastAPI service and upload decoding
demo/                 the demo page and its precomputed samples
tools/                download_model.py (fetches and verifies the model), build_demo_samples.py
eval/                 leakage check, error analysis, inference-path check; results/ (metrics only)
tests/                unit and API tests (stand-in model)
Dockerfile, docker-compose.yml   the container (model mounted at run time)
input_images/         openly licensed work photos (credits.json)
yolov8_ppe_dataset_fineTuning.ipynb   data filtering and training, as run on Colab
```

## Licence

The code in this repository is MIT-licensed ([LICENSE](LICENSE)). It runs on
[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is AGPL-3.0,
and the fine-tuned weights are AGPL-3.0 too; distributing a product built on them
falls under the AGPL unless you hold an Ultralytics enterprise licence. Sample photos:
Wikimedia Commons, each under its own licence, listed in
[`input_images/credits.json`](input_images/credits.json).

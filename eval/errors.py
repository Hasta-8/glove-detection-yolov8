"""V4: error analysis by class and hand size, on the validation set (and its clean subset).

uv run python eval/errors.py <dataset-root> [--subset eval/results/valid_clean.txt]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO  # noqa: E402

CONF, IOU = 0.25, 0.5
NAMES = {0: "gloved_hand", 1: "bare_hand"}


def size_bucket(box) -> str:
    side = max(box[2] - box[0], box[3] - box[1])
    return "small" if side < 32 else "medium" if side <= 96 else "large"


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match(gt, det) -> tuple[list[str], list[str]]:
    """Outcome per ground-truth box and per detection (see the tracker, V4)."""
    g_out = ["missed"] * len(gt)
    d_out = ["background"] * len(det)
    if len(gt) and len(det):
        m = iou(gt[:, 1:], det[:, 2:])
        used = set()
        for j in np.argsort(-det[:, 1], kind="stable"):  # most confident first, same class
            cand = [
                i
                for i in np.argsort(-m[:, j])
                if m[i, j] >= IOU and gt[i, 0] == det[j, 0] and i not in used
            ]
            if cand:
                used.add(cand[0])
                g_out[cand[0]] = "found"
                d_out[j] = "correct"
        for i in range(len(gt)):
            if g_out[i] == "missed" and any(
                m[i, j] >= IOU and d_out[j] != "correct" for j in range(len(det))
            ):
                g_out[i] = "wrong_class"
        for j in range(len(det)):
            if d_out[j] == "background" and any(
                m[i, j] >= IOU and g_out[i] != "found" for i in range(len(gt))
            ):
                d_out[j] = "wrong_class"
    return g_out, d_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--subset", default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    model = YOLO("models/best_50epc.pt")
    root = Path(args.root) / "valid"
    names = sorted(p.name for p in (root / "images").iterdir())
    if args.subset:
        keep = set(Path(args.subset).read_text().split())
        names = [n for n in names if n in keep]
    gt_counts, det_counts = Counter(), Counter()
    for n in names:
        img = cv2.imread(str(root / "images" / n))
        h, w = img.shape[:2]
        rows = []
        for line in (root / "labels" / f"{Path(n).stem}.txt").read_text().split("\n"):
            if line.strip():
                c, xc, yc, bw, bh = map(float, line.split())
                rows.append(
                    [c, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]
                )
        gt = np.array(rows).reshape(-1, 5)
        b = model.predict(img, conf=CONF, imgsz=640, verbose=False)[0].boxes
        det = np.column_stack([b.cls.numpy(), b.conf.numpy(), b.xyxy.numpy()]).reshape(-1, 6)
        g_out, d_out = match(gt, det)
        for row, o in zip(gt, g_out, strict=True):
            gt_counts[(NAMES[int(row[0])], size_bucket(row[1:]), o)] += 1
        for row, o in zip(det, d_out, strict=True):
            det_counts[(NAMES[int(row[0])], o)] += 1
    table = {}
    for cls in NAMES.values():
        for size in ("small", "medium", "large", "all"):
            c = {
                o: sum(
                    v
                    for (k, s, oo), v in gt_counts.items()
                    if k == cls and (size == "all" or s == size) and oo == o
                )
                for o in ("found", "wrong_class", "missed")
            }
            n = sum(c.values())
            table[f"{cls}/{size}"] = {
                **c,
                "total": n,
                "recall": round(c["found"] / n, 3) if n else None,
            }
    res = {
        "experiment": "V4",
        "subset": args.subset or "all validation images",
        "images": len(names),
        "conf": CONF,
        "ground_truth": table,
        "detections": {f"{k}/{o}": v for (k, o), v in sorted(det_counts.items())},
    }
    tag = "clean" if args.subset else "all"
    (Path(__file__).resolve().parent / "results" / f"v4_errors_{tag}.json").write_text(
        json.dumps(res, indent=1)
    )
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

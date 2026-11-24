"""V3: file path vs in-memory array vs the January temp-file round trip.

uv run python eval/inference_paths.py <dataset-root> [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO  # noqa: E402

CONF = 0.25


def boxes(result) -> list[tuple]:
    b = result.boxes
    return [
        (int(c), float(s), *map(float, xyxy))
        for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist(), strict=True)
    ]


def same(a: list[tuple], b: list[tuple]) -> bool:
    if len(a) != len(b):
        return False
    a, b = sorted(a), sorted(b)
    return all(
        x[0] == y[0]
        and abs(x[1] - y[1]) <= 1e-4
        and max(abs(p - q) for p, q in zip(x[2:], y[2:], strict=True)) <= 0.5
        for x, y in zip(a, b, strict=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    model = YOLO("models/best_50epc.pt")
    paths = sorted((Path(args.root) / "valid" / "images").iterdir())[: args.limit or None]
    model.predict(str(paths[0]), conf=CONF, imgsz=640, verbose=False)  # warm-up
    t = {"file": [], "array": [], "tempfile": []}
    same_array = same_temp = 0
    diffs = []
    for p in paths:
        t0 = time.perf_counter()
        r_file = boxes(model.predict(str(p), conf=CONF, imgsz=640, verbose=False)[0])
        t1 = time.perf_counter()
        img = cv2.imread(str(p))
        t2 = time.perf_counter()
        r_arr = boxes(model.predict(img, conf=CONF, imgsz=640, verbose=False)[0])
        t3 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
        r_tmp = boxes(model.predict(tmp.name, conf=CONF, imgsz=640, verbose=False)[0])
        os.remove(tmp.name)
        t4 = time.perf_counter()
        t["file"].append(t1 - t0)
        t["array"].append(t3 - t2)
        t["tempfile"].append(t4 - t3)
        same_array += same(r_file, r_arr)
        ok = same(r_file, r_tmp)
        same_temp += ok
        if not ok:
            diffs.append({"image": p.name, "file": len(r_file), "tempfile": len(r_tmp)})
    res = {
        "experiment": "V3",
        "images": len(paths),
        "array_identical_to_file": same_array,
        "tempfile_identical_to_file": same_temp,
        "tempfile_box_count_changed": sum(d["file"] != d["tempfile"] for d in diffs),
        "median_ms": {k: round(float(np.median(v)) * 1000, 1) for k, v in t.items()},
        "examples": diffs[:10],
    }
    out = Path(__file__).resolve().parent / "results" / "v3_inference_paths.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

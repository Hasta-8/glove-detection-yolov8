"""V2: do validation images leak from the training set?

    uv run python eval/leakage.py <dataset-root>     # root holds train/ and valid/

(a) Same source image: Roboflow export names keep the original file name before
    "_jpg.rf." / "_png.rf." (e.g. frame555_jpg.rf.<hash>.jpg). The same original name in
    both splits means the same photo (or an augmented copy of it) is in both.
(b) Near-duplicates: a 64-bit difference hash per image; a validation image within
    6 bits (Hamming distance) of any training image counts as a near-duplicate, which
    catches consecutive video frames and re-crops.

Writes eval/results/v2_leakage.json (names and counts only; no images) and
eval/results/valid_clean.txt, the validation images with neither problem.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

NEAR_BITS = 6  # fixed before running (tracker, V2)
REPORT_BITS = [0, 2, 4, 6, 8, 10]
SOURCE = re.compile(r"^(.*)_(?:jpe?g|png)\.rf\.[0-9a-f]+\.(?:jpe?g|png)$", re.IGNORECASE)


def source_name(filename: str) -> str:
    m = SOURCE.match(filename)
    return m.group(1) if m else Path(filename).stem


def dhash(path: Path) -> int:
    """Difference hash: 9×8 grayscale, one bit per horizontally adjacent pair."""
    g = np.asarray(
        Image.open(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16
    )
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_min(query: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Smallest Hamming distance from each query hash to any pool hash (uint64 arrays)."""
    out = np.empty(len(query), dtype=np.int64)
    for i, q in enumerate(query):
        x = np.bitwise_xor(pool, q)
        out[i] = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).min()
    return out


def main() -> None:
    root = Path(sys.argv[1])
    train = sorted((root / "train" / "images").iterdir())
    valid = sorted((root / "valid" / "images").iterdir())
    train_sources = Counter(source_name(p.name) for p in train)
    same_source = {p.name for p in valid if source_name(p.name) in train_sources}

    th = np.array([dhash(p) for p in train], dtype=np.uint64)
    vh = np.array([dhash(p) for p in valid], dtype=np.uint64)
    dist = hamming_min(vh, th)
    near = {p.name for p, d in zip(valid, dist, strict=True) if d <= NEAR_BITS}
    flagged = same_source | near
    clean = [p for p in valid if p.name not in flagged]

    prefixes = Counter(re.sub(r"\d.*$", "", source_name(p.name)) or "(digits)" for p in valid)
    result = {
        "experiment": "V2",
        "train_images": len(train),
        "valid_images": len(valid),
        "same_source_in_train": len(same_source),
        "near_duplicate_bits": NEAR_BITS,
        "near_duplicates": len(near),
        "near_duplicates_at": {str(b): int((dist <= b).sum()) for b in REPORT_BITS},
        "flagged": len(flagged),
        "clean": len(clean),
        "valid_name_prefixes": dict(prefixes.most_common(12)),
        "flagged_by_prefix": dict(
            Counter(
                re.sub(r"\d.*$", "", source_name(n)) or "(digits)" for n in flagged
            ).most_common(12)
        ),
    }
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    (out / "v2_leakage.json").write_text(json.dumps(result, indent=1))
    (out / "valid_clean.txt").write_text("".join(f"{p.name}\n" for p in clean))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

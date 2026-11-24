import json

import numpy as np

from utils import draw_detections, save_json_log


def blank(h=100, w=120):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_boxes_are_drawn_in_class_colours():
    img = draw_detections(
        blank(),
        [
            {"label": "gloved_hand", "confidence": 0.9, "bbox": [10, 30, 40, 60]},
            {"label": "bare_hand", "confidence": 0.8, "bbox": [60, 30, 90, 60]},
        ],
    )
    assert tuple(img[45, 10]) == (0, 255, 0)  # gloved: green (BGR), left edge
    assert tuple(img[45, 60]) == (0, 0, 255)  # bare: red (BGR)


def test_boxes_outside_the_image_are_clipped():
    img = draw_detections(
        blank(), [{"label": "bare_hand", "confidence": 0.5, "bbox": [-20, -20, 500, 500]}]
    )
    assert img.shape == (100, 120, 3) and tuple(img[50, 0]) == (0, 0, 255)


def test_no_detections_leaves_the_image_unchanged():
    assert not draw_detections(blank(), []).any()


def test_json_log_format(tmp_path):
    dets = [{"label": "gloved_hand", "confidence": 0.912, "bbox": [1, 2, 3, 4]}]
    save_json_log("shift_a/cam1.jpg", dets, tmp_path)
    data = json.loads((tmp_path / "cam1.json").read_text())
    assert data == {"filename": "shift_a/cam1.jpg", "detections": dets}


def test_to_detections_rounds_clips_and_sorts():
    from detector import to_detections

    dets = to_detections(
        cls=[1, 0],
        conf=[0.41234, 0.9],
        xyxy=[[-3.4, 10.6, 50.5, 999.0], [5.2, 6.7, 20.1, 30.9]],
        names={0: "gloved_hand", 1: "bare_hand"},
        width=100,
        height=80,
    )
    assert dets == [
        {"label": "gloved_hand", "confidence": 0.9, "bbox": [5, 7, 20, 31]},
        {"label": "bare_hand", "confidence": 0.412, "bbox": [0, 11, 50, 80]},
    ]


def test_to_detections_empty():
    from detector import to_detections

    assert to_detections([], [], [], {0: "gloved_hand"}, 10, 10) == []

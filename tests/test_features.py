from __future__ import annotations

import numpy as np

from backend.vision.features import classify_colors, smooth_color_labels
from backend.vision.actions import MotionTracker


def test_classify_colors_prefers_torso_over_background() -> None:
    frame = np.full((480, 640, 3), 210, dtype=np.uint8)
    bbox = [100, 60, 540, 460]
    face_bbox = [230, 90, 410, 220]
    pose = {
        "left_shoulder": [0.70, 0.72],
        "right_shoulder": [0.34, 0.72],
        "left_hip": None,
        "right_hip": None,
    }

    # hoodie region in dark blue-ish BGR
    frame[230:380, 190:450] = (150, 85, 35)
    # trousers region
    frame[380:450, 200:440] = (55, 55, 55)
    # face/skin region
    frame[90:220, 230:410] = (170, 190, 220)

    top, bottom = classify_colors(frame, bbox, face_bbox, pose)
    assert top == "碧藍色"
    assert bottom in {"黑色", "灰色"}


def test_classify_colors_keeps_neutral_gray_under_warm_cast() -> None:
    frame = np.full((480, 640, 3), 230, dtype=np.uint8)
    bbox = [100, 60, 540, 460]
    face_bbox = [230, 90, 410, 220]
    pose = {
        "left_shoulder": [0.70, 0.72],
        "right_shoulder": [0.34, 0.72],
        "left_hip": None,
        "right_hip": None,
    }

    # Warm-lit neutral gray shirt.
    frame[235:380, 180:460] = (92, 88, 84)
    frame[90:220, 230:410] = (170, 190, 220)

    top, _ = classify_colors(frame, bbox, face_bbox, pose)
    assert top == "灰色"


def test_classify_colors_detects_lake_green_torso() -> None:
    frame = np.full((480, 640, 3), 228, dtype=np.uint8)
    bbox = [100, 60, 540, 460]
    face_bbox = [230, 90, 410, 220]
    pose = {
        "left_shoulder": [0.70, 0.72],
        "right_shoulder": [0.34, 0.72],
        "left_hip": None,
        "right_hip": None,
    }

    frame[235:380, 180:460] = (170, 150, 70)
    frame[90:220, 230:410] = (170, 190, 220)

    top, _ = classify_colors(frame, bbox, face_bbox, pose)
    assert top == "湖水綠"


def test_smooth_color_labels_prefers_recent_consensus() -> None:
    labels = ["藍色", "深青藍", "深藍色", "深藍色", "深藍色"]
    assert smooth_color_labels(labels) == "深藍色"


def test_defocus_requires_low_focus_not_just_large_bbox() -> None:
    tracker = MotionTracker()
    action = tracker.update(
        area_ratio=0.6,
        center_y_norm=0.5,
        eye_x_norm=0.5,
        pose=None,
        focus_score=2.5,
        defocus_threshold=0.42,
        focus_score_threshold=0.25,
        crouch_delta_threshold=0.18,
    )
    assert action.defocus is False

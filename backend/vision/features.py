from __future__ import annotations

import cv2
import numpy as np


def focus_score(frame: np.ndarray, bbox: list[int]) -> float:
    x1, y1, x2, y2 = bbox
    roi = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0)


def classify_colors(
    frame: np.ndarray,
    bbox: list[int],
    face_bbox: list[int] | None = None,
    pose_keypoints: dict[str, list[float] | None] | None = None,
) -> tuple[str, str]:
    x1, y1, x2, y2 = bbox
    roi = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    if roi.size == 0:
        return "unknown", "unknown"

    top = _extract_top_clothing_roi(frame, bbox, face_bbox, pose_keypoints)
    bottom = _extract_bottom_clothing_roi(frame, bbox)
    return _classify_region_color(top), _classify_region_color(bottom)


def smooth_color_labels(history: list[str], fallback: str = "unknown") -> str:
    if not history:
        return fallback
    scores: dict[str, float] = {}
    for index, label in enumerate(history, start=1):
        if not label or label == "unknown":
            continue
        weight = float(index)
        if label in {"灰色", "黑色", "白色"}:
            weight *= 1.1
        scores[label] = scores.get(label, 0.0) + weight
    if not scores:
        return fallback
    return max(scores.items(), key=lambda item: (item[1], history[::-1].index(item[0]) * -1))[0]


def classify_body_shape(bbox: list[int], frame_shape: tuple[int, int, int]) -> tuple[str, str]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect = width / height
    height_ratio = height / max(1, frame_shape[0])
    if height_ratio > 0.7:
        height_class = "tall"
    elif height_ratio < 0.45:
        height_class = "short"
    else:
        height_class = "medium"
    if aspect > 0.55:
        build_class = "broad"
    elif aspect < 0.38:
        build_class = "slim"
    else:
        build_class = "average"
    return height_class, build_class


def classify_distance(area_ratio: float, near_threshold: float, defocus_threshold: float) -> str:
    if area_ratio >= defocus_threshold:
        return "too_close"
    if area_ratio >= near_threshold:
        return "near"
    if area_ratio >= near_threshold / 2:
        return "mid"
    return "far"


def _extract_top_clothing_roi(
    frame: np.ndarray,
    bbox: list[int],
    face_bbox: list[int] | None,
    pose_keypoints: dict[str, list[float] | None] | None,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    crop_x1 = x1 + int(width * 0.22)
    crop_x2 = x2 - int(width * 0.22)
    crop_y1 = y1 + int(height * 0.40)
    crop_y2 = y1 + int(height * 0.78)
    if face_bbox is not None:
        crop_y1 = max(crop_y1, face_bbox[3] + int(height * 0.02))
    if pose_keypoints:
        left = pose_keypoints.get("left_shoulder")
        right = pose_keypoints.get("right_shoulder")
        left_hip = pose_keypoints.get("left_hip")
        right_hip = pose_keypoints.get("right_hip")
        if left and right:
            shoulder_left = int(min(left[0], right[0]) * frame.shape[1])
            shoulder_right = int(max(left[0], right[0]) * frame.shape[1])
            shoulder_y = int(min(left[1], right[1]) * frame.shape[0])
            shoulder_span = max(1, shoulder_right - shoulder_left)
            crop_x1 = max(crop_x1, shoulder_left - int(shoulder_span * 0.12))
            crop_x2 = min(crop_x2, shoulder_right + int(shoulder_span * 0.12))
            crop_y1 = max(crop_y1, shoulder_y + int(height * 0.03))
        hips = [point for point in (left_hip, right_hip) if point]
        if hips:
            hip_y = int(min(point[1] for point in hips) * frame.shape[0])
            crop_y2 = min(crop_y2, hip_y - int(height * 0.04))
    min_height = max(28, int(height * 0.12))
    if crop_y2 - crop_y1 < min_height:
        crop_y2 = min(frame.shape[0], crop_y1 + min_height)
    return frame[max(0, crop_y1):max(crop_y1 + 1, crop_y2), max(0, crop_x1):max(crop_x1 + 1, crop_x2)]


def _extract_bottom_clothing_roi(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    crop_x1 = x1 + int(width * 0.20)
    crop_x2 = x2 - int(width * 0.20)
    crop_y1 = y1 + int(height * 0.76)
    crop_y2 = y2 - int(height * 0.04)
    return frame[max(0, crop_y1):max(crop_y1 + 1, crop_y2), max(0, crop_x1):max(crop_x1 + 1, crop_x2)]


def _classify_region_color(roi: np.ndarray) -> str:
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    h = hsv[..., 0].reshape(-1)
    s = hsv[..., 1].reshape(-1)
    v = hsv[..., 2].reshape(-1)
    a = lab[..., 1].reshape(-1).astype(np.float32) - 128.0
    b = lab[..., 2].reshape(-1).astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)

    if len(h) > 6000:
        step = max(1, len(h) // 6000)
        h = h[::step]
        s = s[::step]
        v = v[::step]
        chroma = chroma[::step]

    bright_mean = float(np.mean(v))
    sat_mean = float(np.mean(s))
    chroma_median = float(np.median(chroma))
    chroma_p75 = float(np.percentile(chroma, 75))
    colorful = (s > 35) & (v > 35)
    if bright_mean < 55:
        return "黑色"
    if sat_mean < 28 and bright_mean > 180:
        return "白色"
    if chroma_median < 8.5 and chroma_p75 < 12.0:
        if bright_mean > 172:
            return "白色"
        if bright_mean < 62:
            return "黑色"
        return "灰色"
    if sat_mean < 40:
        if bright_mean > 170:
            return "白色"
        if bright_mean < 65:
            return "黑色"
        return "灰色"

    if not np.any(colorful):
        return "灰色"

    dominant_hue = float(np.median(h[colorful]))
    return _classify_colorful_hue(dominant_hue, sat_mean, bright_mean)


def _classify_colorful_hue(dominant_hue: float, sat_mean: float, bright_mean: float) -> str:
    is_dark = bright_mean < 90
    is_deep = sat_mean > 110 and bright_mean < 145
    is_light = bright_mean > 175 and sat_mean < 120

    if dominant_hue < 8 or dominant_hue >= 172:
        return "深紅" if is_deep else "紅色"
    if dominant_hue < 16:
        return "橙色" if not is_light else "蜜桃色"
    if dominant_hue < 26:
        return "金黃色" if is_deep else "黃色"
    if dominant_hue < 38:
        return "黃綠色" if not is_light else "嫩綠色"
    if dominant_hue < 54:
        return "墨綠色" if is_dark else "綠色"
    if dominant_hue < 72:
        return "青綠色" if not is_light else "薄荷綠"
    if dominant_hue < 98:
        if is_dark:
            return "墨青色"
        return "湖水綠"
    if dominant_hue < 108:
        return "碧藍色" if not is_dark else "深青藍"
    if dominant_hue < 124:
        return "天藍色" if is_light else "深藍色"
    if dominant_hue < 142:
        return "深藍色" if is_deep or is_dark else "寶藍色"
    if dominant_hue < 158:
        return "紫色" if not is_light else "粉紫色"
    return "桃紅色" if sat_mean > 120 else "粉紅色"

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from backend.device_utils import backend_label_for_device, get_torch_device


@dataclass
class PoseSignals:
    shoulder_y_norm: float | None = None
    hip_y_norm: float | None = None
    left_wrist_x_norm: float | None = None
    right_wrist_x_norm: float | None = None
    left_wrist_y_norm: float | None = None
    right_wrist_y_norm: float | None = None
    left_shoulder_y_norm: float | None = None
    right_shoulder_y_norm: float | None = None
    pose_confidence: float = 0.0


class PoseTracker:
    def __init__(self, model_path: str, conf: float = 0.25) -> None:
        self.model_path = model_path
        self.conf = conf
        self.device = get_torch_device()
        self._model: YOLO | None = None

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def warmup(self) -> str:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self._ensure_model().predict(frame, conf=self.conf, verbose=False, device=self.device)
        return backend_label_for_device(self.device)

    def detect(self, frame: np.ndarray, person_bbox: list[int]) -> PoseSignals:
        x1, y1, x2, y2 = person_bbox
        roi = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        if roi.size == 0:
            return PoseSignals()
        results = self._ensure_model().predict(roi, conf=self.conf, verbose=False, device=self.device)
        if not results:
            return PoseSignals()
        result = results[0]
        if result.keypoints is None or result.keypoints.data is None or len(result.keypoints.data) == 0:
            return PoseSignals()
        keypoints = result.keypoints.data[0].cpu().numpy()
        if keypoints.shape[0] < 15:
            return PoseSignals()
        frame_height, frame_width = frame.shape[:2]
        return PoseSignals(
            shoulder_y_norm=_avg_norm_y(keypoints, [5, 6], x1, y1, frame_height),
            hip_y_norm=_avg_norm_y(keypoints, [11, 12], x1, y1, frame_height),
            left_wrist_x_norm=_norm_x(keypoints, 9, x1, frame_width),
            right_wrist_x_norm=_norm_x(keypoints, 10, x1, frame_width),
            left_wrist_y_norm=_norm_y(keypoints, 9, y1, frame_height),
            right_wrist_y_norm=_norm_y(keypoints, 10, y1, frame_height),
            left_shoulder_y_norm=_norm_y(keypoints, 5, y1, frame_height),
            right_shoulder_y_norm=_norm_y(keypoints, 6, y1, frame_height),
            pose_confidence=float(np.mean(keypoints[[5, 6, 9, 10, 11, 12], 2])),
        )


def _norm_x(keypoints: np.ndarray, idx: int, offset_x: int, frame_width: int) -> float | None:
    if keypoints[idx, 2] < 0.2:
        return None
    return float((offset_x + keypoints[idx, 0]) / max(1, frame_width))


def _norm_y(keypoints: np.ndarray, idx: int, offset_y: int, frame_height: int) -> float | None:
    if keypoints[idx, 2] < 0.2:
        return None
    return float((offset_y + keypoints[idx, 1]) / max(1, frame_height))


def _avg_norm_y(keypoints: np.ndarray, indices: list[int], offset_x: int, offset_y: int, frame_height: int) -> float | None:
    values = [_norm_y(keypoints, idx, offset_y, frame_height) for idx in indices]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(sum(present) / len(present))

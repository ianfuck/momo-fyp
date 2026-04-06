from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO

from backend.device_utils import backend_label_for_device, get_vision_device


class PersonDetection(BaseModel):
    track_id: int
    bbox: list[int]
    bbox_area_ratio: float
    center_x_norm: float
    center_y_norm: float


class PersonDetector:
    def __init__(self, model_path: str, conf: float = 0.25) -> None:
        self.model_path = model_path
        self.conf = conf
        self.device = get_vision_device()
        self.loaded = False
        self._model: YOLO | None = None

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
            self._model = YOLO(self.model_path)
            self.loaded = True
        return self._model

    def _inference_kwargs(self) -> dict:
        return {
            "device": self.device,
            "half": self.device.startswith("cuda"),
        }

    def warmup(self) -> str:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self._ensure_model().predict(frame, conf=self.conf, verbose=False, **self._inference_kwargs())
        return backend_label_for_device(self.device)

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        model = self._ensure_model()
        results = model.track(frame, persist=True, classes=[0], conf=self.conf, verbose=False, **self._inference_kwargs())
        detections: list[PersonDetection] = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None:
            return detections

        ids = result.boxes.id.tolist() if result.boxes.id is not None else [None] * len(result.boxes.xyxy)
        height, width = frame.shape[:2]
        frame_area = max(1.0, float(height * width))
        for box, track_id in zip(result.boxes.xyxy.tolist(), ids, strict=False):
            x1, y1, x2, y2 = [int(v) for v in box]
            box_area = max(1.0, float((x2 - x1) * (y2 - y1)))
            detections.append(
                PersonDetection(
                    track_id=int(track_id) if track_id is not None else -1,
                    bbox=[x1, y1, x2, y2],
                    bbox_area_ratio=box_area / frame_area,
                    center_x_norm=((x1 + x2) / 2) / max(1, width),
                    center_y_norm=((y1 + y2) / 2) / max(1, height),
                )
            )
        return detections

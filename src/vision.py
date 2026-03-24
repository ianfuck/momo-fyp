"""Webcam + YOLO person detection."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.paths import PROJECT_ROOT
from src.state_machine import VisionSnapshot

logger = logging.getLogger(__name__)


def _yolo_torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VisionEngine:
    def __init__(self, weights_relative: str) -> None:
        self.weights_relative = weights_relative
        self.model: Any = None
        self._device = _yolo_torch_device()
        self._load_model()
        logger.info("YOLO 推理裝置：%s", self._device)

    def _load_model(self) -> None:
        wp = (PROJECT_ROOT / self.weights_relative).resolve()
        wp.parent.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO

        if wp.exists():
            self.model = YOLO(str(wp))
            return
        wp.parent.mkdir(parents=True, exist_ok=True)
        prev = os.getcwd()
        try:
            os.chdir(wp.parent)
            self.model = YOLO(wp.name)
        finally:
            os.chdir(prev)
        if not wp.exists():
            alt = wp.parent / wp.name
            if alt.exists():
                import shutil

                shutil.copy2(alt, wp)

    def reload_weights(self, weights_relative: str) -> None:
        self.weights_relative = weights_relative
        self._load_model()

    def detect(
        self,
        frame_bgr: np.ndarray,
        bbox_metric: str,
        frame_area: float,
    ) -> VisionSnapshot:
        h, w = frame_bgr.shape[:2]
        results = self.model(
            frame_bgr, verbose=False, classes=[0], device=self._device
        )
        best: tuple[int, int, int, int] | None = None
        best_area = 0.0

        if results and results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for b in boxes:
                x1, y1, x2, y2 = map(int, b[:4])
                bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                area = float(bw * bh)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, bw, bh)

        if best is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            return VisionSnapshot(False, 0.0, None, blur, [])

        x, y, bw, bh = best
        area = float(bw * bh)
        if bbox_metric == "area":
            metric = area
        elif bbox_metric == "height":
            metric = float(bh)
        else:
            metric = area / max(frame_area, 1.0)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        roi = gray[y : y + bh, x : x + bw]
        blur = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0

        tags = _heuristic_motion_tags(frame_bgr.shape, best)
        return VisionSnapshot(True, metric, best, blur, tags)


def _heuristic_motion_tags(shape: tuple[int, ...], bbox: tuple[int, int, int, int]) -> list[str]:
    return []

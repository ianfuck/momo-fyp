from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np

from backend.types import AudienceFeatures, RuntimeConfig, ServoTelemetry
from backend.vision.actions import MotionTracker
from backend.vision.face_eyes import FaceEyeTracker
from backend.vision.features import classify_body_shape, classify_colors, classify_distance, focus_score, smooth_color_labels
from backend.vision.pose_tracker import PoseTracker
from backend.vision.person_detector import PersonDetection, PersonDetector


@dataclass
class VisionState:
    features: AudienceFeatures
    servo: ServoTelemetry
    frame_jpeg: bytes | None
    frame_shape: tuple[int, int] | None
    target_seen_at: float | None


class VisionRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.detector = PersonDetector(config.yolo_model_path)
        self.eyes = FaceEyeTracker()
        self.pose = PoseTracker(config.yolo_pose_model_path)
        self.motion = MotionTracker()
        self.top_color_history: deque[str] = deque(maxlen=6)
        self.bottom_color_history: deque[str] = deque(maxlen=6)
        self.capture: cv2.VideoCapture | None = None
        self.thread: threading.Thread | None = None
        self.running = False
        self.lock = threading.Lock()
        self.failed_open_count = 0
        self.camera_disabled = False
        self.external_frame_at: float | None = None
        self.latest_state = VisionState(
            features=AudienceFeatures(),
            servo=ServoTelemetry(),
            frame_jpeg=None,
            frame_shape=None,
            target_seen_at=None,
        )

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.capture:
            self.capture.release()
            self.capture = None

    def reconfigure(self, config: RuntimeConfig) -> None:
        self.config = config
        self.detector = PersonDetector(config.yolo_model_path)
        self.pose = PoseTracker(config.yolo_pose_model_path)
        self.failed_open_count = 0
        self.camera_disabled = False
        self.top_color_history.clear()
        self.bottom_color_history.clear()
        self.stop()
        self.start()

    def get_snapshot(self) -> VisionState:
        with self.lock:
            return VisionState(
                features=self.latest_state.features.model_copy(deep=True),
                servo=self.latest_state.servo.model_copy(deep=True),
                frame_jpeg=self.latest_state.frame_jpeg,
                frame_shape=self.latest_state.frame_shape,
                target_seen_at=self.latest_state.target_seen_at,
            )

    def list_cameras(self) -> list[dict]:
        if self.config.camera_source == "browser":
            return [
                {
                    "device_id": self.config.camera_device_id,
                    "device_name": "Browser Camera",
                    "modes": [{"width": self.config.camera_width, "height": self.config.camera_height, "fps": self.config.camera_fps}],
                }
            ]
        cameras: list[dict] = []
        for index in range(5):
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                continue
            modes = []
            for width, height, fps in [(640, 480, 30), (1280, 720, 30), (1920, 1080, 30), (1280, 720, 60)]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                got_fps = int(round(cap.get(cv2.CAP_PROP_FPS) or fps))
                mode = {"width": got_w, "height": got_h, "fps": got_fps}
                if mode not in modes:
                    modes.append(mode)
            cameras.append({"device_id": str(index), "device_name": f"Camera {index}", "modes": modes})
            cap.release()
        if not cameras:
            cameras.append(
                {
                    "device_id": "0",
                    "device_name": "Default Camera",
                    "modes": [{"width": self.config.camera_width, "height": self.config.camera_height, "fps": self.config.camera_fps}],
                }
            )
        return cameras

    def _open_capture(self) -> cv2.VideoCapture:
        os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
        device_index = 0 if self.config.camera_device_id == "default" else int(self.config.camera_device_id)
        capture = cv2.VideoCapture(device_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        capture.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        return capture

    def _loop(self) -> None:
        while self.running:
            if self.config.camera_source == "browser":
                if self.capture:
                    self.capture.release()
                    self.capture = None
                if self.external_frame_at and time.monotonic() - self.external_frame_at < 2.0:
                    time.sleep(0.2)
                    continue
                time.sleep(0.5)
                continue
            if self.camera_disabled:
                time.sleep(5)
                continue
            if not self.capture:
                self.capture = self._open_capture()
            if not self.capture.isOpened():
                self.failed_open_count += 1
                if self.failed_open_count >= 3:
                    self.camera_disabled = True
                    time.sleep(5)
                    continue
                time.sleep(2)
                self.capture = self._open_capture()
                continue
            self.failed_open_count = 0
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            features, servo = self._process_frame(frame)
            annotated = self._annotate(frame.copy(), features, servo)
            ok_jpg, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self.lock:
                self.latest_state = VisionState(
                    features=features,
                    servo=servo,
                    frame_jpeg=encoded.tobytes() if ok_jpg else None,
                    frame_shape=(frame.shape[1], frame.shape[0]),
                    target_seen_at=time.monotonic() if features.track_id is not None else self.latest_state.target_seen_at,
                )
            time.sleep(max(0.0, 1.0 / max(1, self.config.camera_fps) * 0.5))

    def submit_jpeg_frame(self, jpeg_bytes: bytes) -> VisionState:
        array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("invalid jpeg frame")
        features, servo = self._process_frame(frame)
        annotated = self._annotate(frame.copy(), features, servo)
        ok_jpg, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        state = VisionState(
            features=features,
            servo=servo,
            frame_jpeg=encoded.tobytes() if ok_jpg else jpeg_bytes,
            frame_shape=(frame.shape[1], frame.shape[0]),
            target_seen_at=time.monotonic() if features.track_id is not None else self.latest_state.target_seen_at,
        )
        with self.lock:
            self.latest_state = state
            self.external_frame_at = time.monotonic()
        return state

    def _process_frame(self, frame: np.ndarray) -> tuple[AudienceFeatures, ServoTelemetry]:
        detections = self.detector.detect(frame)
        if not detections:
            self.top_color_history.clear()
            self.bottom_color_history.clear()
            return AudienceFeatures(), ServoTelemetry()
        person = max(detections, key=lambda item: item.bbox_area_ratio)
        pose = self.pose.detect(frame, person.bbox)
        face = self.eyes.locate(frame, person.bbox, person.center_x_norm)
        top_color, bottom_color = classify_colors(
            frame,
            person.bbox,
            face.face_bbox,
            {
                "left_shoulder": pose.left_shoulder_point,
                "right_shoulder": pose.right_shoulder_point,
                "left_hip": pose.left_hip_point,
                "right_hip": pose.right_hip_point,
            },
        )
        self.top_color_history.append(top_color)
        self.bottom_color_history.append(bottom_color)
        top_color = smooth_color_labels(list(self.top_color_history), top_color)
        bottom_color = smooth_color_labels(list(self.bottom_color_history), bottom_color)
        height_class, build_class = classify_body_shape(person.bbox, frame.shape)
        focus = focus_score(frame, face.face_bbox or person.bbox)
        distance = classify_distance(
            person.bbox_area_ratio,
            self.config.lock_bbox_threshold_ratio,
            self.config.defocus_bbox_threshold_ratio,
        )
        actions = self.motion.update(
            area_ratio=person.bbox_area_ratio,
            center_y_norm=person.center_y_norm,
            eye_x_norm=face.eye_midpoint[0] if face.eye_midpoint else person.center_x_norm,
            pose=pose,
            focus_score=focus,
            defocus_threshold=self.config.defocus_bbox_threshold_ratio,
            focus_score_threshold=self.config.focus_score_threshold,
            crouch_delta_threshold=self.config.crouch_delta_threshold,
        )
        features = AudienceFeatures(
            track_id=person.track_id if person.track_id >= 0 else 1,
            person_bbox=person.bbox,
            bbox_area_ratio=person.bbox_area_ratio,
            center_x_norm=person.center_x_norm,
            center_y_norm=person.center_y_norm,
            distance_class=distance,
            height_class=height_class,
            build_class=build_class,
            top_color=top_color,
            bottom_color=bottom_color,
            focus_score=focus,
            face_bbox=face.face_bbox,
            left_eye_bbox=face.left_eye_bbox,
            right_eye_bbox=face.right_eye_bbox,
            eye_midpoint=face.eye_midpoint,
            eye_confidence=face.eye_confidence,
            pose_keypoints={
                "nose": pose.nose_point,
                "left_shoulder": pose.left_shoulder_point,
                "right_shoulder": pose.right_shoulder_point,
                "left_elbow": pose.left_elbow_point,
                "right_elbow": pose.right_elbow_point,
                "left_wrist": pose.left_wrist_point,
                "right_wrist": pose.right_wrist_point,
                "left_hip": pose.left_hip_point,
                "right_hip": pose.right_hip_point,
            },
            left_wrist_point=[round(pose.left_wrist_x_norm, 4), round(pose.left_wrist_y_norm, 4)]
            if pose.left_wrist_x_norm is not None and pose.left_wrist_y_norm is not None
            else None,
            right_wrist_point=[round(pose.right_wrist_x_norm, 4), round(pose.right_wrist_y_norm, 4)]
            if pose.right_wrist_x_norm is not None and pose.right_wrist_y_norm is not None
            else None,
            pose_confidence=pose.pose_confidence,
            actions=actions,
        )
        servo = ServoTelemetry(tracking_source=face.tracking_source)
        return features, servo

    def _annotate(self, frame: np.ndarray, features: AudienceFeatures, servo: ServoTelemetry) -> np.ndarray:
        height, width = frame.shape[:2]
        if features.track_id is not None:
            if features.face_bbox:
                self._draw_box(frame, features.face_bbox, (97, 226, 148), "Face")
            if features.left_eye_bbox:
                self._draw_box(frame, features.left_eye_bbox, (88, 166, 255), "L Eye")
            if features.right_eye_bbox:
                self._draw_box(frame, features.right_eye_bbox, (255, 122, 162), "R Eye")
            if features.eye_midpoint:
                x = int(features.eye_midpoint[0] * width)
                y = int(features.eye_midpoint[1] * height)
                cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
            if features.left_wrist_point:
                self._draw_point(frame, features.left_wrist_point, width, height, (255, 210, 87), "L Wrist")
            if features.right_wrist_point:
                self._draw_point(frame, features.right_wrist_point, width, height, (255, 148, 87), "R Wrist")
        cv2.putText(
            frame,
            f"track={features.track_id} dist={features.distance_class} focus={features.focus_score:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return frame

    def _draw_box(self, frame: np.ndarray, bbox: list[int], color: tuple[int, int, int], label: str) -> None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_point(
        self,
        frame: np.ndarray,
        point: list[float],
        width: int,
        height: int,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        x = int(point[0] * width)
        y = int(point[1] * height)
        cv2.circle(frame, (x, y), 7, color, -1)
        cv2.putText(frame, label, (x + 10, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

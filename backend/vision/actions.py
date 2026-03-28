from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from backend.types import ActionFlags
from backend.vision.pose_tracker import PoseSignals


@dataclass
class MotionTracker:
    area_history: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    center_y_history: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    shoulder_y_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    left_wrist_x_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    right_wrist_x_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))

    def update(
        self,
        area_ratio: float,
        center_y_norm: float,
        eye_x_norm: float,
        pose: PoseSignals | None,
        focus_score: float,
        defocus_threshold: float,
        focus_score_threshold: float,
        crouch_delta_threshold: float,
    ) -> ActionFlags:
        self.area_history.append(area_ratio)
        self.center_y_history.append(center_y_norm)
        actions = ActionFlags()

        if len(self.area_history) >= 2:
            delta = self.area_history[-1] - self.area_history[0]
            actions.approaching = delta > 0.08
            actions.moving_away = delta < -0.08

        if len(self.center_y_history) >= 2:
            delta_y = self.center_y_history[-1] - min(self.center_y_history)
            actions.crouch = delta_y > crouch_delta_threshold

        if pose and pose.shoulder_y_norm is not None:
            self.shoulder_y_history.append(pose.shoulder_y_norm)
        if pose and pose.left_wrist_x_norm is not None:
            self.left_wrist_x_history.append(pose.left_wrist_x_norm)
        if pose and pose.right_wrist_x_norm is not None:
            self.right_wrist_x_history.append(pose.right_wrist_x_norm)

        if pose and _has_wave_motion(self.left_wrist_x_history, pose.left_wrist_y_norm, pose.left_shoulder_y_norm):
            actions.wave = True
        if pose and _has_wave_motion(self.right_wrist_x_history, pose.right_wrist_y_norm, pose.right_shoulder_y_norm):
            actions.wave = True

        if pose and pose.shoulder_y_norm is not None and len(self.shoulder_y_history) >= 4:
            shoulder_delta = self.shoulder_y_history[-1] - min(self.shoulder_y_history)
            actions.crouch = actions.crouch or shoulder_delta > crouch_delta_threshold * 0.7

        actions.defocus = focus_score < focus_score_threshold
        return actions


def _has_wave_motion(history: deque[float], wrist_y: float | None, shoulder_y: float | None) -> bool:
    if wrist_y is None or shoulder_y is None:
        return False
    if wrist_y > shoulder_y + 0.04:
        return False
    if len(history) < 5:
        return False
    return max(history) - min(history) > 0.12

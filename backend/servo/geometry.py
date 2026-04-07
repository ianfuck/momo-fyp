from __future__ import annotations

import math

from backend.types import ServoTelemetry


def compute_servo_angles(
    eye_midpoint_x_norm: float,
    bbox_area_ratio: float,
    left_zero_deg: float,
    right_zero_deg: float,
    eye_spacing_cm: int,
    left_limits: tuple[float, float],
    right_limits: tuple[float, float],
) -> ServoTelemetry:
    z_est = max(10.0, 120.0 - (bbox_area_ratio * 100.0))
    x_est = (eye_midpoint_x_norm - 0.5) * 60.0
    half_eye_spacing = eye_spacing_cm / 2.0
    left_delta = math.degrees(math.atan2(x_est + half_eye_spacing, z_est))
    right_delta = math.degrees(math.atan2(x_est - half_eye_spacing, z_est))
    left = min(max(left_zero_deg + left_delta, left_limits[0]), left_limits[1])
    right = min(max(right_zero_deg + right_delta, right_limits[0]), right_limits[1])
    return ServoTelemetry(left_deg=round(left, 2), right_deg=round(right, 2), tracking_source="eye_midpoint")

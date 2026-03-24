"""Binocular-ish gaze: two servo angles from single camera + IPD offset."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import RuntimeConfig


def compute_eye_angles(
    cfg: RuntimeConfig,
    frame_w: int,
    frame_h: int,
    bbox: tuple[int, int, int, int] | None,
) -> tuple[float, float]:
    """Return (left_deg, right_deg) for SG90, neutral-centered."""
    cx = frame_w / 2
    cy = frame_h / 2
    f = max(cfg.focal_length_px, 1e-3)

    if bbox is None:
        n = cfg.servo_left_neutral_deg
        m = cfg.servo_right_neutral_deg
        return _clamp_pair(cfg, n, m)

    x, y, bw, bh = bbox
    u = x + bw / 2.0
    v = y + bh / 2.0

    theta_x = math.degrees(math.atan2(u - cx, f))
    theta_y = math.degrees(math.atan2(v - cy, f))

    ipd_m = cfg.ipd_cm / 100.0
    eye_h_m = cfg.camera_to_eye_cm / 100.0
    dist_proxy = max(0.5, (f * 1.7) / max(bh, 1.0))

    vergence_deg = math.degrees(math.atan2(ipd_m / 2.0, max(dist_proxy, 0.1)))
    left_pan = theta_x - vergence_deg * 0.5
    right_pan = theta_x + vergence_deg * 0.5
    tilt = theta_y * 0.4

    left = cfg.servo_left_neutral_deg + left_pan + tilt * 0.15
    right = cfg.servo_right_neutral_deg + right_pan + tilt * 0.15

    if cfg.servo_left_invert:
        left = 2 * cfg.servo_left_neutral_deg - left
    if cfg.servo_right_invert:
        right = 2 * cfg.servo_right_neutral_deg - right

    return _clamp_pair(cfg, left, right)


def _clamp_pair(cfg: RuntimeConfig, l: float, r: float) -> tuple[float, float]:
    l = max(cfg.servo_left_min_deg, min(cfg.servo_left_max_deg, l))
    r = max(cfg.servo_right_min_deg, min(cfg.servo_right_max_deg, r))
    return l, r

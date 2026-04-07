from backend.servo.geometry import compute_servo_angles


def test_servo_angles_diverge_for_left_and_right_eyes():
    telemetry = compute_servo_angles(
        eye_midpoint_x_norm=0.7,
        bbox_area_ratio=0.2,
        left_zero_deg=90,
        right_zero_deg=90,
        eye_spacing_cm=10,
        left_limits=(45, 135),
        right_limits=(45, 135),
    )
    assert telemetry.left_deg != telemetry.right_deg
    assert 45 <= telemetry.left_deg <= 135
    assert 45 <= telemetry.right_deg <= 135


def test_servo_eye_spacing_changes_divergence():
    narrow = compute_servo_angles(
        eye_midpoint_x_norm=0.7,
        bbox_area_ratio=0.2,
        left_zero_deg=90,
        right_zero_deg=90,
        eye_spacing_cm=6,
        left_limits=(45, 135),
        right_limits=(45, 135),
    )
    wide = compute_servo_angles(
        eye_midpoint_x_norm=0.7,
        bbox_area_ratio=0.2,
        left_zero_deg=90,
        right_zero_deg=90,
        eye_spacing_cm=14,
        left_limits=(45, 135),
        right_limits=(45, 135),
    )

    assert (wide.left_deg - wide.right_deg) > (narrow.left_deg - narrow.right_deg)

from backend.vision.actions import MotionTracker
from backend.vision.pose_tracker import PoseSignals


def test_wave_detection_prefers_pose_wrist_motion():
    tracker = MotionTracker()
    action = None
    for wrist_x in [0.35, 0.48, 0.33, 0.5, 0.34, 0.49]:
        action = tracker.update(
            area_ratio=0.2,
            center_y_norm=0.5,
            eye_x_norm=0.5,
            pose=PoseSignals(
                left_wrist_x_norm=wrist_x,
                left_wrist_y_norm=0.22,
                left_shoulder_y_norm=0.32,
                shoulder_y_norm=0.32,
                pose_confidence=0.8,
            ),
            focus_score=0.9,
            defocus_threshold=0.42,
            focus_score_threshold=0.25,
            crouch_delta_threshold=0.18,
        )
    assert action is not None
    assert action.wave is True


def test_crouch_detection_uses_pose_height_change():
    tracker = MotionTracker()
    action = None
    for shoulder_y in [0.3, 0.31, 0.33, 0.39, 0.45]:
        action = tracker.update(
            area_ratio=0.2,
            center_y_norm=0.5,
            eye_x_norm=0.5,
            pose=PoseSignals(shoulder_y_norm=shoulder_y, pose_confidence=0.8),
            focus_score=0.9,
            defocus_threshold=0.42,
            focus_score_threshold=0.25,
            crouch_delta_threshold=0.18,
        )
    assert action is not None
    assert action.crouch is True


def test_head_motion_alone_does_not_trigger_wave():
    tracker = MotionTracker()
    action = None
    for eye_x in [0.41, 0.57, 0.43, 0.58, 0.42, 0.56]:
        action = tracker.update(
            area_ratio=0.2,
            center_y_norm=0.5,
            eye_x_norm=eye_x,
            pose=PoseSignals(shoulder_y_norm=0.32, pose_confidence=0.8),
            focus_score=0.9,
            defocus_threshold=0.42,
            focus_score_threshold=0.25,
            crouch_delta_threshold=0.18,
        )
    assert action is not None
    assert action.wave is False

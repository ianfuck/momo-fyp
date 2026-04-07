from __future__ import annotations

import platform
from pathlib import Path

from pydantic import ValidationError

from backend.types import ConfigField, RuntimeConfig


TRACKING_EXAMPLES = [
    "resource/example/track-example-1.csv",
    "resource/example/track-example-2.csv",
    "resource/example/track-example-3.csv",
]
IDLE_EXAMPLES = ["resource/example/idle-sentences.csv"]

FIELD_DESCRIPTIONS: dict[str, tuple[str, str, str | None]] = {
    "camera_source": ("Camera Source", "Choose browser-uploaded frames or backend OpenCV capture.", None),
    "camera_device_id": ("Camera", "Camera device identifier.", None),
    "camera_width": ("Width", "Requested camera capture width.", ">=320"),
    "camera_height": ("Height", "Requested camera capture height.", ">=240"),
    "camera_fps": ("FPS", "Requested camera frame rate.", "1-60"),
    "camera_mirror_preview": ("Mirror Preview", "Mirror the preview output in UI.", None),
    "yolo_model_path": ("YOLO Model Path", "YOLO person detection model path.", None),
    "yolo_pose_model_path": ("YOLO Pose Model Path", "YOLO pose model path.", None),
    "yolo_device_mode": ("YOLO Device", "Device mode for person detect and pose: auto, cpu, or accelerator for this OS.", None),
    "lock_bbox_threshold_ratio": ("Lock Threshold", "Person bbox area ratio required to enter lock mode.", "0.01-0.95"),
    "unlock_bbox_threshold_ratio": ("Unlock Threshold", "Person bbox area ratio required to remain locked.", "0.01-0.95"),
    "enter_debounce_ms": ("Enter Debounce", "Continuous time above threshold before locking.", ">=0"),
    "exit_debounce_ms": ("Exit Debounce", "Continuous time below threshold before reconnecting.", ">=0"),
    "lost_timeout_ms": ("Lost Timeout", "Time window to reconnect the same audience.", ">=0"),
    "eye_loss_timeout_ms": ("Eye Loss Timeout", "Fallback delay before eye tracking reverts to person center.", ">=0"),
    "defocus_bbox_threshold_ratio": ("Defocus Threshold", "Area ratio considered too close and likely out of focus.", "0.01-0.99"),
    "focus_score_threshold": ("Focus Score", "Minimum focus score to treat face ROI as in focus.", "0-1"),
    "feature_match_threshold": ("Feature Match", "Similarity threshold for reconnecting to the same audience.", "0-1"),
    "wave_detection_enabled": ("Wave Detection", "Enable wave detection heuristics.", None),
    "crouch_detection_enabled": ("Crouch Detection", "Enable crouch detection heuristics.", None),
    "wave_window_ms": ("Wave Window", "Motion window used by wave detection.", ">=0"),
    "crouch_delta_threshold": ("Crouch Delta", "Relative vertical change threshold for crouch detection.", "0-1"),
    "ollama_base_url": ("Ollama URL", "Base URL for Ollama HTTP API.", None),
    "ollama_model": ("Ollama Model", "Model name sent to Ollama generate API.", None),
    "ollama_device_mode": ("Ollama Device", "Preferred Ollama execution mode: auto, cpu, or accelerator for this OS.", None),
    "llm_use_person_crop": ("LLM Person Crop Mode", "Send person crop image to LLM and switch tracking prompt to image-guided audience observation.", None),
    "ollama_timeout_sec": ("Ollama Timeout", "Maximum streaming generation timeout in seconds.", "1-3600"),
    "ollama_max_retries": ("Ollama Retries", "Retry count after timeout or stream failure.", "0-10"),
    "tracking_examples_selected": ("Tracking Examples", "CSV examples used for stage-aligned tracking prompt generation.", None),
    "idle_examples_selected": ("Idle Examples", "CSV examples used for idle prompt generation.", None),
    "history_max_sentences": ("History Size", "History rollover limit. Fixed at 10 for MVP.", "10"),
    "tts_model_path": ("TTS Model Path", "Local Fish Audio S1 Mini model path.", None),
    "tts_device_mode": ("TTS Device", "Device mode for Fish TTS: auto, cpu, or accelerator for this OS.", None),
    "tts_emotion_enabled": ("TTS Emotion", "Let Ollama choose and apply an emotion tag for Fish TTS.", None),
    "tts_clone_voice_enabled": ("Clone Voice", "Use reference audio/text for Fish voice cloning. Turn off for normal TTS.", None),
    "tts_ref_audio_path": ("TTS Ref Audio", "Reference audio used for voice clone prompt.", None),
    "tts_ref_text_path": ("TTS Ref Transcript", "Transcript paired with the reference audio.", None),
    "tts_timeout_sec": ("TTS Timeout", "Maximum time allowed for TTS warmup or synthesis.", "1-3600"),
    "tts_output_volume": ("TTS Volume", "Playback volume multiplier.", "0-2"),
    "audio_output_device": ("Audio Output", "Audio output device id used for playback.", None),
    "tts_autoplay": ("TTS Autoplay", "Automatically play synthesized audio.", None),
    "tts_retry_count": ("TTS Retries", "Retry count after synthesis failure.", "0-10"),
    "serial_port": ("Serial Port", "Serial device path or auto detection.", None),
    "serial_baud_rate": ("Baud Rate", "UART speed for ESP32 serial communication.", ">=1200"),
    "servo_left_zero_deg": ("Left Zero", "Neutral angle for the left eye servo.", "0-180"),
    "servo_right_zero_deg": ("Right Zero", "Neutral angle for the right eye servo.", "0-180"),
    "servo_left_min_deg": ("Left Min", "Left servo lower clamp.", "0-180"),
    "servo_left_max_deg": ("Left Max", "Left servo upper clamp.", "0-180"),
    "servo_right_min_deg": ("Right Min", "Right servo lower clamp.", "0-180"),
    "servo_right_max_deg": ("Right Max", "Right servo upper clamp.", "0-180"),
    "servo_smoothing_alpha": ("Servo Smoothing", "One-pole smoothing factor for servo motion.", "0-1"),
    "servo_max_speed_deg_per_sec": ("Servo Max Speed", "Servo speed cap.", ">=1"),
    "idle_sentence_interval_ms": ("Idle Interval", "Interval between idle utterances.", ">=0"),
    "idle_to_sleep_ms": ("Idle To Sleep", "Idle duration before sleep mode.", ">=0"),
    "sleep_duration_ms": ("Sleep Duration", "Sleep mode duration.", ">=0"),
    "purge_shake_duration_ms": ("Purge Shake", "Duration of purge shake motion.", ">=0"),
    "purge_cooldown_ms": ("Purge Cooldown", "Dead time after sentence 10.", ">=0"),
}

FIELD_GROUPS: dict[str, str] = {
    "camera_source": "camera",
    "camera_device_id": "camera",
    "camera_width": "camera",
    "camera_height": "camera",
    "camera_fps": "camera",
    "camera_mirror_preview": "camera",
    "yolo_model_path": "vision",
    "yolo_pose_model_path": "vision",
    "yolo_device_mode": "vision",
    "lock_bbox_threshold_ratio": "vision",
    "unlock_bbox_threshold_ratio": "vision",
    "enter_debounce_ms": "vision",
    "exit_debounce_ms": "vision",
    "lost_timeout_ms": "vision",
    "eye_loss_timeout_ms": "vision",
    "defocus_bbox_threshold_ratio": "vision",
    "focus_score_threshold": "vision",
    "feature_match_threshold": "vision",
    "wave_detection_enabled": "vision",
    "crouch_detection_enabled": "vision",
    "wave_window_ms": "vision",
    "crouch_delta_threshold": "vision",
    "ollama_base_url": "llm",
    "ollama_model": "llm",
    "ollama_device_mode": "llm",
    "llm_use_person_crop": "llm",
    "ollama_timeout_sec": "llm",
    "ollama_max_retries": "llm",
    "tracking_examples_selected": "prompting",
    "idle_examples_selected": "prompting",
    "history_max_sentences": "prompting",
    "tts_model_path": "tts",
    "tts_device_mode": "tts",
    "tts_emotion_enabled": "tts",
    "tts_clone_voice_enabled": "tts",
    "tts_ref_audio_path": "tts",
    "tts_ref_text_path": "tts",
    "tts_timeout_sec": "tts",
    "tts_output_volume": "tts",
    "audio_output_device": "tts",
    "tts_autoplay": "tts",
    "tts_retry_count": "tts",
    "serial_port": "serial",
    "serial_baud_rate": "serial",
    "servo_left_zero_deg": "servo",
    "servo_right_zero_deg": "servo",
    "servo_left_min_deg": "servo",
    "servo_left_max_deg": "servo",
    "servo_right_min_deg": "servo",
    "servo_right_max_deg": "servo",
    "servo_smoothing_alpha": "servo",
    "servo_max_speed_deg_per_sec": "servo",
    "idle_sentence_interval_ms": "idle",
    "idle_to_sleep_ms": "idle",
    "sleep_duration_ms": "idle",
    "purge_shake_duration_ms": "purge",
    "purge_cooldown_ms": "purge",
}


def build_field_catalog(config: RuntimeConfig) -> list[ConfigField]:
    fields: list[ConfigField] = []
    defaults = RuntimeConfig()
    for key, field_info in RuntimeConfig.model_fields.items():
        label, description, valid_range = FIELD_DESCRIPTIONS.get(
            key,
            (key.replace("_", " ").title(), f"Runtime config for {key}.", None),
        )
        value = getattr(config, key)
        if key == "unlock_bbox_threshold_ratio" and value is None:
            value = config.lock_bbox_threshold_ratio
        default = getattr(defaults, key)
        field_type = _infer_type(value if value is not None else default)
        fields.append(
            ConfigField(
                key=key,
                label=label,
                description=description,
                type=field_type,
                default=default,
                value=value,
                valid_range=valid_range,
                enum=_enum_for_field(key),
                applies_to=FIELD_GROUPS.get(key, "general"),
            )
        )
    return fields


def _infer_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "string[]"
    return "string"


def _enum_for_field(key: str) -> list[str] | None:
    if key == "camera_source":
        return ["browser", "backend"]
    if key in {"yolo_device_mode", "tts_device_mode", "ollama_device_mode"}:
        accelerator = "mps" if platform.system() == "Darwin" else "gpu"
        return ["auto", "cpu", accelerator]
    return None


def validate_runtime_config(candidate: RuntimeConfig) -> list[str]:
    errors: list[str] = []
    if candidate.camera_width < 320:
        errors.append("camera_width must be >= 320")
    if candidate.camera_height < 240:
        errors.append("camera_height must be >= 240")
    if not 1 <= candidate.camera_fps <= 60:
        errors.append("camera_fps must be between 1 and 60")
    if not 0.01 <= candidate.lock_bbox_threshold_ratio <= 0.95:
        errors.append("lock_bbox_threshold_ratio must be between 0.01 and 0.95")
    unlock = (
        candidate.unlock_bbox_threshold_ratio
        if candidate.unlock_bbox_threshold_ratio is not None
        else candidate.lock_bbox_threshold_ratio
    )
    if not 0.01 <= unlock <= 0.95:
        errors.append("unlock_bbox_threshold_ratio must be between 0.01 and 0.95")
    if candidate.ollama_timeout_sec < 1:
        errors.append("ollama_timeout_sec must be >= 1")
    if candidate.tts_timeout_sec < 1:
        errors.append("tts_timeout_sec must be >= 1")
    accelerator = "mps" if platform.system() == "Darwin" else "gpu"
    allowed_device_modes = {"auto", "cpu", accelerator}
    if candidate.yolo_device_mode not in allowed_device_modes:
        errors.append(f"yolo_device_mode must be one of {sorted(allowed_device_modes)}")
    if candidate.tts_device_mode not in allowed_device_modes:
        errors.append(f"tts_device_mode must be one of {sorted(allowed_device_modes)}")
    if candidate.ollama_device_mode not in allowed_device_modes:
        errors.append(f"ollama_device_mode must be one of {sorted(allowed_device_modes)}")
    if candidate.history_max_sentences != 10:
        errors.append("history_max_sentences must be fixed at 10 for MVP")
    for path in candidate.tracking_examples_selected + candidate.idle_examples_selected:
        if not Path(path).exists():
            errors.append(f"example file not found: {path}")
    return errors


def merge_config(current: RuntimeConfig, payload: dict) -> RuntimeConfig:
    try:
        merged = current.model_copy(update=payload)
        return RuntimeConfig.model_validate(merged.model_dump())
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

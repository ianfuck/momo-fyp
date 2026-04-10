from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SystemMode(str, Enum):
    IDLE = "IDLE"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    RECONNECTING = "RECONNECTING"
    SLEEP = "SLEEP"
    PURGE_COOLDOWN = "PURGE_COOLDOWN"


class PipelineStage(str, Enum):
    IDLE = "IDLE"
    LLM = "LLM"
    TTS = "TTS"
    PLAYBACK = "PLAYBACK"
    ERROR = "ERROR"


class ActionFlags(BaseModel):
    wave: bool = False
    crouch: bool = False
    defocus: bool = False
    moving_away: bool = False
    approaching: bool = False
    returned_after_defocus: bool = False


class AudienceFeatures(BaseModel):
    track_id: int | None = None
    person_bbox: list[int] | None = None
    bbox_area_ratio: float = 0.0
    center_x_norm: float = 0.5
    center_y_norm: float = 0.5
    distance_class: str = "unknown"
    height_class: str = "unknown"
    build_class: str = "unknown"
    top_color: str = "unknown"
    bottom_color: str = "unknown"
    focus_score: float = 0.0
    face_bbox: list[int] | None = None
    left_eye_bbox: list[int] | None = None
    right_eye_bbox: list[int] | None = None
    eye_midpoint: list[float] | None = None
    eye_confidence: float = 0.0
    pose_keypoints: dict[str, list[float] | None] = Field(default_factory=dict)
    left_wrist_point: list[float] | None = None
    right_wrist_point: list[float] | None = None
    pose_confidence: float = 0.0
    actions: ActionFlags = Field(default_factory=ActionFlags)


class PipelineStatus(BaseModel):
    stage: PipelineStage = PipelineStage.IDLE
    started_at: str | None = None
    elapsed_ms: int = 0
    last_error: str | None = None


class ServoTelemetry(BaseModel):
    left_deg: float = 90.0
    right_deg: float = 90.0
    tracking_source: str = "none"


class SerialMonitorEntry(BaseModel):
    ts: str = Field(default_factory=utc_now_iso)
    direction: str
    message: str


class SerialMonitorSnapshot(BaseModel):
    port: str | None = None
    baud_rate: int | None = None
    last_tx: str | None = None
    last_tx_at: str | None = None
    last_rx: str | None = None
    last_rx_at: str | None = None
    last_error: str | None = None
    last_error_at: str | None = None
    entries: list[SerialMonitorEntry] = Field(default_factory=list)


class SystemStats(BaseModel):
    memory_rss_mb: float = 0.0
    memory_vms_mb: float = 0.0
    gpu_memory_mb: float | None = None
    temp_file_count: int = 0
    temp_file_size_mb: float = 0.0


class RuntimeComponentStats(BaseModel):
    requested_mode: str | None = None
    effective_device: str | None = None
    backend: str | None = None
    selection_source: str | None = None
    semantic_dispatch_mode: str | None = None
    precision_mode: str | None = None
    ram_mb: float | None = None
    vram_mb: float | None = None


class StatusSnapshot(BaseModel):
    ts: str = Field(default_factory=utc_now_iso)
    mode: SystemMode = SystemMode.IDLE
    pipeline: PipelineStatus = Field(default_factory=PipelineStatus)
    locked_track_id: int | None = None
    sentence_index: int = 0
    active_sentence_index: int = 0
    audience: AudienceFeatures = Field(default_factory=AudienceFeatures)
    servo: ServoTelemetry = Field(default_factory=ServoTelemetry)
    serial_monitor: SerialMonitorSnapshot = Field(default_factory=SerialMonitorSnapshot)
    stats: SystemStats = Field(default_factory=SystemStats)
    camera_device_id: str | None = None
    camera_mode: str | None = None
    serial_connected: bool = False
    ollama_connected: bool = False
    tts_loaded: bool = False
    generating_ms: int = 0
    llm_latency_ms: int | None = None
    tts_latency_ms: int | None = None
    playback_progress: float = 0.0
    yolo_detect_fps: float = 0.0
    tts_emotion_raw: str | None = None
    tts_emotion_applied: str | None = None
    tts_emotion_used: bool = False
    tts_input_text: str | None = None
    tts_reference_raw: str | None = None
    tts_reference_pair: str | None = None
    tts_reference_audio_path: str | None = None
    tts_reference_text_path: str | None = None
    current_prompt_system: str | None = None
    current_prompt_user: str | None = None
    last_llm_output: str | None = None
    last_spoken_text: str | None = None
    yolo_person_runtime: RuntimeComponentStats = Field(default_factory=RuntimeComponentStats)
    yolo_pose_runtime: RuntimeComponentStats = Field(default_factory=RuntimeComponentStats)
    tts_runtime: RuntimeComponentStats = Field(default_factory=RuntimeComponentStats)
    ollama_runtime: RuntimeComponentStats = Field(default_factory=RuntimeComponentStats)
    event_log: list[str] = Field(default_factory=list)


class ConfigField(BaseModel):
    key: str
    label: str
    description: str
    type: str
    default: Any
    value: Any
    valid_range: str | None = None
    enum: list[str] | None = None
    applies_to: str
    requires_restart: bool = False


class RuntimeConfig(BaseModel):
    camera_source: str = "browser"
    camera_device_id: str = "default"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_mirror_preview: bool = False
    camera_flip_vertical: bool = True
    yolo_model_path: str = "model/yolo/yolo26n.pt"
    yolo_pose_model_path: str = "model/yolo/yolo26n-pose.pt"
    yolo_device_mode: str = "auto"
    lock_bbox_threshold_ratio: float = 0.12
    unlock_bbox_threshold_ratio: float | None = None
    enter_debounce_ms: int = 1000
    exit_debounce_ms: int = 500
    lost_timeout_ms: int = 5000
    eye_loss_timeout_ms: int = 300
    defocus_bbox_threshold_ratio: float = 0.42
    focus_score_threshold: float = 0.25
    feature_match_threshold: float = 0.7
    wave_detection_enabled: bool = True
    crouch_detection_enabled: bool = True
    wave_window_ms: int = 600
    crouch_delta_threshold: float = 0.18
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:2b"
    ollama_device_mode: str = "auto"
    llm_use_person_crop: bool = True
    llm_liberation_mode: bool = False
    ollama_timeout_sec: int = 600
    ollama_max_retries: int = 1
    tracking_examples_selected: list[str] = Field(
        default_factory=lambda: [
            "resource/example/track-example-1.csv",
            "resource/example/track-example-2.csv",
            "resource/example/track-example-3.csv",
        ]
    )
    idle_examples_selected: list[str] = Field(
        default_factory=lambda: ["resource/example/idle-sentences.csv"]
    )
    history_max_sentences: int = 10
    tts_model_path: str = "model/huggingface/hf_snapshots/hexgrad__Kokoro-82M-v1.1-zh"
    tts_kokoro_voice: str = "zf_002"
    tts_device_mode: str = "auto"
    tts_emotion_enabled: bool = True
    tts_clone_voice_enabled: bool = True
    tts_reference_mode: str = "ollama_emotion"
    tts_ref_audio_path: str = "resource/voice/ref-voice3.wav"
    tts_ref_text_path: str = "resource/voice/transcript3.txt"
    tts_timeout_sec: int = 300
    tts_output_volume: float = 1.0
    tts_route_via_virtual_device: bool = False
    audio_output_device: str = "default"
    tts_autoplay: bool = True
    tts_retry_count: int = 1
    serial_port: str = "auto"
    serial_baud_rate: int = 115200
    servo_left_zero_deg: float = 87.0
    servo_right_zero_deg: float = 96.0
    servo_output_inverted: bool = False
    servo_left_trim_deg: float = 0.0
    servo_right_trim_deg: float = 0.0
    servo_left_gain: float = 2.5
    servo_right_gain: float = 2.5
    servo_eye_spacing_cm: int = 13
    servo_left_min_deg: float = 45.0
    servo_left_max_deg: float = 135.0
    servo_right_min_deg: float = 45.0
    servo_right_max_deg: float = 135.0
    led_min_brightness_pct: float = 0.0
    led_max_brightness_pct: float = 100.0
    led_brightness_output_inverted: bool = False
    led_left_right_inverted: bool = False
    servo_smoothing_alpha: float = 0.25
    servo_max_speed_deg_per_sec: float = 180.0
    idle_sentence_interval_ms: int = 15000
    idle_to_sleep_ms: int = 120000
    sleep_duration_ms: int = 30000
    purge_shake_duration_ms: int = 1500
    purge_cooldown_ms: int = 10000


class ConfigUpdateResponse(BaseModel):
    applied_config: RuntimeConfig
    validation_errors: list[str]
    effective_changes: list[str]
    apply_checks: list[dict[str, str]] = Field(default_factory=list)
    requires_pipeline_restart: bool

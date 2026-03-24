"""Runtime configuration with defaults matching the project plan."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.paths import PROJECT_ROOT


def _p(path: str) -> Path:
    return (PROJECT_ROOT / path).resolve()


class RuntimeConfig(BaseModel):
    camera_index: int = 0
    capture_target_fps: float = Field(default=15.0, ge=0.5, le=120.0)
    capture_frame_width: int = Field(default=640, ge=160, le=3840)
    capture_frame_height: int = Field(default=480, ge=120, le=2160)
    bbox_metric: str = Field(default="area", pattern="^(area|height|normalized_area)$")
    lock_on_threshold: float = 8000.0
    idle_below_threshold: float = 5000.0
    yolo_weights_path: str = "model/yolo/yolo11n.pt"

    debounce_lock_ms: int = 400
    debounce_lost_ms: int = 600

    post_lock_tts_delay_ms: int = 1000
    reconnect_window_ms: int = 5000
    idle_before_sleep_ms: int = 120_000
    sleep_duration_ms: int = 30_000
    purge_dead_ms: int = 10_000

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:0.8b"
    ollama_timeout_s: float = 600.0
    ollama_max_retries: int = 4
    ollama_warmup: bool = True
    ollama_pull_timeout_s: float = 900.0
    ollama_keep_alive: str = "86400m"
    startup_camera_delay_ms: int = 30_000

    persona_tracking_path: str = "resource/md/system-persona_tracking.md"
    persona_idle_path: str = "resource/md/system-persona_idle.md"
    few_shot_tracking_paths: list[str] = Field(
        default_factory=lambda: [
            "resource/example/track-example-1.csv",
            "resource/example/track-example-2.csv",
            "resource/example/track-example-3.csv",
        ]
    )
    few_shot_idle_paths: list[str] = Field(
        default_factory=lambda: ["resource/example/idle-sentences.csv"]
    )
    few_shot_max_rows_per_file: int = 30

    qwen_tts_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    tts_cache_dir: str = "model/huggingface"
    ref_audio_path: str = "resource/voice/ref-voice.m4a"
    ref_text_path: str = "resource/voice/transcript.txt"

    audio_backend: str = "auto"

    serial_port: str = ""
    serial_baud: int = 115200

    camera_to_eye_cm: float = 5.0
    ipd_cm: float = 10.0
    focal_length_px: float = 600.0

    servo_left_min_deg: float = 0.0
    servo_left_max_deg: float = 180.0
    servo_right_min_deg: float = 0.0
    servo_right_max_deg: float = 180.0
    servo_left_neutral_deg: float = 90.0
    servo_right_neutral_deg: float = 90.0
    servo_left_invert: bool = False
    servo_right_invert: bool = False

    blur_threshold: float = 50.0

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def strip_ollama_model(self) -> RuntimeConfig:
        self.ollama_model = (self.ollama_model or "").strip()
        self.ollama_keep_alive = (self.ollama_keep_alive or "5m").strip() or "5m"
        return self

    @field_validator("few_shot_tracking_paths", "few_shot_idle_paths")
    @classmethod
    def validate_example_paths(cls, v: list[str]) -> list[str]:
        for p in v:
            validate_example_path(p)
        return v


EXAMPLE_DIR = (PROJECT_ROOT / "resource" / "example").resolve()


def validate_example_path(relative: str) -> None:
    """Reject path traversal; must be under resource/example and .csv."""
    rp = (PROJECT_ROOT / relative).resolve()
    if not str(rp).startswith(str(EXAMPLE_DIR)):
        raise ValueError(f"Path not under resource/example: {relative}")
    if rp.suffix.lower() != ".csv":
        raise ValueError(f"Must be .csv: {relative}")


def merge_config(base: RuntimeConfig, patch: dict[str, Any]) -> RuntimeConfig:
    data = base.model_dump()
    for k, v in patch.items():
        if k in data or k in RuntimeConfig.model_fields:
            data[k] = v
    return RuntimeConfig(**{kk: data[kk] for kk in RuntimeConfig.model_fields if kk in data})


def build_config_schema() -> dict[str, Any]:
    """Shape for GET /api/config/schema (fields array)."""
    fields: list[dict[str, Any]] = []

    def add(
        key: str,
        typ: str,
        default: Any,
        help_zh: str,
        **extra: Any,
    ) -> None:
        item: dict[str, Any] = {
            "key": key,
            "type": typ,
            "default": default,
            "help_zh": help_zh,
        }
        item.update(extra)
        fields.append(item)

    d = RuntimeConfig()
    add("camera_index", "number", d.camera_index, "後端 OpenCV 相機索引（整數）。")
    add(
        "capture_target_fps",
        "number",
        d.capture_target_fps,
        "相機主迴圈目標影格率（每秒幾幀）；單幀若超過此週期（含 YOLO）則不額外 sleep，會以實際可達上限為準。",
    )
    add(
        "capture_frame_width",
        "number",
        d.capture_frame_width,
        "請求相機解析度寬度 (px)；套用後會重新開啟相機，實際尺寸以驅動回報為準（見狀態表實測）。",
    )
    add(
        "capture_frame_height",
        "number",
        d.capture_frame_height,
        "請求相機解析度高度 (px)；同上。",
    )
    add(
        "bbox_metric",
        "enum",
        d.bbox_metric,
        "與 threshold 比較的 bbox 度量。",
        enum_options=[
            {"value": "area", "label": "面積 (px²)", "help": "寬×高"},
            {"value": "height", "label": "高度 (px)", "help": "bbox 高度"},
            {"value": "normalized_area", "label": "正規化面積", "help": "面積 / 畫面面積"},
        ],
    )
    add("lock_on_threshold", "number", d.lock_on_threshold, "≥ 此值才允許進入鎖定（與 metric 同單位）。")
    add("idle_below_threshold", "number", d.idle_below_threshold, "低於則視為無有效觀眾候選。")
    add("yolo_weights_path", "string", d.yolo_weights_path, "YOLO 權重路徑；缺檔時自動下載到 model/。")
    add("debounce_lock_ms", "number", d.debounce_lock_ms, "連續滿足可鎖定的最短時間 (ms)。")
    add("debounce_lost_ms", "number", d.debounce_lost_ms, "連續低於閒置條件後才判離場 (ms)。")
    add("post_lock_tts_delay_ms", "number", d.post_lock_tts_delay_ms, "鎖定後延遲首次語音 (ms)。")
    add("reconnect_window_ms", "number", d.reconnect_window_ms, "短暫丟失視窗 foolproof (ms)。")
    add("idle_before_sleep_ms", "number", d.idle_before_sleep_ms, "閒置多久進入休眠 (ms)。")
    add("sleep_duration_ms", "number", d.sleep_duration_ms, "休眠靜音長度 (ms)。")
    add("purge_dead_ms", "number", d.purge_dead_ms, "第十句後裝死 (ms)。")
    add("ollama_base_url", "string", d.ollama_base_url, "Ollama HTTP 根 URL。")
    add(
        "ollama_model",
        "string",
        d.ollama_model,
        "Ollama 模型名稱，可直接輸入（例如 qwen3.5:0.8b）。若本機尚未下載，首次對話前會自動呼叫 Ollama pull。",
        placeholder="qwen3.5:0.8b",
        suggestions=[
            "qwen3.5:0.8b",
            "qwen2.5:0.5b",
            "qwen2.5:1.5b",
            "llama3.2",
            "llama3.2:1b",
        ],
    )
    add(
        "ollama_timeout_s",
        "number",
        d.ollama_timeout_s,
        "單次 LLM 呼叫的**總時限**（秒）：採**串流**讀取，避免非串流整包等待被誤判逾時；仍逾時會自動重試。",
    )
    add(
        "ollama_max_retries",
        "number",
        d.ollama_max_retries,
        "Ollama 連線/串流逾時或暫時錯誤時，最多再試幾次（含首次）。",
    )
    add(
        "ollama_warmup",
        "boolean",
        d.ollama_warmup,
        "每個行程對每個模型只做一次極短 generate，促進 Ollama 把權重載入記憶體，減少首句偶發逾時。",
    )
    add(
        "ollama_pull_timeout_s",
        "number",
        d.ollama_pull_timeout_s,
        "自動 pull 模型時的最長等待 (秒)；模型較大時請加大。",
    )
    add(
        "ollama_keep_alive",
        "string",
        d.ollama_keep_alive,
        "Ollama 請求 keep_alive（如 5m、30m、300s）；載入後模型在記憶體保留的時間，預設 5 分鐘。",
        placeholder="5m",
    )
    add(
        "startup_camera_delay_ms",
        "number",
        d.startup_camera_delay_ms,
        "服務啟動後延遲多久才開相機並進入 YOLO／狀態機／語音佇列 (ms)；期間會先背景預載 Ollama。",
    )
    add("persona_tracking_path", "string", d.persona_tracking_path, "鎖定模式 persona Markdown。")
    add("persona_idle_path", "string", d.persona_idle_path, "閒置模式 persona Markdown。")
    add(
        "few_shot_tracking_paths",
        "path_multiselect",
        d.few_shot_tracking_paths,
        "鎖定模式 Few-shot 用的 CSV（順序即拼接順序）；空則僅 persona。",
    )
    add(
        "few_shot_idle_paths",
        "path_multiselect",
        d.few_shot_idle_paths,
        "閒置模式 Few-shot CSV；空則僅 persona。",
    )
    add("few_shot_max_rows_per_file", "number", d.few_shot_max_rows_per_file, "每個 example CSV 最多餵給 LLM 的列數。")
    add("qwen_tts_model_id", "string", d.qwen_tts_model_id, "HuggingFace Qwen3-TTS 模型 id。")
    add("tts_cache_dir", "string", d.tts_cache_dir, "TTS 權重本機 cache 目錄。")
    add("ref_audio_path", "string", d.ref_audio_path, "語音克隆參考音檔。")
    add("ref_text_path", "string", d.ref_text_path, "參考音對應講稿（宜與音檔對齊）。")
    add(
        "audio_backend",
        "enum",
        d.audio_backend,
        "音訊播放後端；auto 依 OS 自動偵測。",
        enum_options=[
            {"value": "auto", "label": "自動", "help": "偵測 afplay / PowerShell / aplay / pygame"},
            {"value": "afplay", "label": "afplay (macOS)", "help": ""},
            {"value": "powershell", "label": "PowerShell (Windows)", "help": ""},
            {"value": "aplay", "label": "aplay (Linux)", "help": ""},
            {"value": "pygame", "label": "pygame", "help": "需安裝 pygame"},
        ],
    )
    add("serial_port", "string", d.serial_port, "Arduino 序列埠，空字串則不連線。")
    add("serial_baud", "number", d.serial_baud, "鮑率。")
    add("camera_to_eye_cm", "number", d.camera_to_eye_cm, "眼比鏡頭高 (cm)。")
    add("ipd_cm", "number", d.ipd_cm, "瞳距 (cm)。")
    add("focal_length_px", "number", d.focal_length_px, "焦距像素（距離代理用）。")
    add("servo_left_min_deg", "number", d.servo_left_min_deg, "左眼最小角 (deg)。")
    add("servo_left_max_deg", "number", d.servo_left_max_deg, "左眼最大角 (deg)。")
    add("servo_right_min_deg", "number", d.servo_right_min_deg, "右眼最小角 (deg)。")
    add("servo_right_max_deg", "number", d.servo_right_max_deg, "右眼最大角 (deg)。")
    add("servo_left_neutral_deg", "number", d.servo_left_neutral_deg, "左眼中立。")
    add("servo_right_neutral_deg", "number", d.servo_right_neutral_deg, "右眼中立。")
    add("servo_left_invert", "boolean", d.servo_left_invert, "左眼方向反轉。")
    add("servo_right_invert", "boolean", d.servo_right_invert, "右眼方向反轉。")
    add("blur_threshold", "number", d.blur_threshold, "Laplacian 變異低於此視為失焦候選。")

    return {"fields": fields}


def list_example_csvs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ex = PROJECT_ROOT / "resource" / "example"
    if not ex.is_dir():
        return out
    for f in sorted(ex.glob("*.csv")):
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        base = f.name.lower()
        if base.startswith("track-"):
            sm = "tracking"
        elif base.startswith("idle-"):
            sm = "idle"
        else:
            sm = "any"
        out.append(
            {
                "path": rel,
                "basename": f.name,
                "suggested_mode": sm,
            }
        )
    return out

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

from backend.types import (
    ActionFlags,
    AudienceFeatures,
    PipelineStage,
    PipelineStatus,
    ServoTelemetry,
    StatusSnapshot,
    SystemMode,
)


@dataclass
class RuntimeState:
    mode: SystemMode = SystemMode.IDLE
    locked_track_id: int | None = None
    sentence_index: int = 0
    active_sentence_index: int = 0
    pipeline: PipelineStatus = field(default_factory=PipelineStatus)
    audience: AudienceFeatures = field(default_factory=AudienceFeatures)
    servo: ServoTelemetry = field(default_factory=ServoTelemetry)
    event_log: list[str] = field(default_factory=list)
    llm_latency_ms: int | None = None
    tts_latency_ms: int | None = None
    tts_emotion_raw: str | None = None
    tts_emotion_applied: str | None = None
    tts_emotion_used: bool = False
    tts_input_text: str | None = None
    current_prompt_system: str | None = None
    current_prompt_user: str | None = None
    last_llm_output: str | None = None
    last_spoken_text: str | None = None
    _stage_started: float | None = None

    def set_mode(self, mode: SystemMode, note: str | None = None) -> None:
        self.mode = mode
        if note:
            self.event_log = [note, *self.event_log][:20]

    def set_pipeline_stage(self, stage: PipelineStage, error: str | None = None) -> None:
        self._stage_started = monotonic()
        self.pipeline = PipelineStatus(
            stage=stage,
            started_at=None,
            elapsed_ms=0,
            last_error=error,
        )

    def tick(self) -> None:
        if self._stage_started is not None:
            self.pipeline.elapsed_ms = int((monotonic() - self._stage_started) * 1000)

    def apply_detection(
        self,
        track_id: int,
        bbox_area_ratio: float,
        center_x_norm: float,
        top_color: str = "unknown",
        actions: ActionFlags | None = None,
    ) -> None:
        self.locked_track_id = track_id
        self.audience.track_id = track_id
        self.audience.bbox_area_ratio = bbox_area_ratio
        self.audience.center_x_norm = center_x_norm
        self.audience.top_color = top_color
        self.audience.actions = actions or ActionFlags()

    def snapshot(self) -> StatusSnapshot:
        self.tick()
        return StatusSnapshot(
            mode=self.mode,
            pipeline=self.pipeline,
            locked_track_id=self.locked_track_id,
            sentence_index=self.sentence_index,
            active_sentence_index=self.active_sentence_index,
            audience=self.audience,
            servo=self.servo,
            llm_latency_ms=self.llm_latency_ms,
            tts_latency_ms=self.tts_latency_ms,
            tts_emotion_raw=self.tts_emotion_raw,
            tts_emotion_applied=self.tts_emotion_applied,
            tts_emotion_used=self.tts_emotion_used,
            tts_input_text=self.tts_input_text,
            current_prompt_system=self.current_prompt_system,
            current_prompt_user=self.current_prompt_user,
            last_llm_output=self.last_llm_output,
            last_spoken_text=self.last_spoken_text,
            event_log=self.event_log,
        )

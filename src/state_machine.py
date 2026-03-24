"""State machine: bbox-primary thresholds, debounce, foolproof <5s / >5s."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import RuntimeConfig


class SMState(str, Enum):
    IDLE = "IDLE"
    LOCK = "LOCK"
    DEAD = "DEAD"
    SLEEP = "SLEEP"


@dataclass
class VisionSnapshot:
    has_person: bool
    metric_value: float
    bbox: tuple[int, int, int, int] | None
    blur_score: float
    behavior_tags: list[str] = field(default_factory=list)


@dataclass
class StateMachine:
    state: SMState = SMState.IDLE
    sentence_index: int = 0
    feature_cache: str = ""

    lock_debounce_ms: float = 0.0
    lost_debounce_ms: float = 0.0
    lost_ms: float = 0.0
    lost_display_ms: float = 0.0
    idle_accum_ms: float = 0.0
    sleep_remain_ms: float = 0.0
    dead_remain_ms: float = 0.0
    lock_elapsed_ms: float = 0.0

    suppress_tts: bool = False

    def reset_round(self) -> None:
        self.sentence_index = 0
        self.feature_cache = ""

    def tick(self, cfg: RuntimeConfig, snap: VisionSnapshot, dt_ms: float) -> None:
        self.suppress_tts = snap.blur_score < cfg.blur_threshold and snap.has_person

        in_lock_zone = (
            snap.has_person
            and snap.metric_value >= cfg.lock_on_threshold
            and snap.blur_score >= cfg.blur_threshold
        )
        in_idle_zone = (not snap.has_person) or (snap.metric_value < cfg.idle_below_threshold)

        if self.state == SMState.DEAD:
            self.dead_remain_ms = max(0.0, self.dead_remain_ms - dt_ms)
            if self.dead_remain_ms <= 0:
                self.state = SMState.IDLE
                self.reset_round()
                self.idle_accum_ms = 0.0
                self.lost_display_ms = 0.0
            return

        if self.state == SMState.SLEEP:
            self.sleep_remain_ms = max(0.0, self.sleep_remain_ms - dt_ms)
            if self.sleep_remain_ms <= 0:
                self.state = SMState.IDLE
                self.idle_accum_ms = 0.0
            return

        if self.state == SMState.IDLE:
            self.lost_display_ms = max(0.0, self.lost_display_ms - dt_ms)
            if in_lock_zone:
                self.lock_debounce_ms += dt_ms
                self.idle_accum_ms = 0.0
            else:
                self.lock_debounce_ms = 0.0
                if in_idle_zone:
                    self.idle_accum_ms += dt_ms

            if self.lock_debounce_ms >= cfg.debounce_lock_ms:
                self.state = SMState.LOCK
                self.lock_debounce_ms = 0.0
                self.lost_ms = 0.0
                self.lost_debounce_ms = 0.0
                self.lock_elapsed_ms = 0.0
                self.feature_cache = _snapshot_features(snap)
                self.idle_accum_ms = 0.0
                self.lost_display_ms = 0.0
            elif self.idle_accum_ms >= cfg.idle_before_sleep_ms:
                self.state = SMState.SLEEP
                self.sleep_remain_ms = float(cfg.sleep_duration_ms)
                self.idle_accum_ms = 0.0
            return

        # LOCK
        self.lock_elapsed_ms += dt_ms

        if in_lock_zone:
            if 0 < self.lost_ms < cfg.reconnect_window_ms:
                if _features_match(self.feature_cache, snap):
                    pass
                else:
                    self.reset_round()
                    self.feature_cache = _snapshot_features(snap)
            self.lost_ms = 0.0
            self.lost_debounce_ms = 0.0
        elif in_idle_zone:
            self.lost_debounce_ms += dt_ms
            if self.lost_debounce_ms >= cfg.debounce_lost_ms:
                self.lost_ms += dt_ms
            if self.lost_ms >= cfg.reconnect_window_ms:
                self.state = SMState.IDLE
                self.reset_round()
                self.idle_accum_ms = 0.0
                self.lost_ms = 0.0
                self.lost_debounce_ms = 0.0
                self.lost_display_ms = 1800.0

    def on_tts_playback_ended(self, cfg: RuntimeConfig) -> None:
        if self.state != SMState.LOCK:
            return
        self.sentence_index += 1
        if self.sentence_index >= 10:
            self.state = SMState.DEAD
            self.dead_remain_ms = float(cfg.purge_dead_ms)
            self.reset_round()

    def can_request_tracking_llm(self, cfg: RuntimeConfig) -> bool:
        if self.state != SMState.LOCK:
            return False
        if self.suppress_tts:
            return False
        if self.lock_elapsed_ms < cfg.post_lock_tts_delay_ms:
            return False
        return True

    def can_request_idle_llm(self) -> bool:
        return self.state == SMState.IDLE


def _snapshot_features(snap: VisionSnapshot) -> str:
    parts: list[str] = []
    if snap.bbox:
        x, y, w, h = snap.bbox
        parts.append(f"box={w}x{h}")
        parts.append(f"cx={x + w // 2}")
    parts.append(f"m={snap.metric_value:.0f}")
    parts.extend(snap.behavior_tags[:4])
    return "|".join(parts)


def _features_match(cached: str, snap: VisionSnapshot) -> bool:
    if not cached:
        return True
    cur = _snapshot_features(snap)
    ca = set(cached.split("|"))
    cb = set(cur.split("|"))
    return len(ca & cb) >= 2

"""Background capture loop, state machine, speech pipeline, MJPEG buffer."""

from __future__ import annotations

import math
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.audio_playback import play_wav_async, resolve_backend
from src.config import RuntimeConfig
from src.geometry_eyes import compute_eye_angles
from src.gpu_metrics import collect_gpu_metrics
from src.ollama_client import generate_line_sync, preload_ollama_model
from src.paths import PROJECT_ROOT
from src.serial_servo import SerialServo
from src.state_machine import SMState, StateMachine, VisionSnapshot
from src.runtime_log import append_log
from src.tts_qwen import TTSBackend
from src.vision import VisionEngine


def _ui_state_label(
    cfg: RuntimeConfig, sm: StateMachine, now_m: float, frenzy_until: float
) -> str:
    """WS / 儀表板狀態名（對齊規格：含 RECONNECT_PENDING、PURGE、LOST）。"""
    if sm.state == SMState.DEAD:
        if now_m < frenzy_until:
            return "PURGE"
        return "DEAD"
    if sm.state == SMState.SLEEP:
        return "SLEEP"
    if sm.state == SMState.IDLE:
        if sm.lost_display_ms > 0:
            return "LOST"
        return "IDLE"
    if 0 < sm.lost_ms < cfg.reconnect_window_ms:
        return "RECONNECT_PENDING"
    return "LOCK"


@dataclass
class PublicState:
    state: str = "IDLE"
    bbox_value: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    audience_features: str = ""
    sentence_index: int = 0
    llm_generating: bool = False
    tts_generating: bool = False
    last_llm_ms: int = 0
    last_tts_ms: int = 0
    # 語音管線（儀表板：LLM 第幾句 → TTS → 播放）
    speech_phase: str = "idle"  # idle | queued | llm | tts | playing
    speech_mode: str = "none"  # none | tracking | idle
    speech_sentence: int = 0  # 鎖定模式：本輪將產出之句序 1–10；閒置為 0
    angle_left_deg: float = 90.0
    angle_right_deg: float = 90.0
    mode_servo: str = "IDLE"
    audio_backend_active: str = ""
    tts_backend: str = ""
    behavior_tags: list[str] = field(default_factory=list)
    vision_device: str = ""
    gpu_metrics: dict[str, Any] = field(default_factory=dict)
    capture_fps: float = 0.0
    capture_width: int = 0
    capture_height: int = 0
    capture_target_fps: float = 15.0
    capture_frame_width: int = 640
    capture_frame_height: int = 480


class Orchestrator:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cfg = RuntimeConfig()
        self.sm = StateMachine()
        self.vision = VisionEngine(self._cfg.yolo_weights_path)
        self.serial = SerialServo()
        self.tts = TTSBackend()
        self._tts_mode = "stub"

        self._cap: cv2.VideoCapture | None = None
        self._jpeg_lock = threading.Lock()
        self._jpeg: bytes = b""
        self.public = PublicState()

        self._speech_lock = threading.Lock()
        self._speech_busy = False
        self._job_q: queue.Queue[str] = queue.Queue(maxsize=4)
        self._last_idle_utterance_t = 0.0
        self._frenzy_until = 0.0
        self._prev_sm_state = SMState.IDLE

        self._pending_round_reset = False
        self._camera_reopen_requested = False
        self._audience_file_live = False

        self._running = False
        self._threads: list[threading.Thread] = []
        self._capture_fps_count = 0
        self._capture_fps_window_t0 = 0.0

    def _reset_speech_pipeline_public(self) -> None:
        self.public.speech_phase = "idle"
        self.public.speech_mode = "none"
        self.public.speech_sentence = 0

    def get_config(self) -> dict[str, Any]:
        with self._lock:
            return self._cfg.model_dump()

    def patch_config(self, patch: dict[str, Any]) -> None:
        with self._lock:
            d = self._cfg.model_dump()
            for k, v in patch.items():
                if k in RuntimeConfig.model_fields:
                    d[k] = v
            self._cfg = RuntimeConfig(**d)
            if "yolo_weights_path" in patch:
                self.vision.reload_weights(self._cfg.yolo_weights_path)
            if "serial_port" in patch or "serial_baud" in patch:
                self.serial.connect(self._cfg.serial_port, self._cfg.serial_baud)
            if any(
                k in patch
                for k in ("camera_index", "capture_frame_width", "capture_frame_height")
            ):
                self._camera_reopen_requested = True
            if any(
                k in patch
                for k in (
                    "qwen_tts_model_id",
                    "tts_cache_dir",
                    "ref_audio_path",
                    "ref_text_path",
                )
            ):
                self._tts_mode = self.tts.ensure_model(
                    self._cfg.qwen_tts_model_id,
                    self._cfg.tts_cache_dir,
                    self._cfg.ref_audio_path,
                    self._cfg.ref_text_path,
                )

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        with self._lock:
            self.serial.connect(self._cfg.serial_port, self._cfg.serial_baud)
            self._tts_mode = self.tts.ensure_model(
                self._cfg.qwen_tts_model_id,
                self._cfg.tts_cache_dir,
                self._cfg.ref_audio_path,
                self._cfg.ref_text_path,
            )
        self._audience_file_live = False
        append_log("system", "orchestrator_start", {"tts_mode": self._tts_mode})
        self.public.vision_device = self.vision._device
        tp = threading.Thread(target=self._ollama_preload_worker, daemon=True)
        tp.start()
        self._threads.append(tp)
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()
        self._threads.append(t)
        t2 = threading.Thread(target=self._speech_worker, daemon=True)
        t2.start()
        self._threads.append(t2)
        tg = threading.Thread(target=self._gpu_metrics_loop, daemon=True)
        tg.start()
        self._threads.append(tg)

    def _gpu_metrics_loop(self) -> None:
        while self._running:
            try:
                snap = collect_gpu_metrics()
                with self._lock:
                    self.public.gpu_metrics = snap
            except Exception:
                with self._lock:
                    self.public.gpu_metrics = {"ts": time.time(), "error": "metrics_collect_failed"}
            for _ in range(10):
                if not self._running:
                    break
                time.sleep(0.1)

    def _ollama_preload_worker(self) -> None:
        try:
            with self._lock:
                cfg = self._cfg.model_copy(deep=False)
            preload_ollama_model(
                base_url=cfg.ollama_base_url,
                model=cfg.ollama_model,
                pull_timeout_s=cfg.ollama_pull_timeout_s,
                keep_alive=cfg.ollama_keep_alive,
            )
            append_log(
                "system",
                "ollama_preload_ok",
                {"model": cfg.ollama_model, "keep_alive": cfg.ollama_keep_alive},
            )
        except Exception as e:
            append_log("system", "ollama_preload_error", {"error": str(e)})

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()

    def mjpeg_frame(self) -> bytes:
        with self._jpeg_lock:
            return self._jpeg

    def consume_round_reset(self) -> bool:
        with self._lock:
            if self._pending_round_reset:
                self._pending_round_reset = False
                return True
            return False

    def _open_camera(self) -> None:
        if self._cap:
            self._cap.release()
        with self._lock:
            idx = self._cfg.camera_index
            fw = int(self._cfg.capture_frame_width)
            fh = int(self._cfg.capture_frame_height)
        fw = max(160, min(3840, fw))
        fh = max(120, min(2160, fh))
        self._cap = cv2.VideoCapture(idx)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(fw))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(fh))

    def _capture_loop(self) -> None:
        with self._lock:
            delay_s = max(0, int(self._cfg.startup_camera_delay_ms)) / 1000.0
        vision_ready_at = time.monotonic() + delay_s
        while self._running and time.monotonic() < vision_ready_at:
            rem = vision_ready_at - time.monotonic()
            time.sleep(min(0.25, rem) if rem > 0 else 0.0)
        if not self._running:
            return
        self._open_camera()
        with self._lock:
            frame_area = float(max(1, self._cfg.capture_frame_width) * max(1, self._cfg.capture_frame_height))
        last = time.perf_counter()
        idle_sway_t = 0.0
        while self._running:
            if self._camera_reopen_requested:
                self._camera_reopen_requested = False
                self._open_camera()
            now = time.perf_counter()
            dt_ms = (now - last) * 1000
            last = now
            ok, frame = self._cap.read() if self._cap else (False, None)
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            t_frame_start = time.perf_counter()
            h, w = frame.shape[:2]
            frame_area = float(w * h)

            now_m_fps = time.monotonic()
            if self._capture_fps_window_t0 <= 0.0:
                self._capture_fps_window_t0 = now_m_fps
            self._capture_fps_count += 1
            elapsed_fps = now_m_fps - self._capture_fps_window_t0
            if elapsed_fps >= 1.0:
                self.public.capture_fps = self._capture_fps_count / elapsed_fps
                self._capture_fps_count = 0
                self._capture_fps_window_t0 = now_m_fps
            self.public.capture_width = int(w)
            self.public.capture_height = int(h)

            with self._lock:
                cfg = self._cfg.model_copy(deep=False)
            self.public.capture_target_fps = float(cfg.capture_target_fps)
            self.public.capture_frame_width = int(cfg.capture_frame_width)
            self.public.capture_frame_height = int(cfg.capture_frame_height)

            snap = self.vision.detect(frame, cfg.bbox_metric, frame_area)
            prev_fsm = self._prev_sm_state
            self.sm.tick(cfg, snap, dt_ms)

            if prev_fsm != self.sm.state:
                append_log(
                    "state_machine",
                    f"{prev_fsm.value}->{self.sm.state.value}",
                    {
                        "metric": round(snap.metric_value, 1),
                        "bbox": list(snap.bbox) if snap.bbox else None,
                    },
                )

            if self._prev_sm_state == SMState.DEAD and self.sm.state != SMState.DEAD:
                with self._lock:
                    self._pending_round_reset = True

            if self._prev_sm_state != self.sm.state and self.sm.state == SMState.DEAD:
                self._frenzy_until = time.monotonic() + 2.0
            self._prev_sm_state = self.sm.state

            l_deg, r_deg = compute_eye_angles(cfg, w, h, snap.bbox)
            now_m = time.monotonic()
            if self.sm.state == SMState.DEAD:
                if now_m < self._frenzy_until:
                    l_deg += random.uniform(-8, 8)
                    r_deg += random.uniform(-8, 8)
                    self.serial.send("FRENZY", l_deg, r_deg)
                    mode = "FRENZY"
                else:
                    self.serial.send("DEAD", cfg.servo_left_neutral_deg, cfg.servo_right_neutral_deg)
                    mode = "DEAD"
            elif self.sm.state == SMState.LOCK:
                self.serial.send("TRACK", l_deg, r_deg)
                mode = "TRACK"
            elif self.sm.state == SMState.SLEEP:
                self.serial.send("IDLE", cfg.servo_left_neutral_deg, cfg.servo_right_neutral_deg)
                mode = "IDLE"
            else:
                idle_sway_t += dt_ms / 1000.0
                off = 15 * math.sin(idle_sway_t * 0.4)
                self.serial.send(
                    "IDLE",
                    cfg.servo_left_neutral_deg + off,
                    cfg.servo_right_neutral_deg - off * 0.7,
                )
                mode = "IDLE"

            ui_lbl = _ui_state_label(cfg, self.sm, now_m, self._frenzy_until)
            self.public.state = ui_lbl
            self.public.bbox_value = snap.metric_value
            self.public.bbox = snap.bbox
            self.public.audience_features = self.sm.feature_cache
            self.public.sentence_index = self.sm.sentence_index
            self.public.angle_left_deg = l_deg
            self.public.angle_right_deg = r_deg
            self.public.mode_servo = mode
            self.public.behavior_tags = list(snap.behavior_tags)
            self.public.audio_backend_active = resolve_backend(cfg.audio_backend)
            self.public.tts_backend = self._tts_mode
            self.public.vision_device = self.vision._device

            if self.sm.state == SMState.LOCK and self.sm.feature_cache:
                self._audience_file_live = True
            else:
                if self._audience_file_live:
                    append_log(
                        "audience_cache",
                        "cleared",
                        {"state": self.sm.state.value, "had_features": bool(self.sm.feature_cache)},
                    )
                    self._audience_file_live = False

            x, y, bw, bh = (0, 0, 0, 0)
            if snap.bbox:
                x, y, bw, bh = snap.bbox
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{ui_lbl} m={snap.metric_value:.0f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if ok2:
                with self._jpeg_lock:
                    self._jpeg = buf.tobytes()

            self._maybe_enqueue_speech(cfg, snap)
            target_fps = max(0.5, min(120.0, float(cfg.capture_target_fps)))
            frame_budget_s = 1.0 / target_fps
            spent_s = time.perf_counter() - t_frame_start
            rem_s = frame_budget_s - spent_s
            if rem_s > 0:
                time.sleep(rem_s)

    def _maybe_enqueue_speech(self, cfg: RuntimeConfig, snap: VisionSnapshot) -> None:
        with self._speech_lock:
            if self._speech_busy:
                return
        if self.sm.state == SMState.LOCK and self.sm.can_request_tracking_llm(cfg):
            if not self.sm.suppress_tts:
                try:
                    self._job_q.put_nowait("tracking")
                    self.public.speech_phase = "queued"
                    self.public.speech_mode = "tracking"
                    self.public.speech_sentence = self.sm.sentence_index + 1
                    with self._speech_lock:
                        self._speech_busy = True
                except queue.Full:
                    pass
        elif self.sm.state == SMState.IDLE:
            now = time.monotonic()
            if now - self._last_idle_utterance_t > 12.0:
                try:
                    self._job_q.put_nowait("idle")
                    self.public.speech_phase = "queued"
                    self.public.speech_mode = "idle"
                    self.public.speech_sentence = 0
                    with self._speech_lock:
                        self._speech_busy = True
                    self._last_idle_utterance_t = now
                except queue.Full:
                    pass

    def _speech_worker(self) -> None:
        while self._running:
            try:
                job = self._job_q.get(timeout=0.3)
            except queue.Empty:
                continue
            with self._lock:
                cfg = self._cfg.model_copy(deep=False)
            try:
                if job == "tracking":
                    self._run_tracking_utterance(cfg)
                elif job == "idle":
                    self._run_idle_utterance(cfg)
            except Exception:
                self._reset_speech_pipeline_public()
                with self._speech_lock:
                    self._speech_busy = False

    def _run_tracking_utterance(self, cfg: RuntimeConfig) -> None:
        if self.sm.state != SMState.LOCK:
            self._reset_speech_pipeline_public()
            with self._speech_lock:
                self._speech_busy = False
            return
        sn = self.sm.sentence_index + 1
        self.public.speech_mode = "tracking"
        self.public.speech_sentence = sn
        self.public.speech_phase = "llm"
        payload = {
            "sentence_index": sn,
            "feature_cache": self.sm.feature_cache,
            "behavior_tags": self.public.behavior_tags,
            "suppress_tts": self.sm.suppress_tts,
        }
        self.public.llm_generating = True
        try:
            text, llm_ms = generate_line_sync(
                base_url=cfg.ollama_base_url,
                model=cfg.ollama_model,
                timeout_s=cfg.ollama_timeout_s,
                pull_timeout_s=cfg.ollama_pull_timeout_s,
                max_retries=cfg.ollama_max_retries,
                warmup=cfg.ollama_warmup,
                keep_alive=cfg.ollama_keep_alive,
                system_persona_rel=cfg.persona_tracking_path,
                few_shot_paths=list(cfg.few_shot_tracking_paths),
                few_shot_max_rows=cfg.few_shot_max_rows_per_file,
                user_payload=payload,
                max_chars=22,
            )
            append_log(
                "llm",
                "tracking_line",
                {"sentence": sn, "ms": llm_ms, "text": text[:500]},
            )
        except Exception as e:
            text, llm_ms = f"（Ollama 錯誤：{e}）"[:22], 0
            append_log(
                "llm",
                "tracking_error",
                {"sentence": sn, "error": str(e)},
            )
        self.public.llm_generating = False
        self.public.last_llm_ms = llm_ms
        self.public.speech_phase = "tts"
        self._synthesize_and_play(cfg, text, tracking=True, max_chars=22)

    def _run_idle_utterance(self, cfg: RuntimeConfig) -> None:
        if self.sm.state != SMState.IDLE:
            self._reset_speech_pipeline_public()
            with self._speech_lock:
                self._speech_busy = False
            return
        self.public.speech_mode = "idle"
        self.public.speech_sentence = 0
        self.public.speech_phase = "llm"
        payload = {"mode": "idle", "ts": time.time()}
        self.public.llm_generating = True
        try:
            text, llm_ms = generate_line_sync(
                base_url=cfg.ollama_base_url,
                model=cfg.ollama_model,
                timeout_s=cfg.ollama_timeout_s,
                pull_timeout_s=cfg.ollama_pull_timeout_s,
                max_retries=cfg.ollama_max_retries,
                warmup=cfg.ollama_warmup,
                keep_alive=cfg.ollama_keep_alive,
                system_persona_rel=cfg.persona_idle_path,
                few_shot_paths=list(cfg.few_shot_idle_paths),
                few_shot_max_rows=cfg.few_shot_max_rows_per_file,
                user_payload=payload,
                max_chars=15,
            )
            append_log("llm", "idle_line", {"ms": llm_ms, "text": text[:500]})
        except Exception as e:
            text, llm_ms = f"靜默{e}"[:15], 0
            append_log("llm", "idle_error", {"error": str(e)})
        self.public.llm_generating = False
        self.public.last_llm_ms = llm_ms
        self.public.speech_phase = "tts"
        self._synthesize_and_play(cfg, text, tracking=False, max_chars=15)

    def _synthesize_and_play(
        self, cfg: RuntimeConfig, text: str, *, tracking: bool, max_chars: int
    ) -> None:
        import time as time_mod

        t0 = time_mod.perf_counter()
        self.public.tts_generating = True
        out = PROJECT_ROOT / "tmp" / "utterance.wav"
        try:
            tts_kind = self.tts.synthesize_to_wav(text[: max_chars + 5], out)
            append_log(
                "tts",
                "synthesize_qwen" if tts_kind == "qwen" else "synthesize_stub",
                {"preview": text[:120], "path": str(out), "kind": tts_kind},
            )
        except Exception as e:
            append_log("tts", "synthesize_exception", {"error": str(e), "preview": text[:120]})
            try:
                out.write_bytes(b"")
            except Exception:
                pass
        self.public.tts_generating = False
        self.public.last_tts_ms = int((time_mod.perf_counter() - t0) * 1000)
        self.public.speech_phase = "playing"

        def on_end() -> None:
            try:
                if tracking and self.sm.state == SMState.LOCK:
                    self.sm.on_tts_playback_ended(cfg)
            finally:
                self._reset_speech_pipeline_public()
                with self._speech_lock:
                    self._speech_busy = False

        play_wav_async(out, cfg.audio_backend, on_end)

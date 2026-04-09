from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import random
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.audio.player import AudioPlayer
from backend.config import build_field_catalog, merge_config, validate_runtime_config
from backend.device_utils import (
    backend_label_for_device,
    expected_accelerator_label,
    expected_tts_backend_label,
    expected_vision_backend_label,
)
from backend.llm.ollama_client import OllamaClient
from backend.model_manager import ensure_runtime_models
from backend.prompting.prompt_builder import PromptBuilder, validate_generated_sentence
from backend.resource_manager import ResourceManager
from backend.runtime_shutdown import clear_shutdown_request, install_shutdown_signal_bridge, request_shutdown, shutdown_requested
from backend.serial.esp32_link import ESP32Link
from backend.servo.geometry import compute_servo_angles
from backend.state_machine import RuntimeState
from backend.storage.csv_logger import append_audience_snapshot
from backend.telemetry.system_stats import capture_process_footprint, diff_process_footprint, get_system_stats
from backend.tts.model_profiles import DEFAULT_TTS_MODEL_PROFILE, resolve_tts_model_profile
from backend.tts.qwen_clone import BenchmarkShutdownRequested, FishCloneTTS, TTSAutoBenchmarkSelection
from backend.tts.reference_selection import (
    ReferencePair,
    build_fixed_reference_pair,
    choose_random_emotional_reference_pair,
    emotional_reference_pair_map,
    load_emotional_reference_pairs,
)
from backend.types import ConfigUpdateResponse, PipelineStage, RuntimeComponentStats, RuntimeConfig, SystemMode
from backend.vision.runtime import VisionRuntime

DEFAULT_TTS_EMOTION = "confident"
REFERENCE_PAIR_TAG_TO_NAME = {
    "感激愧疚": "感激與愧疚",
    "刻薄質問": "刻薄的質問與探詢",
    "冷靜專業": "冷靜且專業",
    "激動不信": "情緒激動且不可置信",
    "震驚崩潰": "震驚與崩潰",
    "懷疑嚴厲": "懷疑且嚴厲",
    "悲慟自責絕望": "極度的悲慟和自責與絕望",
}
REFERENCE_PAIR_NAME_TO_TAG = {value: key for key, value in REFERENCE_PAIR_TAG_TO_NAME.items()}
RUNTIME_STATUS_REFRESH_INTERVAL_SEC = 2.0


class DisabledTTS:
    def __init__(self) -> None:
        self.loaded = False
        self.device = "disabled"
        self.device_backend = "disabled"
        self.semantic_dispatch_mode = None
        self.precision_mode = None
        self.model_profile = resolve_tts_model_profile(RuntimeConfig().tts_model_path)

    def preload(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def synthesize(self, text: str, output_path: str, *, request_overrides: dict | None = None) -> str:
        raise RuntimeError("TTS is disabled in YOLO-only mode")

    def set_reference_paths(self, ref_audio_path: str, ref_text_path: str) -> None:
        return None


def should_run_yolo_only_mode() -> bool:
    return os.getenv("MOMO_YOLO_ONLY") == "1"


class Brain:
    def __init__(self) -> None:
        self.config = RuntimeConfig()
        self.yolo_only_mode = should_run_yolo_only_mode()
        self.state = RuntimeState()
        self.prompts = PromptBuilder("resource/md/system-persona_tracking.md", "resource/md/system-persona_idle.md")
        self.resources = ResourceManager("tmp")
        self.audio = AudioPlayer()
        self.tts_benchmark_selected: str | None = None
        self.tts_benchmark_results: list[str] = []
        if self.yolo_only_mode:
            self.tts_runtime = RuntimeComponentStats(
                requested_mode="disabled",
                effective_device="disabled",
                backend="disabled",
                selection_source="startup-flag",
            )
            self.tts = DisabledTTS()
        else:
            self.tts_runtime = RuntimeComponentStats()
            self.tts = self._build_tts_runtime(selection_source="default")
        self.serial = ESP32Link(self.config.serial_port, self.config.serial_baud_rate)
        self.vision = VisionRuntime(self.config)
        self.yolo_person_runtime = RuntimeComponentStats(
            requested_mode=self.config.yolo_device_mode,
            effective_device=self.vision.detector.device,
            backend=backend_label_for_device(self.vision.detector.device),
            selection_source="default",
        )
        self.yolo_pose_runtime = RuntimeComponentStats(
            requested_mode=self.config.yolo_device_mode,
            effective_device=self.vision.pose.device,
            backend=backend_label_for_device(self.vision.pose.device),
            selection_source="default",
        )
        self.ollama_runtime = RuntimeComponentStats(
            requested_mode="disabled" if self.yolo_only_mode else self.config.ollama_device_mode,
            effective_device="disabled" if self.yolo_only_mode else self._expected_ollama_backend_label(expected_accelerator_label()),
            backend="disabled" if self.yolo_only_mode else self._expected_ollama_backend_label(expected_accelerator_label()),
            selection_source="startup-flag" if self.yolo_only_mode else "default",
        )
        self.ollama_connected = False
        self.history: deque[str] = deque(maxlen=10)
        self.last_target_seen = 0.0
        self.lock_started_at: float | None = None
        self.last_idle_line_at = 0.0
        self.last_sentence_finished_at = 0.0
        self.cooldown_until = 0.0
        self.generation_lock = asyncio.Lock()
        self.background_tasks: list[asyncio.Task] = []
        self._random = random.Random()
        self._last_tts_reference_pair_key: str | None = None

    async def start(self) -> None:
        clear_shutdown_request()
        if should_prepare_models():
            if self.yolo_only_mode:
                await asyncio.to_thread(self._prepare_vision_models_only)
            else:
                await asyncio.to_thread(self._prepare_runtime_models)
        await self._print_startup_diagnostics()
        if not self.yolo_only_mode:
            await self.refresh_runtime_status()
        self.vision.start()
        self.background_tasks = [
            asyncio.create_task(self.vision_loop()),
            asyncio.create_task(self.housekeeping_loop()),
            asyncio.create_task(self.runtime_status_loop()),
        ]

    async def stop(self) -> None:
        request_shutdown()
        for task in self.background_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.vision.stop()
        self.serial.close()

    def _prepare_runtime_models(self) -> None:
        ensure_runtime_models(self.config)
        self._replace_tts_runtime(self._select_tts_runtime(selection_source="default"))
        try:
            self._preload_tts_runtime()
        except Exception as exc:
            if not self._recover_tts_from_oom(exc):
                self.state.event_log = [f"TTS preload failed: {exc}", *self.state.event_log][:20]
        self.vision = VisionRuntime(self.config)
        person_device = getattr(getattr(self.vision, "detector", None), "device", None)
        pose_device = getattr(getattr(self.vision, "pose", None), "device", None)
        self.yolo_person_runtime = RuntimeComponentStats(
            requested_mode=self.config.yolo_device_mode,
            effective_device=person_device,
            backend=backend_label_for_device(person_device or "cpu") if person_device else None,
            selection_source="default",
        )
        self.yolo_pose_runtime = RuntimeComponentStats(
            requested_mode=self.config.yolo_device_mode,
            effective_device=pose_device,
            backend=backend_label_for_device(pose_device or "cpu") if pose_device else None,
            selection_source="default",
        )

    def _prepare_vision_models_only(self) -> None:
        ensure_runtime_models(self.config, vision_only=True)
        self.vision = VisionRuntime(self.config)
        person_device = getattr(getattr(self.vision, "detector", None), "device", None)
        pose_device = getattr(getattr(self.vision, "pose", None), "device", None)
        self.yolo_person_runtime = RuntimeComponentStats(
            requested_mode=self.config.yolo_device_mode,
            effective_device=person_device,
            backend=backend_label_for_device(person_device or "cpu") if person_device else None,
            selection_source="default",
        )
        self.yolo_pose_runtime = RuntimeComponentStats(
            requested_mode=self.config.yolo_device_mode,
            effective_device=pose_device,
            backend=backend_label_for_device(pose_device or "cpu") if pose_device else None,
            selection_source="default",
        )

    def _build_tts_runtime(
        self,
        selection: TTSAutoBenchmarkSelection | None = None,
        *,
        selection_source: str,
    ) -> FishCloneTTS:
        startup_reference = self._startup_tts_reference_pair()
        if selection is not None:
            requested_mode = self.config.tts_device_mode
            selected_mode = getattr(selection.result, "device_mode", getattr(selection.tts, "device_mode", self.config.tts_device_mode))
            self.config.tts_device_mode = selected_mode
            self.tts_benchmark_selected = selection.result.name
            self.tts_benchmark_results = [
                f"{item.name}:{'ok' if item.ok else 'error'}:{item.preload_ms}:{item.synth_ms}:{item.elapsed_ms}:"
                f"{item.peak_vram_mb}:{item.semantic_dispatch_mode}:{item.precision_mode}"
                + (f":{item.detail}" if item.detail else "")
                for item in selection.results
            ]
            self.tts_runtime = RuntimeComponentStats(
                requested_mode=requested_mode,
                effective_device=getattr(selection.tts, "device", None),
                backend=getattr(selection.tts, "device_backend", None),
                selection_source="benchmark",
                semantic_dispatch_mode=getattr(selection.tts, "semantic_dispatch_mode", None),
                precision_mode=getattr(selection.tts, "precision_mode", getattr(selection.result, "precision_mode", None)),
                ram_mb=getattr(selection.result, "ram_mb", None),
                vram_mb=getattr(selection.result, "vram_mb", None),
            )
            return selection.tts

        self.tts_benchmark_selected = None
        self.tts_benchmark_results = []
        tts = FishCloneTTS(
            self.config.tts_model_path,
            startup_reference.audio_path,
            startup_reference.text_path,
            clone_voice_enabled=self.config.tts_clone_voice_enabled,
            kokoro_voice=self.config.tts_kokoro_voice,
            device_mode=self.config.tts_device_mode,
        )
        self.tts_runtime = RuntimeComponentStats(
            requested_mode=self.config.tts_device_mode,
            effective_device=getattr(tts, "device", None),
            backend=getattr(tts, "device_backend", None),
            selection_source=selection_source,
            semantic_dispatch_mode=getattr(tts, "semantic_dispatch_mode", None),
            precision_mode=getattr(tts, "precision_mode", None),
        )
        return tts

    def _preload_tts_runtime(self) -> None:
        if self.tts.loaded:
            return
        tts_device = getattr(self.tts, "device", None)
        before = capture_process_footprint(tts_device)
        self.tts.preload()
        after = capture_process_footprint(tts_device)
        ram_mb, vram_mb = diff_process_footprint(before, after)
        self.tts_runtime.ram_mb = ram_mb
        self.tts_runtime.vram_mb = vram_mb

    def _replace_tts_runtime(self, next_tts: FishCloneTTS) -> None:
        previous = getattr(self, "tts", None)
        self.tts = next_tts
        if previous is None or previous is next_tts:
            return
        unload = getattr(previous, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception as exc:
                self.state.event_log = [f"TTS unload warning: {exc}", *self.state.event_log][:20]

    def _recover_tts_from_oom(self, exc: Exception) -> bool:
        current_mode = (self.config.tts_device_mode or "").strip().lower()
        if current_mode not in {"gpu", "mps"}:
            return False
        if should_skip_tts_benchmark() or not self._looks_like_memory_pressure(exc):
            return False

        requested_mode = current_mode
        self.state.event_log = [f"TTS hit OOM on {requested_mode}; retrying with auto benchmark.", *self.state.event_log][:20]
        self.config.tts_device_mode = "auto"
        try:
            self._replace_tts_runtime(self._select_tts_runtime(selection_source="benchmark"))
            self._preload_tts_runtime()
            self.tts_runtime.requested_mode = requested_mode
            self.state.event_log = [
                f"TTS recovered from OOM via benchmark fallback: {self.tts_runtime.effective_device or self.tts_runtime.backend}.",
                *self.state.event_log,
            ][:20]
            return True
        except Exception as fallback_exc:
            self.state.event_log = [f"TTS auto fallback failed after OOM: {fallback_exc}", *self.state.event_log][:20]
            return False

    def _looks_like_memory_pressure(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "out of memory",
                "cuda out of memory",
                "mps backend out of memory",
                "not enough memory",
                "cuda error: out of memory",
            )
        )

    def _select_tts_runtime(self, *, selection_source: str) -> FishCloneTTS:
        profile = resolve_tts_model_profile(self.config.tts_model_path)
        if self.config.tts_device_mode != "auto" or should_skip_tts_benchmark() or not profile.supports_startup_benchmark:
            return self._build_tts_runtime(selection_source=selection_source)

        benchmark_reference = self._benchmark_tts_reference_pair()
        selection = FishCloneTTS.benchmark_auto_profiles(
            self.config.tts_model_path,
            benchmark_reference.audio_path,
            benchmark_reference.text_path,
            clone_voice_enabled=self.config.tts_clone_voice_enabled,
            kokoro_voice=self.config.tts_kokoro_voice,
        )
        if selection is None:
            self.state.event_log = [
                "TTS auto benchmark did not find a usable profile; falling back to default auto mode.",
                *self.state.event_log,
            ][:20]
            return self._build_tts_runtime(selection_source=selection_source)

        self.state.event_log = [
            f"TTS benchmark selected {selection.result.name} with synth {selection.result.synth_ms} ms "
            f"(preload {selection.result.preload_ms} ms, total {selection.result.elapsed_ms} ms).",
            *self.state.event_log,
        ][:20]
        return self._build_tts_runtime(selection, selection_source="benchmark")

    async def _print_startup_diagnostics(self) -> None:
        for line in await self._collect_startup_diagnostics():
            print(line, flush=True)

    async def _collect_startup_diagnostics(self) -> list[str]:
        expected = expected_accelerator_label()
        vision_expected = expected_vision_backend_label(self.config.yolo_device_mode)
        lines = [f"[startup] expected_accelerator={expected}", f"[startup] expected_vision_backend={vision_expected}"]
        try:
            await asyncio.to_thread(self._refresh_vision_runtime_stats)
            person_backend = self.yolo_person_runtime.backend or "unknown"
            pose_backend = self.yolo_pose_runtime.backend or "unknown"
            lines.append(
                f"[startup] yolo person={person_backend} pose={pose_backend} target={vision_expected} ok={person_backend == vision_expected and pose_backend == vision_expected}"
            )
        except Exception as exc:
            lines.append(f"[startup] yolo error={exc}")

        if self.yolo_only_mode:
            lines.append("[startup] yolo_only=true tts=disabled ollama=disabled")
            return lines

        tts_expected = expected_tts_backend_label(self.config.tts_device_mode)
        lines.append(
            f"[startup] tts backend={self.tts.device_backend} target={tts_expected} loaded={self.tts.loaded} ok={self.tts.device_backend == tts_expected}"
        )
        if self.tts_benchmark_selected:
            lines.append(f"[startup] tts benchmark selected={self.tts_benchmark_selected}")
        if self.tts_benchmark_results:
            lines.append(f"[startup] tts benchmark results={'; '.join(self.tts_benchmark_results)}")

        try:
            ollama = OllamaClient(self.config.ollama_base_url, min(self.config.ollama_timeout_sec, 30), self.config.ollama_device_mode)
            models = await ollama.list_models()
            if self.config.ollama_model not in models:
                lines.append(f"[startup] ollama model={self.config.ollama_model} available=false")
                return lines
            await ollama.warmup_model(self.config.ollama_model)
            running = await ollama.running_models()
            current = next(
                (
                    item for item in running
                    if item.get("name") == self.config.ollama_model or item.get("model") == self.config.ollama_model
                ),
                None,
            )
            size_vram = int(current.get("size_vram", 0)) if current else 0
            target = self._expected_ollama_backend_label(expected)
            backend = expected if size_vram > 0 else "cpu"
            lines.append(
                f"[startup] ollama backend={backend} target={target} size_vram={size_vram} ok={target == 'auto' or backend == target}"
            )
        except Exception as exc:
            lines.append(f"[startup] ollama error={exc}")
        return lines

    def _expected_ollama_backend_label(self, accelerator: str) -> str:
        mode = (self.config.ollama_device_mode or "auto").strip().lower()
        if mode == "cpu":
            return "cpu"
        if mode == "auto":
            return "auto"
        return accelerator

    def _refresh_vision_runtime_stats(self, person_backend: str | None = None, pose_backend: str | None = None) -> None:
        person_device = getattr(self.vision.detector, "device", "cpu")
        pose_device = getattr(self.vision.pose, "device", "cpu")
        if person_backend is None:
            person_backend = backend_label_for_device(person_device)
        if pose_backend is None:
            pose_backend = backend_label_for_device(pose_device)
        person_before = capture_process_footprint(person_device)
        person_backend = self.vision.detector.warmup()
        person_after = capture_process_footprint(person_device)
        person_ram_mb, person_vram_mb = diff_process_footprint(person_before, person_after)
        pose_before = capture_process_footprint(pose_device)
        pose_backend = self.vision.pose.warmup()
        pose_after = capture_process_footprint(pose_device)
        pose_ram_mb, pose_vram_mb = diff_process_footprint(pose_before, pose_after)
        source = "default" if self.config.yolo_device_mode == RuntimeConfig().yolo_device_mode else "user"
        self.yolo_person_runtime = RuntimeComponentStats(
            requested_mode=self.config.yolo_device_mode,
            effective_device=person_device,
            backend=person_backend,
            selection_source=source,
            ram_mb=person_ram_mb,
            vram_mb=person_vram_mb,
        )
        self.yolo_pose_runtime = RuntimeComponentStats(
            requested_mode=self.config.yolo_device_mode,
            effective_device=pose_device,
            backend=pose_backend,
            selection_source=source,
            ram_mb=pose_ram_mb,
            vram_mb=pose_vram_mb,
        )

    async def refresh_runtime_status(self) -> None:
        self.serial.refresh_connection()
        if self.yolo_only_mode:
            self.ollama_connected = False
            self.ollama_runtime = RuntimeComponentStats(
                requested_mode="disabled",
                effective_device="disabled",
                backend="disabled",
                selection_source="startup-flag",
            )
            return
        await self._refresh_ollama_runtime_stats()

    async def _refresh_ollama_runtime_stats(self) -> None:
        accelerator = expected_accelerator_label()
        source = "default" if self.config.ollama_device_mode == RuntimeConfig().ollama_device_mode else "user"
        runtime = RuntimeComponentStats(
            requested_mode=self.config.ollama_device_mode,
            effective_device=self._expected_ollama_backend_label(accelerator),
            backend=self._expected_ollama_backend_label(accelerator),
            selection_source=source,
        )
        try:
            client = OllamaClient(
                self.config.ollama_base_url,
                min(self.config.ollama_timeout_sec, 15),
                self.config.ollama_device_mode,
            )
            running = await client.running_models()
            current = next(
                (
                    item for item in running
                    if item.get("name") == self.config.ollama_model or item.get("model") == self.config.ollama_model
                ),
                None,
            )
            if current is not None:
                size_vram = int(current.get("size_vram", 0))
                total_size = int(current.get("size", 0))
                backend = accelerator if size_vram > 0 else "cpu"
                runtime.effective_device = backend
                runtime.backend = backend
                runtime.vram_mb = round(size_vram / (1024 * 1024), 2) if size_vram else 0.0
                if total_size > 0:
                    runtime.ram_mb = round(max(total_size - size_vram, 0) / (1024 * 1024), 2)
            self.ollama_connected = True
        except Exception:
            self.ollama_connected = False
        self.ollama_runtime = runtime

    def snapshot(self):
        vision = self.vision.get_snapshot()
        self.state.audience = vision.features
        self.state.servo = self._compute_servo_from_features(vision.features, vision.servo.tracking_source)
        snap = self.state.snapshot()
        snap.stats = get_system_stats("tmp")
        snap.serial_connected = self.serial.connected
        snap.serial_monitor = self.serial.snapshot()
        snap.tts_loaded = self.tts.loaded
        snap.camera_device_id = self.config.camera_device_id
        snap.camera_mode = f"{self.config.camera_width}x{self.config.camera_height}@{self.config.camera_fps}"
        snap.ollama_connected = self.ollama_connected
        snap.playback_progress = self.audio.progress()
        snap.yolo_detect_fps = self.vision.detect_fps()
        snap.yolo_person_runtime = self.yolo_person_runtime
        snap.yolo_pose_runtime = self.yolo_pose_runtime
        snap.tts_runtime = self.tts_runtime
        snap.ollama_runtime = self.ollama_runtime
        return snap

    def send_servo_for_features(self, features, tracking_source: str) -> None:
        servo = self._compute_servo_from_features(features, tracking_source)
        self.state.servo = servo
        if features.track_id is None:
            return
        led_left_pct, led_right_pct = self._compute_led_brightness_from_features(features)
        self.serial.send_servo_command(
            servo.left_deg,
            servo.right_deg,
            mode="track",
            tracking_source=servo.tracking_source,
            led_left_pct=led_left_pct,
            led_right_pct=led_right_pct,
        )

    async def housekeeping_loop(self) -> None:
        while True:
            self.resources.cleanup_temp_audio()
            await asyncio.sleep(30)

    async def runtime_status_loop(self) -> None:
        while True:
            try:
                await self.refresh_runtime_status()
            except Exception as exc:
                self.state.event_log = [f"runtime status refresh error: {exc}", *self.state.event_log][:20]
            await asyncio.sleep(RUNTIME_STATUS_REFRESH_INTERVAL_SEC)

    async def vision_loop(self) -> None:
        while True:
            try:
                self._recover_if_pipeline_stuck()
                self._update_mode_from_vision()
                if self.state.mode == SystemMode.TRACKING:
                    await self._maybe_generate_tracking_line()
                elif self.state.mode == SystemMode.IDLE:
                    await self._maybe_generate_idle_line()
                append_audience_snapshot("tmp/audience.csv", self.snapshot())
            except Exception as exc:
                self.state.set_pipeline_stage(PipelineStage.ERROR, error=str(exc))
                self.state.event_log = [f"vision loop error: {exc}", *self.state.event_log][:20]
                self.last_sentence_finished_at = time.monotonic()
            await asyncio.sleep(max(0.02, 1.0 / max(1, self.config.camera_fps)))

    def _recover_if_pipeline_stuck(self) -> None:
        stage = self.state.pipeline.stage
        elapsed_ms = self.state.pipeline.elapsed_ms
        llm_limit_ms = (self.config.ollama_timeout_sec + 5) * 1000
        tts_limit_ms = self.config.tts_timeout_sec * 1000
        if stage == PipelineStage.LLM and elapsed_ms > llm_limit_ms:
            self.state.set_pipeline_stage(PipelineStage.ERROR, error="LLM timeout watchdog")
            self.state.event_log = ["watchdog reset LLM stage", *self.state.event_log][:20]
            self.last_sentence_finished_at = time.monotonic()
        if stage == PipelineStage.TTS and elapsed_ms > tts_limit_ms:
            self.state.set_pipeline_stage(PipelineStage.ERROR, error="TTS timeout watchdog")
            self.state.event_log = ["watchdog reset TTS stage", *self.state.event_log][:20]
            self.last_sentence_finished_at = time.monotonic()
        if stage == PipelineStage.PLAYBACK and not self.audio.is_playing():
            if self.audio.last_error:
                self.state.set_pipeline_stage(PipelineStage.ERROR, error=self.audio.last_error)
                self.state.event_log = [f"audio error: {self.audio.last_error}", *self.state.event_log][:20]
            else:
                self.state.set_pipeline_stage(PipelineStage.IDLE)
            self.last_sentence_finished_at = time.monotonic()

    def _update_mode_from_vision(self) -> None:
        now = time.monotonic()
        vision = self.vision.get_snapshot()
        features = vision.features
        self.state.audience = features
        self.send_servo_for_features(features, vision.servo.tracking_source)

        if now < self.cooldown_until:
            self.state.set_mode(SystemMode.PURGE_COOLDOWN)
            return

        if features.track_id is None or features.bbox_area_ratio < self.config.lock_bbox_threshold_ratio:
            if self.state.mode == SystemMode.TRACKING and now - self.last_target_seen <= self.config.lost_timeout_ms / 1000:
                self.state.set_mode(SystemMode.RECONNECTING, "Target temporarily lost")
            elif now - self.last_target_seen > self.config.lost_timeout_ms / 1000:
                self.state.set_mode(SystemMode.IDLE)
                self.state.locked_track_id = None
                self.state.sentence_index = 0
                self.lock_started_at = None
            return

        self.last_target_seen = now
        if self.lock_started_at is None:
            self.lock_started_at = now
            self.state.set_mode(SystemMode.ACQUIRING)
            return
        if now - self.lock_started_at < self.config.enter_debounce_ms / 1000:
            self.state.set_mode(SystemMode.ACQUIRING)
            return

        if self.state.mode != SystemMode.TRACKING:
            self.state.set_mode(SystemMode.TRACKING, "Locked target acquired")
            self.state.locked_track_id = features.track_id
            if self.state.sentence_index == 0:
                self.last_sentence_finished_at = now

    def _compute_servo_from_features(self, features, tracking_source: str):
        eye_midpoint_x = features.eye_midpoint[0] if features.eye_midpoint else features.center_x_norm
        servo = compute_servo_angles(
            eye_midpoint_x_norm=eye_midpoint_x,
            bbox_area_ratio=features.bbox_area_ratio,
            left_zero_deg=self.config.servo_left_zero_deg,
            right_zero_deg=self.config.servo_right_zero_deg,
            eye_spacing_cm=self.config.servo_eye_spacing_cm,
            left_limits=(self.config.servo_left_min_deg, self.config.servo_left_max_deg),
            right_limits=(self.config.servo_right_min_deg, self.config.servo_right_max_deg),
        )
        servo.left_deg = self._apply_servo_output_calibration(
            angle=servo.left_deg,
            zero_deg=self.config.servo_left_zero_deg,
            min_deg=self.config.servo_left_min_deg,
            max_deg=self.config.servo_left_max_deg,
            gain=self.config.servo_left_gain,
            trim_deg=self.config.servo_left_trim_deg,
        )
        servo.right_deg = self._apply_servo_output_calibration(
            angle=servo.right_deg,
            zero_deg=self.config.servo_right_zero_deg,
            min_deg=self.config.servo_right_min_deg,
            max_deg=self.config.servo_right_max_deg,
            gain=self.config.servo_right_gain,
            trim_deg=self.config.servo_right_trim_deg,
        )
        servo.tracking_source = tracking_source
        return servo

    def _compute_led_brightness_from_features(self, features) -> tuple[float, float]:
        midpoint_x = features.eye_midpoint[0] if features.eye_midpoint else features.center_x_norm
        midpoint_x = min(max(midpoint_x, 0.0), 1.0)
        if self.config.servo_output_inverted:
            midpoint_x = 1.0 - midpoint_x
        if self.config.led_left_right_inverted:
            midpoint_x = 1.0 - midpoint_x

        brightness_span = self.config.led_max_brightness_pct - self.config.led_min_brightness_pct
        left_pct = self.config.led_min_brightness_pct + ((1.0 - midpoint_x) * brightness_span)
        right_pct = self.config.led_min_brightness_pct + (midpoint_x * brightness_span)

        if self.config.led_brightness_output_inverted:
            brightness_total = self.config.led_min_brightness_pct + self.config.led_max_brightness_pct
            left_pct = brightness_total - left_pct
            right_pct = brightness_total - right_pct

        return round(min(max(left_pct, 0.0), 100.0), 2), round(min(max(right_pct, 0.0), 100.0), 2)

    def _apply_servo_output_calibration(
        self,
        *,
        angle: float,
        zero_deg: float,
        min_deg: float,
        max_deg: float,
        gain: float,
        trim_deg: float,
    ) -> float:
        delta = angle - zero_deg
        if self.config.servo_output_inverted:
            delta = -delta
        calibrated = zero_deg + (delta * gain) + trim_deg
        return round(min(max(calibrated, min_deg), max_deg), 2)

    def _yolo_only_mode_enabled(self) -> bool:
        return self.yolo_only_mode

    async def _maybe_generate_tracking_line(self) -> None:
        if self._yolo_only_mode_enabled():
            return
        if self.audio.is_playing() or self.generation_lock.locked():
            return
        if self.state.sentence_index >= 10:
            self.cooldown_until = time.monotonic() + (self.config.purge_cooldown_ms / 1000)
            self.state.set_mode(SystemMode.PURGE_COOLDOWN, "Sentence limit reached")
            return
        delay = 1.0 if self.state.sentence_index == 0 else 0.05
        if time.monotonic() - self.last_sentence_finished_at < delay:
            return
        await self.generate_tracking_line()

    async def _maybe_generate_idle_line(self) -> None:
        if self._yolo_only_mode_enabled():
            return
        if self.audio.is_playing() or self.generation_lock.locked():
            return
        if time.monotonic() - self.last_idle_line_at < self.config.idle_sentence_interval_ms / 1000:
            return
        await self.generate_idle_line()

    async def generate_tracking_line(self) -> None:
        if self._yolo_only_mode_enabled():
            self.state.set_pipeline_stage(PipelineStage.IDLE)
            return
        async with self.generation_lock:
            sentence_index = self.state.sentence_index + 1
            self.state.active_sentence_index = sentence_index
            self.state.set_pipeline_stage(PipelineStage.LLM)
            vision_snapshot = self.vision.get_snapshot()
            use_person_crop = self.config.llm_use_person_crop and vision_snapshot.person_crop_jpeg is not None
            if self.config.llm_use_person_crop and not use_person_crop:
                self.state.event_log = ["LLM person crop mode requested but no crop was available; falling back to prompt-only mode", *self.state.event_log][:20]
            prompt = self.prompts.build_tracking_prompt(
                sentence_index=sentence_index,
                selected_examples=self.config.tracking_examples_selected,
                audience=self.state.audience,
                event_summary=self._event_summary(),
                reacquired=self.state.mode == SystemMode.RECONNECTING or self.state.audience.actions.returned_after_defocus,
                use_visual_audience=use_person_crop,
            )
            self.state.current_prompt_system = prompt["system"]
            self.state.current_prompt_user = prompt["user"]
            text = await self._generate_with_ollama(
                str(prompt["system"]),
                str(prompt["user"]),
                22,
                required_terms=list(prompt.get("required_terms", [])),
                images=[vision_snapshot.person_crop_jpeg] if use_person_crop and vision_snapshot.person_crop_jpeg else None,
            )
            await self._speak_line(text)
            self.state.sentence_index = sentence_index
            self.state.active_sentence_index = sentence_index
            self.history.append(text)
            self.last_sentence_finished_at = time.monotonic()

    async def generate_idle_line(self) -> None:
        if self._yolo_only_mode_enabled():
            self.state.set_pipeline_stage(PipelineStage.IDLE)
            return
        async with self.generation_lock:
            self.state.set_pipeline_stage(PipelineStage.LLM)
            prompt = self.prompts.build_idle_prompt(
                selected_examples=self.config.idle_examples_selected,
                idle_duration_ms=int((time.monotonic() - self.last_target_seen) * 1000),
            )
            self.state.current_prompt_system = prompt["system"]
            self.state.current_prompt_user = prompt["user"]
            text = await self._generate_with_ollama(
                str(prompt["system"]),
                str(prompt["user"]),
                15,
                required_terms=list(prompt.get("required_terms", [])),
            )
            await self._speak_line(text)
            self.last_idle_line_at = time.monotonic()

    async def _generate_with_ollama(
        self,
        system: str,
        prompt: str,
        limit: int,
        required_terms: list[str] | None = None,
        images: list[bytes] | None = None,
    ) -> str:
        client = OllamaClient(self.config.ollama_base_url, self.config.ollama_timeout_sec, self.config.ollama_device_mode)
        started = time.monotonic()
        text = await self._generate_validated_text(client, system, prompt, limit, required_terms or [], images=images)
        self.state.llm_latency_ms = int((time.monotonic() - started) * 1000)
        self.state.last_llm_output = text
        return text

    async def _generate_validated_text(
        self,
        client: OllamaClient,
        system: str,
        prompt: str,
        limit: int,
        required_terms: list[str],
        images: list[bytes] | None = None,
    ) -> str:
        attempts = [
            {
                "system": system,
                "prompt": prompt,
                "options": {
                    "num_predict": 48,
                    "temperature": 0.55,
                    "top_p": 0.9,
                    "repeat_penalty": 1.15,
                    "stop": ["\n\n", "\n- ", "</s>"],
                },
            },
            {
                "system": system,
                "prompt": (
                    f"{prompt}\n"
                    "重試規則:\n"
                    f"上次輸出不合規或過短，這次必須直接輸出一個可朗讀的完整句子。\n"
                    f"句長 8 到 {limit} 字。\n"
                    "禁止空字串，禁止只輸出標點，禁止解釋。"
                ),
                "options": {
                    "num_predict": 32,
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "repeat_penalty": 1.2,
                    "stop": ["\n\n", "\n- ", "</s>"],
                },
            },
        ]
        last_text = ""
        last_errors: list[str] = []
        for attempt in attempts:
            text = await asyncio.wait_for(
                self._stream_text(client, attempt["system"], attempt["prompt"], attempt["options"], images=images),
                timeout=self.config.ollama_timeout_sec,
            )
            errors = self._validate_output(text, limit, required_terms)
            if not errors:
                return self._normalize_sentence(text, limit)
            last_text = text
            last_errors = errors
        repair_prompt = (
            "把下面這句重寫成繁體中文單句，保留關鍵觀眾特徵或事件，但必須壓到"
            f" {limit} 字內，不能照抄 examples，不能解釋：\n"
            f"{last_text}"
        )
        repaired = await asyncio.wait_for(
            self._stream_text(
                client,
                system,
                repair_prompt,
                {
                    "num_predict": 24,
                    "temperature": 0.2,
                    "top_p": 0.75,
                    "repeat_penalty": 1.2,
                    "stop": ["\n\n", "\n- ", "</s>"],
                },
                images=images,
            ),
            timeout=self.config.ollama_timeout_sec,
        )
        errors = self._validate_output(repaired, limit, required_terms)
        if not errors:
            return self._normalize_sentence(repaired, limit)
        compressed = await asyncio.wait_for(
            self._stream_text(
                client,
                system,
                (
                    f"把這句再縮短成 10 到 {limit} 字的單句繁中台詞，保留一個最重要的觀眾特徵或事件即可：\n"
                    f"{repaired or last_text}"
                ),
                {
                    "num_predict": 18,
                    "temperature": 0.1,
                    "top_p": 0.65,
                    "repeat_penalty": 1.15,
                    "stop": ["\n\n", "\n- ", "</s>"],
                },
                images=images,
            ),
            timeout=self.config.ollama_timeout_sec,
        )
        errors = self._validate_output(compressed, limit, required_terms)
        if not errors:
            return self._normalize_sentence(compressed, limit)
        if self._contains_required_terms(last_text, required_terms):
            return self._normalize_sentence(last_text, limit)
        if self._contains_required_terms(repaired, required_terms):
            return self._normalize_sentence(repaired, limit)
        if self._contains_required_terms(compressed, required_terms):
            return self._normalize_sentence(compressed, limit)
        raise RuntimeError(
            f"LLM returned invalid output: {last_errors}; text={last_text!r}"
        )

    def _validate_output(self, text: str, limit: int, required_terms: list[str]) -> list[str]:
        errors = validate_generated_sentence(text, limit)
        if required_terms and not self._contains_required_terms(text, required_terms):
            errors.append(f"must mention one of required terms: {required_terms}")
        return errors

    def _contains_required_terms(self, text: str, required_terms: list[str]) -> bool:
        return any(term and term in text for term in required_terms)

    async def _stream_text(
        self,
        client: OllamaClient,
        system: str,
        prompt: str,
        options: dict,
        images: list[bytes] | None = None,
    ) -> str:
        tokens: list[str] = []
        async for token in client.generate_stream(
            self.config.ollama_model,
            system,
            prompt,
            options=options,
            images=images,
        ):
            tokens.append(token)
        return "".join(tokens).strip()

    def _normalize_sentence(self, text: str, limit: int) -> str:
        cleaned = (
            text.strip()
            .replace("\n", "")
            .replace("「", "")
            .replace("」", "")
            .replace('"', "")
        )
        cleaned = cleaned.strip(" ,.，。!?！？")
        if len(cleaned) <= limit:
            return cleaned if cleaned.endswith(("。", "！", "？")) else f"{cleaned[: max(0, limit - 1)]}。"
        for mark in ["。", "，", "！", "？", ",", "."]:
            if mark in cleaned[:limit]:
                segment = cleaned[: cleaned[:limit].rfind(mark)]
                if segment and len(segment.strip(" ,.，。!?！？")) >= 4:
                    return segment[: max(0, limit - 1)] + "。"
        trimmed = cleaned[: max(0, limit - 1)].strip(" ,.，。!?！？")
        if len(trimmed) < 4:
            raise RuntimeError(f"LLM returned too-short sentence after normalization: {text!r}")
        return trimmed + "。"

    async def _speak_line(self, text: str) -> None:
        self.state.set_pipeline_stage(PipelineStage.TTS)
        started = time.monotonic()
        profile = self._current_tts_model_profile()
        clone_voice_active = self.config.tts_clone_voice_enabled and profile.supports_voice_clone
        if clone_voice_active:
            reference_pair = await self._select_tts_reference_pair(text)
            self._apply_tts_reference_pair(reference_pair)
        else:
            self.state.tts_reference_raw = None
            self.state.tts_reference_pair = None
            self.state.tts_reference_audio_path = None
            self.state.tts_reference_text_path = None
        if self.config.tts_emotion_enabled and profile.supports_structured_emotion:
            emotion_raw, emotion = await self._classify_tts_emotion(text)
            tts_text = self._apply_tts_emotion(text, emotion)
        else:
            emotion_raw = None
            emotion = None
            tts_text = text
        self.state.tts_emotion_raw = emotion_raw
        self.state.tts_emotion_applied = emotion
        self.state.tts_emotion_used = bool(emotion)
        self.state.tts_input_text = tts_text
        try:
            output = await asyncio.wait_for(
                asyncio.to_thread(self.tts.synthesize, tts_text, "tmp/generated.wav"),
                timeout=self.config.tts_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self.state.tts_latency_ms = elapsed_ms
            raise RuntimeError(f"TTS timeout after {elapsed_ms} ms (limit {self.config.tts_timeout_sec * 1000} ms)") from exc
        self.state.tts_latency_ms = int((time.monotonic() - started) * 1000)
        self.state.set_pipeline_stage(PipelineStage.PLAYBACK)
        self.audio.set_routed_playback(self.config.tts_route_via_virtual_device)
        self.audio.set_output_device(self.config.audio_output_device)
        self.audio.play(output, volume=self.config.tts_output_volume)
        self.state.last_spoken_text = text

    async def _classify_tts_emotion(self, text: str) -> tuple[str, str]:
        profile = self._current_tts_model_profile()
        supported_tags = profile.emotion_tags
        client = OllamaClient(self.config.ollama_base_url, min(self.config.ollama_timeout_sec, 15), self.config.ollama_device_mode)
        prompt = (
            "你是 TTS 情緒標記分類器。"
            f"以下是 {profile.emotion_prompt_label}。"
            f"你必須只從這個清單選一個最適合該句子的標記並直接輸出，不要解釋：{', '.join(supported_tags)}。\n"
            "每句都一定要有情緒，不可輸出 none、neutral、tone、very、extremely 或其他清單外文字。\n"
            "只可輸出清單中的單一精確 tag，不可加任何修飾詞。\n"
            f"句子：{text}"
        )
        try:
            raw = await asyncio.wait_for(
                self._stream_text(
                    client,
                    "只輸出一個標記名稱。",
                    prompt,
                    {
                        "num_predict": 12,
                        "temperature": 0,
                        "top_p": 0.2,
                        "repeat_penalty": 1.0,
                        "stop": ["\n", "。", ","],
                    },
                ),
                timeout=min(self.config.ollama_timeout_sec, 15),
            )
        except Exception:
            fallback = self._fallback_tts_emotion(text, profile=profile)
            return fallback, fallback
        candidate = self._clean_tts_emotion(raw) or DEFAULT_TTS_EMOTION
        normalized = self._normalize_tts_emotion(candidate, profile=profile)
        if normalized is None:
            normalized = self._fallback_tts_emotion(text, profile=profile)
        return candidate, normalized

    def _clean_tts_emotion(self, raw: str) -> str | None:
        cleaned = raw.strip().strip("()[]{}<>\"'`.,，。!！？").lower().replace("_", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned or None

    def _normalize_tts_emotion(self, raw: str, *, profile=None) -> str | None:
        cleaned = self._clean_tts_emotion(raw)
        active_profile = profile or self._current_tts_model_profile()
        return cleaned if cleaned in active_profile.emotion_tags else None

    def _fallback_tts_emotion(self, text: str, *, profile=None) -> str:
        active_profile = profile or self._current_tts_model_profile()
        lowered = text.lower()
        if any(token in lowered for token in ("怒", "氣", "恨", "煩", "滾", "閉嘴", "討厭")):
            return "angry"
        if any(token in lowered for token in ("哭", "難過", "悲", "孤獨", "寂寞", "抱歉", "遺憾", "失望")):
            return "sad"
        if any(token in lowered for token in ("怕", "危險", "小心", "糟", "怎麼辦")):
            return "worried" if "worried" in active_profile.emotion_tags else "nervous"
        if any(token in lowered for token in ("?", "？", "嗎", "呢", "為什麼", "怎樣")):
            return "curious"
        if any(token in lowered for token in ("!", "！", "太棒", "好耶", "快", "立刻")):
            return "excited"
        return DEFAULT_TTS_EMOTION if DEFAULT_TTS_EMOTION in active_profile.emotion_tags else active_profile.emotion_tags[0]

    def _apply_tts_emotion(self, text: str, emotion: str) -> str:
        return self._current_tts_model_profile().format_emotion_text(text, emotion)

    def _current_tts_model_profile(self):
        return getattr(self.tts, "model_profile", None) or resolve_tts_model_profile(self.config.tts_model_path) or DEFAULT_TTS_MODEL_PROFILE

    def _startup_tts_reference_pair(self) -> ReferencePair:
        if not self.config.tts_clone_voice_enabled or self.config.tts_reference_mode == "fixed":
            return build_fixed_reference_pair(self.config.tts_ref_audio_path, self.config.tts_ref_text_path)
        return load_emotional_reference_pairs()[0]

    def _benchmark_tts_reference_pair(self) -> ReferencePair:
        fixed_pair = build_fixed_reference_pair(self.config.tts_ref_audio_path, self.config.tts_ref_text_path)
        fixed_audio = Path(fixed_pair.audio_path)
        fixed_text = Path(fixed_pair.text_path)
        if fixed_audio.exists() and fixed_text.exists() and fixed_audio.suffix.lower() == ".wav":
            return fixed_pair

        bundled_pair = build_fixed_reference_pair("resource/voice/ref-voice3.wav", "resource/voice/transcript3.txt")
        if Path(bundled_pair.audio_path).exists() and Path(bundled_pair.text_path).exists():
            return bundled_pair

        return self._startup_tts_reference_pair()

    async def _select_tts_reference_pair(self, text: str) -> ReferencePair:
        mode = (self.config.tts_reference_mode or "fixed").strip().lower()
        if mode == "fixed":
            self.state.tts_reference_raw = None
            return build_fixed_reference_pair(self.config.tts_ref_audio_path, self.config.tts_ref_text_path)
        if mode == "random":
            self.state.tts_reference_raw = None
            return choose_random_emotional_reference_pair(self._random)
        raw, pair = await self._classify_tts_reference_pair(text)
        previous_pair = self._last_tts_reference_pair_key
        if previous_pair and pair.key == previous_pair:
            retry_raw, retry_pair = await self._classify_tts_reference_pair(text, excluded={previous_pair})
            if retry_pair.key != previous_pair:
                raw, pair = retry_raw, retry_pair
            else:
                pair_map = emotional_reference_pair_map()
                pair = self._fallback_tts_reference_pair(text, pair_map, excluded={previous_pair})
        self.state.tts_reference_raw = raw
        return pair

    async def _classify_tts_reference_pair(
        self,
        text: str,
        *,
        excluded: set[str] | None = None,
    ) -> tuple[str, ReferencePair]:
        pair_map = emotional_reference_pair_map()
        banned = excluded or set()
        choices = [tag for tag, pair_name in REFERENCE_PAIR_TAG_TO_NAME.items() if pair_name in pair_map and pair_name not in banned]
        if not choices:
            fallback_pair = self._fallback_tts_reference_pair(text, pair_map, excluded=banned)
            return REFERENCE_PAIR_NAME_TO_TAG.get(fallback_pair.key, fallback_pair.key), fallback_pair
        client = OllamaClient(self.config.ollama_base_url, min(self.config.ollama_timeout_sec, 15), self.config.ollama_device_mode)
        disallow_text = ""
        if banned:
            disallow_tags = [REFERENCE_PAIR_NAME_TO_TAG.get(name, name) for name in banned]
            disallow_text = f"上一句剛用過：{', '.join(disallow_tags)}。這次不可輸出這些標籤。\n"
        prompt = (
            "你是 reference voice / transcript 情緒分類器。"
            f"你必須只從這個清單挑一個最適合該句子情緒與語氣的簡短情緒標籤，直接輸出標籤本身，不要解釋：{', '.join(choices)}。\n"
            f"{disallow_text}"
            "不可輸出清單外內容，不可加標點，不可加前後綴。\n"
            f"句子：{text}"
        )
        try:
            raw = await asyncio.wait_for(
                self._stream_text(
                    client,
                    "只輸出一個情緒標籤。",
                    prompt,
                    {
                        "num_predict": 24,
                        "temperature": 0,
                        "top_p": 0.2,
                        "repeat_penalty": 1.0,
                        "stop": ["\n", "。", ","],
                    },
                ),
                timeout=min(self.config.ollama_timeout_sec, 15),
            )
        except Exception:
            fallback_pair = self._fallback_tts_reference_pair(text, pair_map, excluded=banned)
            return REFERENCE_PAIR_NAME_TO_TAG.get(fallback_pair.key, fallback_pair.key), fallback_pair
        normalized_tag = self._normalize_tts_reference_tag(raw)
        if normalized_tag is None:
            fallback_pair = self._fallback_tts_reference_pair(text, pair_map, excluded=banned)
            return self._clean_tts_reference_label(raw), fallback_pair
        pair_name = REFERENCE_PAIR_TAG_TO_NAME[normalized_tag]
        pair = pair_map.get(pair_name)
        if pair is None:
            fallback_pair = self._fallback_tts_reference_pair(text, pair_map, excluded=banned)
            return normalized_tag, fallback_pair
        return normalized_tag, pair

    def _clean_tts_reference_label(self, raw: str) -> str:
        return "".join(raw.strip().strip("()[]{}<>\"'`.,，。!！？").split())

    def _normalize_tts_reference_tag(self, raw: str) -> str | None:
        cleaned = self._clean_tts_reference_label(raw)
        if not cleaned:
            return None
        if cleaned in REFERENCE_PAIR_TAG_TO_NAME:
            return cleaned
        for pair_name, tag in REFERENCE_PAIR_NAME_TO_TAG.items():
            if self._clean_tts_reference_label(pair_name) == cleaned:
                return tag
        aliases = {
            "感激與愧疚": "感激愧疚",
            "刻薄的質問與探詢": "刻薄質問",
            "冷靜且專業": "冷靜專業",
            "情緒激動且不可置信": "激動不信",
            "震驚與崩潰": "震驚崩潰",
            "懷疑且嚴厲": "懷疑嚴厲",
            "極度的悲慟和自責與絕望": "悲慟自責絕望",
        }
        return aliases.get(cleaned)

    def _fallback_tts_reference_pair(
        self,
        text: str,
        pair_map: dict[str, ReferencePair],
        *,
        excluded: set[str] | None = None,
    ) -> ReferencePair:
        banned = excluded or set()
        heuristics = [
            (("謝", "感激", "抱歉", "愧疚"), "感激與愧疚"),
            (("質問", "逼問", "探詢", "你到底"), "刻薄的質問與探詢"),
            (("冷靜", "分析", "專業", "流程"), "冷靜且專業"),
            (("不可置信", "怎麼可能", "太誇張", "激動"), "情緒激動且不可置信"),
            (("震驚", "崩潰", "完了", "不可能"), "震驚與崩潰"),
            (("懷疑", "嚴厲", "老實說", "最好"), "懷疑且嚴厲"),
            (("悲", "自責", "絕望", "痛苦"), "極度的悲慟和自責與絕望"),
        ]
        for tokens, key in heuristics:
            if any(token in text for token in tokens) and key in pair_map and key not in banned:
                return pair_map[key]
        if "冷靜且專業" in pair_map and "冷靜且專業" not in banned:
            return pair_map["冷靜且專業"]
        for pair in pair_map.values():
            if pair.key not in banned:
                return pair
        return next(iter(pair_map.values()))

    def _apply_tts_reference_pair(self, pair: ReferencePair) -> None:
        if hasattr(self.tts, "set_reference_paths"):
            self.tts.set_reference_paths(pair.audio_path, pair.text_path)
        else:
            self.tts.ref_audio_path = pair.audio_path
            self.tts.ref_text_path = pair.text_path
        self._last_tts_reference_pair_key = pair.key
        self.state.tts_reference_pair = pair.key
        self.state.tts_reference_audio_path = pair.audio_path
        self.state.tts_reference_text_path = pair.text_path

    def _event_summary(self) -> str:
        actions = self.state.audience.actions
        labels = {
            "wave": "揮手",
            "crouch": "蹲下或突然降低高度",
            "defocus": "過近失焦",
            "moving_away": "正在遠離",
            "approaching": "正在貼近",
            "returned_after_defocus": "失焦後重新進入視線",
        }
        names = [labels.get(name, name) for name, value in actions.model_dump().items() if value]
        return ", ".join(names) if names else "無"


brain = Brain()


def should_prepare_models() -> bool:
    return os.getenv("MOMO_SKIP_MODEL_BOOTSTRAP") != "1"


def should_skip_tts_benchmark() -> bool:
    return os.getenv("MOMO_SKIP_TTS_BENCHMARK") == "1"


async def build_apply_checks(payload: dict, config: RuntimeConfig) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    changed = set(payload.keys())

    if changed & {"camera_source", "camera_device_id", "camera_width", "camera_height", "camera_fps", "camera_mirror_preview", "camera_flip_vertical"}:
        orientation = f"hflip={'on' if config.camera_mirror_preview else 'off'}, vflip={'on' if config.camera_flip_vertical else 'off'}"
        if config.camera_source == "browser":
            checks.append(
                {
                    "component": "vision",
                    "status": "ok",
                    "message": (
                        f"Browser camera config staged: {config.camera_width}x{config.camera_height}@{config.camera_fps}, "
                        f"{orientation}. Applies on next uploaded frame."
                    ),
                }
            )
        else:
            checks.append(
                {
                    "component": "vision",
                    "status": "ok",
                    "message": (
                        f"Backend capture reconfigured to device {config.camera_device_id} at "
                        f"{config.camera_width}x{config.camera_height}@{config.camera_fps}, {orientation}."
                    ),
                }
            )

    if changed & {"tracking_examples_selected", "idle_examples_selected"}:
        try:
            brain.prompts.build_tracking_prompt(
                sentence_index=1,
                selected_examples=config.tracking_examples_selected,
                audience=brain.state.audience,
                event_summary="無",
                reacquired=False,
            )
            brain.prompts.build_idle_prompt(config.idle_examples_selected, 0)
            checks.append({"component": "prompting", "status": "ok", "message": "Prompt examples reloaded successfully."})
        except Exception as exc:
            checks.append({"component": "prompting", "status": "error", "message": f"Prompt reload failed: {exc}"})

    if changed & {"yolo_model_path", "yolo_pose_model_path", "yolo_device_mode"}:
        checks.append({"component": "vision-model", "status": "ok", "message": f"YOLO model paths updated and verified. Device mode is {config.yolo_device_mode}."})

    if changed & {"ollama_base_url", "ollama_model", "ollama_device_mode", "ollama_timeout_sec", "llm_use_person_crop"}:
        client = OllamaClient(config.ollama_base_url, min(config.ollama_timeout_sec, 15), config.ollama_device_mode)
        try:
            models = await client.list_models()
            if config.ollama_model in models:
                mode = "person-crop image mode" if config.llm_use_person_crop else "prompt-only mode"
                checks.append({"component": "llm", "status": "ok", "message": f"Ollama reachable and model {config.ollama_model} is available. LLM is in {mode}. Device mode is {config.ollama_device_mode}."})
            else:
                checks.append({"component": "llm", "status": "error", "message": f"Ollama reachable but model {config.ollama_model} was not found."})
        except Exception as exc:
            checks.append({"component": "llm", "status": "error", "message": f"Ollama check failed: {exc}"})

    if changed & {"tts_model_path", "tts_kokoro_voice", "tts_device_mode", "tts_emotion_enabled", "tts_clone_voice_enabled", "tts_reference_mode", "tts_ref_audio_path", "tts_ref_text_path", "tts_timeout_sec"}:
        errors: list[str] = []
        model_path = Path(config.tts_model_path)
        profile = resolve_tts_model_profile(config.tts_model_path)
        clone_voice_active = config.tts_clone_voice_enabled and profile.supports_voice_clone
        emotion_active = config.tts_emotion_enabled and profile.supports_structured_emotion
        if not model_path.exists():
            errors.append(f"missing model path: {model_path}")
        if clone_voice_active:
            if config.tts_reference_mode == "fixed":
                ref_pair = build_fixed_reference_pair(config.tts_ref_audio_path, config.tts_ref_text_path)
                if not Path(ref_pair.audio_path).exists():
                    errors.append(f"missing ref audio: {ref_pair.audio_path}")
                if not Path(ref_pair.text_path).exists():
                    errors.append(f"missing ref transcript: {ref_pair.text_path}")
            else:
                try:
                    pair_count = len(load_emotional_reference_pairs())
                except Exception as exc:
                    errors.append(str(exc))
                else:
                    mode = "clone voice" if clone_voice_active else "normal TTS"
                    emotion_mode = "emotion on" if emotion_active else "emotion off"
                    checks.append(
                        {
                            "component": "tts",
                            "status": "ok",
                            "message": (
                                f"TTS ready in {mode}, {emotion_mode}. Emotional reference library ready with {pair_count} pairs. "
                                f"Reference mode is {config.tts_reference_mode}. Timeout set to {config.tts_timeout_sec * 1000} ms."
                            ),
                        }
                    )
        if errors:
            checks.append({"component": "tts", "status": "error", "message": "; ".join(errors)})
        elif config.tts_reference_mode == "fixed" or not clone_voice_active:
            mode = "clone voice" if clone_voice_active else "normal TTS"
            emotion_mode = "emotion on" if emotion_active else "emotion off"
            detail_suffix = ""
            if config.tts_clone_voice_enabled and not profile.supports_voice_clone:
                detail_suffix = f" {profile.display_name} does not support clone voice, so reference audio is ignored."
            elif config.tts_emotion_enabled and not profile.supports_structured_emotion:
                detail_suffix = f" {profile.display_name} does not use structured emotion tags."
            checks.append(
                {
                    "component": "tts",
                    "status": "ok",
                    "message": (
                        f"TTS ready in {mode}, {emotion_mode}, ref mode {config.tts_reference_mode}, "
                        f"device {config.tts_device_mode}. Timeout set to {config.tts_timeout_sec * 1000} ms."
                        f"{detail_suffix}"
                        + (
                            f" Kokoro voice is {config.tts_kokoro_voice}."
                            if profile.runtime_family == "kokoro"
                            else ""
                        )
                    ),
                }
            )

    if changed & {"audio_output_device", "tts_output_volume", "tts_route_via_virtual_device"}:
        available_ids = {item["id"] for item in AudioPlayer.list_output_devices()}
        if config.audio_output_device != "default" and config.audio_output_device not in available_ids:
            checks.append({"component": "audio", "status": "error", "message": f"Audio output device {config.audio_output_device} was not found."})
        elif config.tts_route_via_virtual_device and config.audio_output_device == "default":
            checks.append(
                {
                    "component": "audio",
                    "status": "warning",
                    "message": (
                        "Virtual TTS routing is enabled, but Audio Output is still default. "
                        "Choose an explicit virtual device on macOS or Windows for deterministic routing into VCV Rack 2."
                    ),
                }
            )
        elif config.tts_route_via_virtual_device:
            checks.append(
                {
                    "component": "audio",
                    "status": "ok",
                    "message": (
                        f"Audio output routed through device {config.audio_output_device}. "
                        "Native default playback is disabled so downstream virtual-device processing stays in the chain."
                    ),
                }
            )
        else:
            checks.append({"component": "audio", "status": "ok", "message": f"Audio output set to {config.audio_output_device}."})

    if changed & {"serial_port", "serial_baud_rate"}:
        if config.serial_port == "auto":
            message = f"Serial reconfigured to auto detect at {config.serial_baud_rate} baud."
            status = "ok" if brain.serial.connected else "warning"
        else:
            message = f"Serial reconfigured to {config.serial_port} at {config.serial_baud_rate} baud."
            status = "ok" if brain.serial.connected else "warning"
            if not brain.serial.connected:
                message += " Port not connected right now."
        checks.append({"component": "serial", "status": status, "message": message})

    if changed & {
        "servo_left_zero_deg",
        "servo_right_zero_deg",
        "servo_output_inverted",
        "servo_left_trim_deg",
        "servo_right_trim_deg",
        "servo_left_gain",
        "servo_right_gain",
        "servo_eye_spacing_cm",
        "servo_left_min_deg",
        "servo_left_max_deg",
        "servo_right_min_deg",
        "servo_right_max_deg",
        "servo_smoothing_alpha",
        "servo_max_speed_deg_per_sec",
    }:
        checks.append({"component": "servo", "status": "ok", "message": "Servo math config updated and will apply on next tracking update."})

    if changed & {
        "lock_bbox_threshold_ratio",
        "unlock_bbox_threshold_ratio",
        "enter_debounce_ms",
        "exit_debounce_ms",
        "lost_timeout_ms",
        "eye_loss_timeout_ms",
        "defocus_bbox_threshold_ratio",
        "focus_score_threshold",
        "feature_match_threshold",
        "wave_detection_enabled",
        "crouch_detection_enabled",
        "wave_window_ms",
        "crouch_delta_threshold",
    }:
        checks.append({"component": "vision-rules", "status": "ok", "message": "Vision thresholds and action rules updated."})

    if not checks:
        checks.append({"component": "config", "status": "ok", "message": "No runtime changes detected."})
    return checks


@asynccontextmanager
async def lifespan(_: FastAPI):
    restore_signal_handlers = install_shutdown_signal_bridge()
    startup_interrupted = False
    try:
        try:
            await brain.start()
        except BenchmarkShutdownRequested:
            if not shutdown_requested():
                raise
            startup_interrupted = True
        yield
    finally:
        request_shutdown()
        try:
            if not startup_interrupted:
                await brain.stop()
        finally:
            restore_signal_handlers()
            clear_shutdown_request()


app = FastAPI(title="Momo Brain", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
async def get_status():
    return brain.snapshot()


@app.get("/api/config")
async def get_config():
    return {"config": brain.config, "fields": build_field_catalog(brain.config)}


@app.post("/api/config", response_model=ConfigUpdateResponse)
async def update_config(payload: dict):
    try:
        merged = merge_config(brain.config, payload)
    except ValueError as exc:
        return ConfigUpdateResponse(
            applied_config=brain.config,
            validation_errors=[str(exc)],
            effective_changes=[],
            apply_checks=[{"component": "config", "status": "error", "message": "Payload merge failed."}],
            requires_pipeline_restart=False,
        )
    errors = validate_runtime_config(merged)
    if errors:
        return ConfigUpdateResponse(
            applied_config=brain.config,
            validation_errors=errors,
            effective_changes=[],
            apply_checks=[{"component": "config", "status": "error", "message": "Validation failed. Nothing was applied."}],
            requires_pipeline_restart=False,
        )
    if should_prepare_models():
        try:
            await asyncio.to_thread(ensure_runtime_models, merged)
        except Exception as exc:
            return ConfigUpdateResponse(
                applied_config=brain.config,
                validation_errors=[str(exc)],
                effective_changes=[],
                apply_checks=[{"component": "model-download", "status": "error", "message": f"Model preparation failed: {exc}"}],
                requires_pipeline_restart=False,
            )
    changed = [key for key, value in payload.items() if getattr(brain.config, key) != value]
    changed_keys = set(changed)
    brain.config = merged
    previous_serial = brain.serial
    previous_serial.close()
    brain.serial = ESP32Link(brain.config.serial_port, brain.config.serial_baud_rate)
    if changed_keys & {
        "tts_model_path",
        "tts_kokoro_voice",
        "tts_device_mode",
        "tts_clone_voice_enabled",
        "tts_ref_audio_path",
        "tts_ref_text_path",
    }:
        if "tts_device_mode" in changed_keys:
            requested_mode = str(payload.get("tts_device_mode", brain.config.tts_device_mode))
            source = "user" if requested_mode != RuntimeConfig().tts_device_mode else "default"
        else:
            source = brain.tts_runtime.selection_source or "default"
        brain._replace_tts_runtime(brain._select_tts_runtime(selection_source=source))
    brain.vision.reconfigure(brain.config)
    if changed_keys & {"yolo_model_path", "yolo_pose_model_path", "yolo_device_mode"}:
        try:
            await asyncio.to_thread(brain._refresh_vision_runtime_stats)
        except Exception as exc:
            brain.state.event_log = [f"YOLO warmup failed after config change: {exc}", *brain.state.event_log][:20]
    if changed_keys & {"ollama_base_url", "ollama_model", "ollama_device_mode"}:
        await brain.refresh_runtime_status()
    apply_checks = await build_apply_checks(payload, brain.config)
    return ConfigUpdateResponse(
        applied_config=brain.config,
        validation_errors=[],
        effective_changes=changed,
        apply_checks=apply_checks,
        requires_pipeline_restart=False,
    )


@app.get("/api/cameras")
async def get_cameras():
    return brain.vision.list_cameras()


@app.get("/api/camera/frame.jpg")
async def get_camera_frame():
    frame = brain.vision.get_snapshot().frame_jpeg
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    return Response(content=frame, media_type="image/jpeg")


@app.post("/api/camera/frame")
async def post_camera_frame(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty frame")
    state = brain.vision.submit_jpeg_frame(body)
    if brain.config.camera_source == "browser":
        brain.send_servo_for_features(state.features, state.servo.tracking_source)
    return {
        "track_id": state.features.track_id,
        "eye_confidence": state.features.eye_confidence,
        "tracking_source": state.servo.tracking_source,
    }


@app.get("/api/serial/ports")
async def get_serial_ports():
    return ESP32Link.list_ports()


@app.get("/api/audio/devices")
async def get_audio_devices():
    return AudioPlayer.list_output_devices()


@app.get("/api/ollama/models")
async def get_ollama_models():
    if brain.yolo_only_mode:
        return {"models": [brain.config.ollama_model]}
    client = OllamaClient(brain.config.ollama_base_url, brain.config.ollama_timeout_sec, brain.config.ollama_device_mode)
    try:
        models = await client.list_models()
    except Exception:
        models = [brain.config.ollama_model]
    return {"models": models}


@app.post("/api/control/recenter-servos")
async def recenter_servos():
    payload = brain.serial.send_servo_command(brain.config.servo_left_zero_deg, brain.config.servo_right_zero_deg, mode="idle_scan", tracking_source="manual")
    return {"command": payload}


@app.post("/api/control/simulate-track")
async def simulate_track(payload: dict):
    sentence_index = int(payload.get("sentence_index", 1))
    if not 1 <= sentence_index <= 10:
        raise HTTPException(status_code=400, detail="sentence_index must be between 1 and 10")
    await brain.generate_tracking_line()
    return brain.snapshot()


@app.post("/api/control/simulate-pipeline")
async def simulate_pipeline(payload: dict):
    sentence_index = int(payload.get("sentence_index", 1))
    if sentence_index > 1:
        brain.state.sentence_index = sentence_index - 1
    if "top_color" in payload:
        brain.state.audience.top_color = str(payload["top_color"])
    if "bottom_color" in payload:
        brain.state.audience.bottom_color = str(payload["bottom_color"])
    if "height_class" in payload:
        brain.state.audience.height_class = str(payload["height_class"])
    if "build_class" in payload:
        brain.state.audience.build_class = str(payload["build_class"])
    if "distance_class" in payload:
        brain.state.audience.distance_class = str(payload["distance_class"])
    await brain.generate_tracking_line()
    return {"snapshot": brain.snapshot()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Momo backend.")
    parser.add_argument("--host", default=os.getenv("MOMO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MOMO_PORT", "8000")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Start in vision-only mode. Skip Ollama and TTS loading, warmup, and runtime use.",
    )
    parser.add_argument(
        "--skip-tts-benchmark",
        action="store_true",
        help="Skip startup TTS benchmark auto-selection and use the configured TTS device mode directly.",
    )
    args = parser.parse_args(argv)
    if args.yolo_only:
        os.environ["MOMO_YOLO_ONLY"] = "1"
    if args.skip_tts_benchmark:
        os.environ["MOMO_SKIP_TTS_BENCHMARK"] = "1"
    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main(sys.argv[1:])

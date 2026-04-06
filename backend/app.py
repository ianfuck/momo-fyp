from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.audio.player import AudioPlayer
from backend.config import build_field_catalog, merge_config, validate_runtime_config
from backend.device_utils import expected_accelerator_label, expected_tts_backend_label, expected_vision_backend_label
from backend.llm.ollama_client import OllamaClient
from backend.model_manager import ensure_runtime_models
from backend.prompting.prompt_builder import PromptBuilder, validate_generated_sentence
from backend.resource_manager import ResourceManager
from backend.serial.esp32_link import ESP32Link
from backend.servo.geometry import compute_servo_angles
from backend.state_machine import RuntimeState
from backend.storage.csv_logger import append_audience_snapshot
from backend.telemetry.system_stats import get_system_stats
from backend.tts.qwen_clone import FishCloneTTS
from backend.types import ConfigUpdateResponse, PipelineStage, RuntimeConfig, SystemMode
from backend.vision.runtime import VisionRuntime

TTS_EMOTION_TAGS = (
    "happy",
    "sad",
    "angry",
    "excited",
    "calm",
    "nervous",
    "confident",
    "surprised",
    "satisfied",
    "delighted",
    "scared",
    "worried",
    "upset",
    "frustrated",
    "depressed",
    "empathetic",
    "embarrassed",
    "disgusted",
    "moved",
    "proud",
    "relaxed",
    "grateful",
    "curious",
    "sarcastic",
    "disdainful",
    "unhappy",
    "anxious",
    "hysterical",
    "indifferent",
    "uncertain",
    "doubtful",
    "confused",
    "disappointed",
    "regretful",
    "guilty",
    "ashamed",
    "jealous",
    "envious",
    "hopeful",
    "optimistic",
    "pessimistic",
    "nostalgic",
    "lonely",
    "bored",
    "contemptuous",
    "sympathetic",
    "compassionate",
    "determined",
    "resigned",
)
DEFAULT_TTS_EMOTION = "confident"


class Brain:
    def __init__(self) -> None:
        self.config = RuntimeConfig()
        self.state = RuntimeState()
        self.prompts = PromptBuilder("resource/md/system-persona_tracking.md", "resource/md/system-persona_idle.md")
        self.resources = ResourceManager("tmp")
        self.audio = AudioPlayer()
        self.tts = FishCloneTTS(
            self.config.tts_model_path,
            self.config.tts_ref_audio_path,
            self.config.tts_ref_text_path,
            clone_voice_enabled=self.config.tts_clone_voice_enabled,
        )
        self.serial = ESP32Link(self.config.serial_port, self.config.serial_baud_rate)
        self.vision = VisionRuntime(self.config)
        self.history: deque[str] = deque(maxlen=10)
        self.last_target_seen = 0.0
        self.lock_started_at: float | None = None
        self.last_idle_line_at = 0.0
        self.last_sentence_finished_at = 0.0
        self.cooldown_until = 0.0
        self.generation_lock = asyncio.Lock()
        self.background_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        if should_prepare_models():
            await asyncio.to_thread(self._prepare_runtime_models)
        await self._print_startup_diagnostics()
        self.vision.start()
        self.background_tasks = [
            asyncio.create_task(self.vision_loop()),
            asyncio.create_task(self.housekeeping_loop()),
        ]

    async def stop(self) -> None:
        for task in self.background_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.vision.stop()

    def _prepare_runtime_models(self) -> None:
        ensure_runtime_models(self.config)
        self.tts = FishCloneTTS(
            self.config.tts_model_path,
            self.config.tts_ref_audio_path,
            self.config.tts_ref_text_path,
            clone_voice_enabled=self.config.tts_clone_voice_enabled,
        )
        try:
            self.tts.preload()
        except Exception as exc:
            self.state.event_log = [f"TTS preload failed: {exc}", *self.state.event_log][:20]
        self.vision = VisionRuntime(self.config)

    async def _print_startup_diagnostics(self) -> None:
        for line in await self._collect_startup_diagnostics():
            print(line, flush=True)

    async def _collect_startup_diagnostics(self) -> list[str]:
        expected = expected_accelerator_label()
        tts_expected = expected_tts_backend_label()
        vision_expected = expected_vision_backend_label()
        lines = [f"[startup] expected_accelerator={expected}", f"[startup] expected_vision_backend={vision_expected}"]
        try:
            person_backend = await asyncio.to_thread(self.vision.detector.warmup)
            pose_backend = await asyncio.to_thread(self.vision.pose.warmup)
            lines.append(
                f"[startup] yolo person={person_backend} pose={pose_backend} target={vision_expected} ok={person_backend == vision_expected and pose_backend == vision_expected}"
            )
        except Exception as exc:
            lines.append(f"[startup] yolo error={exc}")

        lines.append(
            f"[startup] tts backend={self.tts.device_backend} target={tts_expected} loaded={self.tts.loaded} ok={self.tts.device_backend == tts_expected}"
        )

        try:
            ollama = OllamaClient(self.config.ollama_base_url, min(self.config.ollama_timeout_sec, 30))
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
            backend = expected if size_vram > 0 else "cpu"
            lines.append(
                f"[startup] ollama backend={backend} target={expected} size_vram={size_vram} ok={backend == expected}"
            )
        except Exception as exc:
            lines.append(f"[startup] ollama error={exc}")
        return lines

    def snapshot(self):
        vision = self.vision.get_snapshot()
        self.state.audience = vision.features
        self.state.servo = self._compute_servo_from_features(vision.features, vision.servo.tracking_source)
        snap = self.state.snapshot()
        snap.stats = get_system_stats("tmp")
        snap.serial_connected = self.serial.connected
        snap.tts_loaded = self.tts.loaded
        snap.camera_device_id = self.config.camera_device_id
        snap.camera_mode = f"{self.config.camera_width}x{self.config.camera_height}@{self.config.camera_fps}"
        snap.ollama_connected = True
        snap.playback_progress = self.audio.progress()
        return snap

    async def housekeeping_loop(self) -> None:
        while True:
            self.resources.cleanup_temp_audio()
            await asyncio.sleep(30)

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
            await asyncio.sleep(0.2)

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
        self.state.servo = self._compute_servo_from_features(features, vision.servo.tracking_source)
        if self.serial.connected and features.track_id is not None:
            self.serial.send_servo_command(
                self.state.servo.left_deg,
                self.state.servo.right_deg,
                mode="track",
                tracking_source=self.state.servo.tracking_source,
            )

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
            left_limits=(self.config.servo_left_min_deg, self.config.servo_left_max_deg),
            right_limits=(self.config.servo_right_min_deg, self.config.servo_right_max_deg),
        )
        servo.tracking_source = tracking_source
        return servo

    async def _maybe_generate_tracking_line(self) -> None:
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
        if self.audio.is_playing() or self.generation_lock.locked():
            return
        if time.monotonic() - self.last_idle_line_at < self.config.idle_sentence_interval_ms / 1000:
            return
        await self.generate_idle_line()

    async def generate_tracking_line(self) -> None:
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
        client = OllamaClient(self.config.ollama_base_url, self.config.ollama_timeout_sec)
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
        if self.config.tts_emotion_enabled:
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
        self.audio.set_output_device(self.config.audio_output_device)
        self.audio.play(output, volume=self.config.tts_output_volume)
        self.state.last_spoken_text = text

    async def _classify_tts_emotion(self, text: str) -> tuple[str, str]:
        client = OllamaClient(self.config.ollama_base_url, min(self.config.ollama_timeout_sec, 15))
        prompt = (
            "你是 TTS 情緒標記分類器。"
            "以下是 Fish Audio S1 支援的情緒標記。"
            f"你必須只從這個清單選一個最適合該句子的標記並直接輸出，不要解釋：{', '.join(TTS_EMOTION_TAGS)}。\n"
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
            fallback = self._fallback_tts_emotion(text)
            return fallback, fallback
        candidate = self._clean_tts_emotion(raw) or DEFAULT_TTS_EMOTION
        normalized = self._normalize_tts_emotion(candidate)
        if normalized is None:
            normalized = self._fallback_tts_emotion(text)
        return candidate, normalized

    def _clean_tts_emotion(self, raw: str) -> str | None:
        cleaned = raw.strip().strip("()[]{}<>\"'`.,，。!！？").lower().replace("_", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned or None

    def _normalize_tts_emotion(self, raw: str) -> str | None:
        cleaned = self._clean_tts_emotion(raw)
        return cleaned if cleaned in TTS_EMOTION_TAGS else None

    def _fallback_tts_emotion(self, text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("怒", "氣", "恨", "煩", "滾", "閉嘴", "討厭")):
            return "angry"
        if any(token in lowered for token in ("哭", "難過", "悲", "孤獨", "寂寞", "抱歉", "遺憾", "失望")):
            return "sad"
        if any(token in lowered for token in ("怕", "危險", "小心", "糟", "怎麼辦")):
            return "worried"
        if any(token in lowered for token in ("?", "？", "嗎", "呢", "為什麼", "怎樣")):
            return "curious"
        if any(token in lowered for token in ("!", "！", "太棒", "好耶", "快", "立刻")):
            return "excited"
        return DEFAULT_TTS_EMOTION

    def _apply_tts_emotion(self, text: str, emotion: str) -> str:
        return f"({emotion}) {text}"

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


async def build_apply_checks(payload: dict, config: RuntimeConfig) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    changed = set(payload.keys())

    if changed & {"camera_source", "camera_device_id", "camera_width", "camera_height", "camera_fps"}:
        if config.camera_source == "browser":
            checks.append(
                {
                    "component": "vision",
                    "status": "ok",
                    "message": f"Browser camera config staged: {config.camera_width}x{config.camera_height}@{config.camera_fps}. Applies on next uploaded frame.",
                }
            )
        else:
            checks.append(
                {
                    "component": "vision",
                    "status": "ok",
                    "message": f"Backend capture reconfigured to device {config.camera_device_id} at {config.camera_width}x{config.camera_height}@{config.camera_fps}.",
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

    if changed & {"yolo_model_path", "yolo_pose_model_path"}:
        checks.append({"component": "vision-model", "status": "ok", "message": "YOLO model paths updated and verified."})

    if changed & {"ollama_base_url", "ollama_model", "ollama_timeout_sec", "llm_use_person_crop"}:
        client = OllamaClient(config.ollama_base_url, min(config.ollama_timeout_sec, 15))
        try:
            models = await client.list_models()
            if config.ollama_model in models:
                mode = "person-crop image mode" if config.llm_use_person_crop else "prompt-only mode"
                checks.append({"component": "llm", "status": "ok", "message": f"Ollama reachable and model {config.ollama_model} is available. LLM is in {mode}."})
            else:
                checks.append({"component": "llm", "status": "error", "message": f"Ollama reachable but model {config.ollama_model} was not found."})
        except Exception as exc:
            checks.append({"component": "llm", "status": "error", "message": f"Ollama check failed: {exc}"})

    if changed & {"tts_model_path", "tts_emotion_enabled", "tts_clone_voice_enabled", "tts_ref_audio_path", "tts_ref_text_path", "tts_timeout_sec"}:
        errors: list[str] = []
        model_path = brain.tts.model_path if hasattr(brain.tts.model_path, "exists") else Path(brain.tts.model_path)
        ref_audio_path = brain.tts.ref_audio_path if hasattr(brain.tts.ref_audio_path, "exists") else Path(brain.tts.ref_audio_path)
        ref_text_path = brain.tts.ref_text_path if hasattr(brain.tts.ref_text_path, "exists") else Path(brain.tts.ref_text_path)
        if not model_path.exists():
            errors.append(f"missing model path: {model_path}")
        if config.tts_clone_voice_enabled:
            if not ref_audio_path.exists():
                errors.append(f"missing ref audio: {ref_audio_path}")
            if not ref_text_path.exists():
                errors.append(f"missing ref transcript: {ref_text_path}")
        if errors:
            checks.append({"component": "tts", "status": "error", "message": "; ".join(errors)})
        else:
            mode = "clone voice" if config.tts_clone_voice_enabled else "normal TTS"
            emotion_mode = "emotion on" if config.tts_emotion_enabled else "emotion off"
            checks.append({"component": "tts", "status": "ok", "message": f"TTS ready in {mode}, {emotion_mode}. Timeout set to {config.tts_timeout_sec * 1000} ms."})

    if changed & {"audio_output_device", "tts_output_volume"}:
        available_ids = {item["id"] for item in AudioPlayer.list_output_devices()}
        if config.audio_output_device == "default" or config.audio_output_device in available_ids:
            checks.append({"component": "audio", "status": "ok", "message": f"Audio output set to {config.audio_output_device}."})
        else:
            checks.append({"component": "audio", "status": "error", "message": f"Audio output device {config.audio_output_device} was not found."})

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
    await brain.start()
    try:
        yield
    finally:
        await brain.stop()


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
    brain.config = merged
    brain.serial = ESP32Link(brain.config.serial_port, brain.config.serial_baud_rate)
    brain.tts = FishCloneTTS(
        brain.config.tts_model_path,
        brain.config.tts_ref_audio_path,
        brain.config.tts_ref_text_path,
        clone_voice_enabled=brain.config.tts_clone_voice_enabled,
    )
    brain.vision.reconfigure(brain.config)
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
    client = OllamaClient(brain.config.ollama_base_url, brain.config.ollama_timeout_sec)
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

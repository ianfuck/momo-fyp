import asyncio
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.app as app_module
import backend.tts.qwen_clone as qwen_module
from backend.app import app, brain
from backend.tts.model_profiles import (
    FISH_AUDIO_S1_MINI_PROFILE,
    FISH_SPEECH_V1_5_PROFILE,
    QWEN3_TTS_0_6B_BASE_PROFILE,
    QWEN3_TTS_1_7B_BASE_PROFILE,
    resolve_tts_model_profile,
)
from backend.tts.qwen_clone import BenchmarkShutdownRequested, QwenCloneTTS
from backend.tts.qwen_runtime import QwenVoiceCloneTTS
from backend.tts.reference_selection import ReferencePair
from backend.tts.semantic_runtime import SemanticBenchmarkResult, SemanticRuntimePlan
from backend.types import AudienceFeatures, PipelineStage, ServoTelemetry
from backend.vision.runtime import VisionState


client = TestClient(app)


def test_status_endpoint_returns_pipeline_and_stats():
    response = client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert "pipeline" in payload
    assert "stats" in payload
    assert "serial_monitor" in payload
    assert "tts_emotion_raw" in payload
    assert "tts_emotion_applied" in payload
    assert "tts_emotion_used" in payload
    assert "tts_input_text" in payload
    assert "tts_reference_raw" in payload
    assert "tts_reference_pair" in payload
    assert "tts_reference_audio_path" in payload
    assert "tts_reference_text_path" in payload
    assert "tts_runtime" in payload
    assert "ollama_runtime" in payload
    assert "yolo_person_runtime" in payload
    assert "yolo_pose_runtime" in payload


def test_simulate_pipeline_returns_prompt_and_snapshot():
    async def fake_generate_tracking_line():
        brain.state.sentence_index = 3
        brain.state.set_pipeline_stage(PipelineStage.PLAYBACK)

    original = brain.generate_tracking_line
    brain.generate_tracking_line = fake_generate_tracking_line
    response = client.post("/api/control/simulate-pipeline", json={"sentence_index": 3, "event_summary": "蹲下"})
    brain.generate_tracking_line = original
    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshot"]["pipeline"]["stage"] == "PLAYBACK"


def test_browser_frame_upload_sends_servo_immediately():
    original_submit = brain.vision.submit_jpeg_frame
    original_serial = brain.serial
    original_config = brain.config.model_copy(deep=True)

    sent: list[tuple[float, float, str, str]] = []

    class FakeSerial:
        connected = False

        def send_servo_command(self, left_deg: float, right_deg: float, mode: str = "track", tracking_source: str = "eye_midpoint"):
            sent.append((left_deg, right_deg, mode, tracking_source))
            return "ok"

        def snapshot(self):
            from backend.types import SerialMonitorSnapshot

            return SerialMonitorSnapshot()

        def close(self):
            return None

    brain.config = original_config.model_copy(update={"camera_source": "browser"})
    brain.serial = FakeSerial()
    brain.vision.submit_jpeg_frame = lambda _: VisionState(
        features=AudienceFeatures(track_id=1, bbox_area_ratio=0.35, center_x_norm=0.5, eye_midpoint=[0.72, 0.5]),
        servo=ServoTelemetry(tracking_source="eye_midpoint"),
        frame_jpeg=None,
        frame_shape=(640, 480),
        target_seen_at=None,
    )

    try:
        response = client.post("/api/camera/frame", content=b"jpeg-bytes", headers={"Content-Type": "image/jpeg"})
        assert response.status_code == 200
        assert sent
        assert sent[0][2] == "track"
        assert sent[0][3] == "eye_midpoint"
    finally:
        brain.vision.submit_jpeg_frame = original_submit
        brain.serial = original_serial
        brain.config = original_config


def test_update_mode_attempts_send_even_when_serial_marked_disconnected():
    original_snapshot = brain.vision.get_snapshot
    original_serial = brain.serial

    sent: list[tuple[float, float]] = []

    class FakeSerial:
        connected = False

        def send_servo_command(self, left_deg: float, right_deg: float, mode: str = "track", tracking_source: str = "eye_midpoint"):
            sent.append((left_deg, right_deg))
            return "ok"

        def snapshot(self):
            from backend.types import SerialMonitorSnapshot

            return SerialMonitorSnapshot()

        def close(self):
            return None

    brain.serial = FakeSerial()
    brain.vision.get_snapshot = lambda: VisionState(
        features=AudienceFeatures(track_id=1, bbox_area_ratio=0.35, center_x_norm=0.5, eye_midpoint=[0.72, 0.5]),
        servo=ServoTelemetry(tracking_source="eye_midpoint"),
        frame_jpeg=None,
        frame_shape=(640, 480),
        target_seen_at=None,
    )

    try:
        brain._update_mode_from_vision()
        assert sent
    finally:
        brain.vision.get_snapshot = original_snapshot
        brain.serial = original_serial


def test_generate_tracking_line_uses_person_crop_mode_when_enabled():
    original_config = brain.config.model_copy(deep=True)
    original_prompt_builder = brain.prompts.build_tracking_prompt
    original_generate = brain._generate_with_ollama
    original_speak = brain._speak_line
    original_snapshot = brain.vision.get_snapshot
    original_sentence_index = brain.state.sentence_index
    original_active_sentence_index = brain.state.active_sentence_index
    original_audience = brain.state.audience.model_copy(deep=True)
    original_event_log = list(brain.state.event_log)

    captured: dict[str, object] = {}

    def fake_build_tracking_prompt(*args, **kwargs):
        captured["use_visual_audience"] = kwargs["use_visual_audience"]
        return {"system": "sys", "user": "user", "required_terms": []}

    async def fake_generate(system: str, prompt: str, limit: int, required_terms=None, images=None):
        captured["images"] = images
        return "測試台詞。"

    async def fake_speak(_: str):
        return None

    brain.config.llm_use_person_crop = True
    brain.state.sentence_index = 0
    brain.state.active_sentence_index = 0
    brain.state.audience = AudienceFeatures(top_color="灰色", distance_class="near")
    brain.prompts.build_tracking_prompt = fake_build_tracking_prompt
    brain._generate_with_ollama = fake_generate
    brain._speak_line = fake_speak
    brain.vision.get_snapshot = lambda: VisionState(
        features=AudienceFeatures(track_id=1, person_bbox=[2, 3, 10, 12]),
        servo=ServoTelemetry(tracking_source="person"),
        person_crop_jpeg=b"fake-crop",
    )

    try:
        asyncio.run(brain.generate_tracking_line())
        assert captured["use_visual_audience"] is True
        assert captured["images"] == [b"fake-crop"]
    finally:
        brain.config = original_config
        brain.prompts.build_tracking_prompt = original_prompt_builder
        brain._generate_with_ollama = original_generate
        brain._speak_line = original_speak
        brain.vision.get_snapshot = original_snapshot
        brain.state.sentence_index = original_sentence_index
        brain.state.active_sentence_index = original_active_sentence_index
        brain.state.audience = original_audience
        brain.state.event_log = original_event_log


def test_generate_tracking_line_falls_back_when_person_crop_missing():
    original_config = brain.config.model_copy(deep=True)
    original_prompt_builder = brain.prompts.build_tracking_prompt
    original_generate = brain._generate_with_ollama
    original_speak = brain._speak_line
    original_snapshot = brain.vision.get_snapshot
    original_sentence_index = brain.state.sentence_index
    original_active_sentence_index = brain.state.active_sentence_index
    original_audience = brain.state.audience.model_copy(deep=True)
    original_event_log = list(brain.state.event_log)

    captured: dict[str, object] = {}

    def fake_build_tracking_prompt(*args, **kwargs):
        captured["use_visual_audience"] = kwargs["use_visual_audience"]
        return {"system": "sys", "user": "user", "required_terms": []}

    async def fake_generate(system: str, prompt: str, limit: int, required_terms=None, images=None):
        captured["images"] = images
        return "測試台詞。"

    async def fake_speak(_: str):
        return None

    brain.config.llm_use_person_crop = True
    brain.state.sentence_index = 0
    brain.state.active_sentence_index = 0
    brain.state.audience = AudienceFeatures(top_color="灰色", distance_class="near")
    brain.prompts.build_tracking_prompt = fake_build_tracking_prompt
    brain._generate_with_ollama = fake_generate
    brain._speak_line = fake_speak
    brain.vision.get_snapshot = lambda: VisionState(
        features=AudienceFeatures(track_id=1, person_bbox=[2, 3, 10, 12]),
        servo=ServoTelemetry(tracking_source="person"),
        person_crop_jpeg=None,
    )

    try:
        asyncio.run(brain.generate_tracking_line())
        assert captured["use_visual_audience"] is False
        assert captured["images"] is None
        assert any("falling back to prompt-only mode" in item for item in brain.state.event_log)
    finally:
        brain.config = original_config
        brain.prompts.build_tracking_prompt = original_prompt_builder
        brain._generate_with_ollama = original_generate
        brain._speak_line = original_speak
        brain.vision.get_snapshot = original_snapshot
        brain.state.sentence_index = original_sentence_index
        brain.state.active_sentence_index = original_active_sentence_index
        brain.state.audience = original_audience
        brain.state.event_log = original_event_log


def test_update_config_returns_apply_checks():
    response = client.post("/api/config", json={"camera_width": 640})
    assert response.status_code == 200
    payload = response.json()
    assert "apply_checks" in payload
    assert isinstance(payload["apply_checks"], list)


def test_update_config_validation_failure_returns_feedback():
    response = client.post("/api/config", json={"camera_width": 100})
    assert response.status_code == 200
    payload = response.json()
    assert payload["validation_errors"]
    assert payload["apply_checks"][0]["status"] == "error"


def test_update_config_can_reapply_same_payload():
    first = client.post("/api/config", json={"camera_width": 640})
    second = client.post("/api/config", json={"camera_width": 640})
    assert first.status_code == 200
    assert second.status_code == 200
    payload = second.json()
    assert payload["validation_errors"] == []
    assert payload["apply_checks"]


def test_update_config_accepts_servo_eye_spacing():
    original_config = brain.config.model_copy(deep=True)
    original_serial = brain.serial
    try:
        response = client.post("/api/config", json={"servo_eye_spacing_cm": 12})
        assert response.status_code == 200
        payload = response.json()
        assert payload["validation_errors"] == []
        assert payload["applied_config"]["servo_eye_spacing_cm"] == 12
    finally:
        brain.config = original_config
        brain.serial.close()
        brain.serial = original_serial


def test_update_config_accepts_servo_output_inverted():
    original_config = brain.config.model_copy(deep=True)
    original_serial = brain.serial
    try:
        response = client.post("/api/config", json={"servo_output_inverted": True})
        assert response.status_code == 200
        payload = response.json()
        assert payload["validation_errors"] == []
        assert payload["applied_config"]["servo_output_inverted"] is True
    finally:
        brain.config = original_config
        brain.serial.close()
        brain.serial = original_serial


def test_update_config_accepts_servo_trim_and_gain():
    original_config = brain.config.model_copy(deep=True)
    original_serial = brain.serial
    try:
        response = client.post(
            "/api/config",
            json={
                "servo_left_trim_deg": 2.5,
                "servo_right_trim_deg": -1.5,
                "servo_left_gain": 1.2,
                "servo_right_gain": 0.8,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["validation_errors"] == []
        assert payload["applied_config"]["servo_left_trim_deg"] == 2.5
        assert payload["applied_config"]["servo_right_trim_deg"] == -1.5
        assert payload["applied_config"]["servo_left_gain"] == 1.2
        assert payload["applied_config"]["servo_right_gain"] == 0.8
    finally:
        brain.config = original_config
        brain.serial.close()
        brain.serial = original_serial


def test_update_config_tts_path_no_server_error():
    response = client.post(
        "/api/config",
        json={"tts_model_path": "model/huggingface/hf_snapshots/fishaudio__s1-mini"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["validation_errors"] == []
    assert any(item["component"] == "tts" for item in payload["apply_checks"])


def test_update_config_unloads_previous_tts_runtime(monkeypatch):
    original_tts = brain.tts
    original_config = brain.config.model_copy(deep=True)
    original_select = brain._select_tts_runtime
    original_reconfigure = brain.vision.reconfigure

    unloaded: list[str] = []

    class FakeTTS:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path
            self.ref_audio_path = "resource/voice/ref-voice3.wav"
            self.ref_text_path = "resource/voice/transcript3.txt"
            self.loaded = False
            self.device = "cpu"
            self.device_backend = "cpu"
            self.semantic_dispatch_mode = "single"
            self.model_profile = resolve_tts_model_profile(model_path)

        def unload(self) -> None:
            unloaded.append(self.model_path)

    old_tts = FakeTTS("model/huggingface/hf_snapshots/Qwen__Qwen3-TTS-12Hz-0.6B-Base")
    new_tts = FakeTTS("model/huggingface/hf_snapshots/Qwen__Qwen3-TTS-12Hz-1.7B-Base")
    brain.tts = old_tts
    brain.config = original_config.model_copy(update={"tts_model_path": old_tts.model_path})

    monkeypatch.setattr(app_module, "should_prepare_models", lambda: False)
    monkeypatch.setattr(brain, "_select_tts_runtime", lambda selection_source: new_tts)
    monkeypatch.setattr(brain.vision, "reconfigure", lambda *_: None)

    try:
        response = client.post("/api/config", json={"tts_model_path": new_tts.model_path})
        assert response.status_code == 200
        assert unloaded == [old_tts.model_path]
        assert brain.tts is new_tts
    finally:
        brain.tts = original_tts
        brain.config = original_config
        brain._select_tts_runtime = original_select
        brain.vision.reconfigure = original_reconfigure


def test_update_config_can_disable_clone_voice():
    original = brain.config.tts_clone_voice_enabled
    try:
        response = client.post("/api/config", json={"tts_clone_voice_enabled": False})
        assert response.status_code == 200
        payload = response.json()
        assert payload["validation_errors"] == []
        assert payload["applied_config"]["tts_clone_voice_enabled"] is False
        assert any(
            item["component"] == "tts" and "normal TTS" in item["message"]
            for item in payload["apply_checks"]
        )
    finally:
        client.post("/api/config", json={"tts_clone_voice_enabled": original})


def test_update_config_can_disable_tts_emotion():
    original = brain.config.tts_emotion_enabled
    try:
        response = client.post("/api/config", json={"tts_emotion_enabled": False})
        assert response.status_code == 200
        payload = response.json()
        assert payload["validation_errors"] == []
        assert payload["applied_config"]["tts_emotion_enabled"] is False
        assert any(
            item["component"] == "tts" and "emotion off" in item["message"]
            for item in payload["apply_checks"]
        )
    finally:
        client.post("/api/config", json={"tts_emotion_enabled": original})


def test_update_config_tts_timeout_validation_failure_returns_feedback():
    response = client.post("/api/config", json={"tts_timeout_sec": 0})
    assert response.status_code == 200
    payload = response.json()
    assert "tts_timeout_sec must be >= 1" in payload["validation_errors"]


def test_speak_line_reports_tts_timeout_ms():
    original_tts = brain.tts
    original_timeout = brain.config.tts_timeout_sec

    class SlowTTS:
        loaded = False

        def synthesize(self, text: str, output_path: str) -> str:
            time.sleep(0.05)
            return output_path

    brain.tts = SlowTTS()
    brain.config.tts_timeout_sec = 0.01

    try:
        try:
            asyncio.run(brain._speak_line("測試"))
            assert False, "expected timeout"
        except RuntimeError as exc:
            assert "TTS timeout after" in str(exc)
            assert "limit 10.0 ms" in str(exc)
    finally:
        brain.tts = original_tts
        brain.config.tts_timeout_sec = original_timeout


def test_speak_line_records_emotion_selected_by_ollama_and_applied_to_fish():
    original_tts = brain.tts
    original_classify = brain._classify_tts_emotion
    original_set_output = brain.audio.set_output_device
    original_play = brain.audio.play
    original_stage = brain.state.pipeline
    original_tts_latency = brain.state.tts_latency_ms
    original_raw = brain.state.tts_emotion_raw
    original_applied = brain.state.tts_emotion_applied
    original_used = brain.state.tts_emotion_used
    original_input_text = brain.state.tts_input_text
    original_last_spoken = brain.state.last_spoken_text
    captured: dict[str, object] = {}

    class FakeTTS:
        loaded = True

        def synthesize(self, text: str, output_path: str) -> str:
            captured["tts_text"] = text
            return output_path

    async def fake_classify(_: str) -> tuple[str, str]:
        return "excited", "excited"

    brain.tts = FakeTTS()
    brain._classify_tts_emotion = fake_classify
    brain.audio.set_output_device = lambda *_: None
    brain.audio.play = lambda wav_path, volume=1.0: wav_path

    try:
        asyncio.run(brain._speak_line("測試台詞。"))
        assert captured["tts_text"] == "(excited)測試台詞。"
        assert brain.state.tts_emotion_raw == "excited"
        assert brain.state.tts_emotion_applied == "excited"
        assert brain.state.tts_emotion_used is True
        assert brain.state.tts_input_text == "(excited)測試台詞。"
        assert brain.state.last_spoken_text == "測試台詞。"
        assert brain.state.pipeline.stage == PipelineStage.PLAYBACK
    finally:
        brain.tts = original_tts
        brain._classify_tts_emotion = original_classify
        brain.audio.set_output_device = original_set_output
        brain.audio.play = original_play
        brain.state.pipeline = original_stage
        brain.state.tts_latency_ms = original_tts_latency
        brain.state.tts_emotion_raw = original_raw
        brain.state.tts_emotion_applied = original_applied
        brain.state.tts_emotion_used = original_used
        brain.state.tts_input_text = original_input_text
        brain.state.last_spoken_text = original_last_spoken


def test_speak_line_skips_emotion_when_tts_emotion_disabled():
    original_tts = brain.tts
    original_classify = brain._classify_tts_emotion
    original_set_output = brain.audio.set_output_device
    original_play = brain.audio.play
    original_enabled = brain.config.tts_emotion_enabled
    original_raw = brain.state.tts_emotion_raw
    original_applied = brain.state.tts_emotion_applied
    original_used = brain.state.tts_emotion_used
    original_input_text = brain.state.tts_input_text
    captured: dict[str, object] = {}

    class FakeTTS:
        loaded = True

        def synthesize(self, text: str, output_path: str) -> str:
            captured["tts_text"] = text
            return output_path

    async def fail_classify(_: str) -> tuple[str, str]:
        raise AssertionError("emotion classifier should not be called")

    brain.tts = FakeTTS()
    brain._classify_tts_emotion = fail_classify
    brain.audio.set_output_device = lambda *_: None
    brain.audio.play = lambda wav_path, volume=1.0: wav_path
    brain.config.tts_emotion_enabled = False

    try:
        asyncio.run(brain._speak_line("測試台詞。"))
        assert captured["tts_text"] == "測試台詞。"
        assert brain.state.tts_emotion_raw is None
        assert brain.state.tts_emotion_applied is None
        assert brain.state.tts_emotion_used is False
        assert brain.state.tts_input_text == "測試台詞。"
    finally:
        brain.tts = original_tts
        brain._classify_tts_emotion = original_classify
        brain.audio.set_output_device = original_set_output
        brain.audio.play = original_play
        brain.config.tts_emotion_enabled = original_enabled
        brain.state.tts_emotion_raw = original_raw
        brain.state.tts_emotion_applied = original_applied
        brain.state.tts_emotion_used = original_used
        brain.state.tts_input_text = original_input_text


def test_prepare_runtime_models_does_not_crash_on_tts_preload_failure(monkeypatch):
    original_vision = brain.vision
    original_tts = brain.tts
    original_event_log = list(brain.state.event_log)

    class FailingTTS:
        loaded = False

        def __init__(self, *args, **kwargs):
            pass

        def preload(self) -> None:
            raise RuntimeError("decode failed")

    class DummyVision:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr("backend.app.ensure_runtime_models", lambda config: [])
    monkeypatch.setattr("backend.app.FishCloneTTS", FailingTTS)
    monkeypatch.setattr("backend.app.VisionRuntime", DummyVision)

    try:
        brain._prepare_runtime_models()
        assert brain.vision.config is brain.config
        assert brain.tts.loaded is False
        assert any("TTS preload failed: decode failed" in item for item in brain.state.event_log)
    finally:
        brain.vision = original_vision
        brain.tts = original_tts
        brain.state.event_log = original_event_log


def test_prepare_runtime_models_recovers_from_tts_oom_via_benchmark(monkeypatch):
    original_vision = brain.vision
    original_tts = brain.tts
    original_config = brain.config.model_copy(deep=True)
    original_event_log = list(brain.state.event_log)

    class OomTTS:
        loaded = False
        device = "cuda:0"
        device_backend = "gpu"
        semantic_dispatch_mode = "single"

        def __init__(self, *args, **kwargs):
            self.device_mode = kwargs.get("device_mode", "gpu")

        def preload(self) -> None:
            raise RuntimeError("CUDA out of memory")

    class RecoveredTTS:
        loaded = False
        device = "cpu"
        device_backend = "cpu"
        semantic_dispatch_mode = "single"

        def preload(self) -> None:
            self.loaded = True

    class DummyVision:
        def __init__(self, config):
            self.config = config

    class FakeSelection:
        def __init__(self):
            self.tts = RecoveredTTS()
            self.result = type(
                "Result",
                (),
                {
                    "name": "cpu",
                    "elapsed_ms": 123,
                    "preload_ms": 80,
                    "synth_ms": 43,
                    "device_mode": "cpu",
                    "precision_mode": "float32",
                    "peak_vram_mb": 0.0,
                    "ram_mb": 10.0,
                    "vram_mb": 0.0,
                },
            )()
            self.results = [
                type(
                    "Result",
                    (),
                    {
                        "name": "cpu-float32",
                        "ok": True,
                        "elapsed_ms": 123,
                        "preload_ms": 80,
                        "synth_ms": 43,
                        "semantic_dispatch_mode": "single",
                        "precision_mode": "float32",
                        "peak_vram_mb": 0.0,
                        "detail": "",
                    },
                )()
            ]

    brain.config.tts_device_mode = "gpu"
    monkeypatch.setattr("backend.app.ensure_runtime_models", lambda config: [])
    monkeypatch.setattr("backend.app.should_skip_tts_benchmark", lambda: False)
    monkeypatch.setattr("backend.app.FishCloneTTS", OomTTS)
    monkeypatch.setattr("backend.app.VisionRuntime", DummyVision)
    monkeypatch.setattr(
        OomTTS,
        "benchmark_auto_profiles",
        classmethod(lambda cls, *args, **kwargs: FakeSelection()),
        raising=False,
    )

    try:
        brain._prepare_runtime_models()
        assert brain.tts.loaded is True
        assert brain.tts_runtime.selection_source == "benchmark"
        assert brain.tts_runtime.requested_mode == "gpu"
        assert any("recovered from OOM via benchmark fallback" in item for item in brain.state.event_log)
    finally:
        brain.vision = original_vision
        brain.tts = original_tts
        brain.config = original_config
        brain.state.event_log = original_event_log


def test_select_tts_runtime_uses_benchmark_when_auto(monkeypatch):
    original_tts = brain.tts
    original_config = brain.config.model_copy(deep=True)
    original_event_log = list(brain.state.event_log)

    class FakeTTS:
        loaded = True
        device_backend = "gpu"
        device = "cuda:0"
        semantic_dispatch_mode = "auto"
        precision_mode = "float32"

        def __init__(self, *args, **kwargs):
            self.device_mode = kwargs.get("device_mode", "auto")

        @classmethod
        def benchmark_auto_profiles(cls, *args, **kwargs):
            return FakeSelection()

    class FakeSelection:
        def __init__(self):
            self.tts = FakeTTS()
            self.result = type(
                "Result",
                (),
                {
                    "name": "semantic-auto-gpu-float32",
                    "elapsed_ms": 321,
                    "preload_ms": 221,
                    "synth_ms": 100,
                    "device_mode": "gpu",
                    "precision_mode": "float32",
                    "peak_vram_mb": 1024.0,
                },
            )()
            self.results = [
                type(
                    "Result",
                    (),
                    {
                        "name": "semantic-auto-gpu-float32",
                        "ok": True,
                        "elapsed_ms": 321,
                        "preload_ms": 221,
                        "synth_ms": 100,
                        "semantic_dispatch_mode": "auto",
                        "precision_mode": "float32",
                        "peak_vram_mb": 1024.0,
                        "detail": "",
                    },
                )()
            ]

    brain.config.tts_device_mode = "auto"
    monkeypatch.setattr("backend.app.should_skip_tts_benchmark", lambda: False)
    monkeypatch.setattr("backend.app.FishCloneTTS", FakeTTS)

    try:
        selected = brain._select_tts_runtime(selection_source="default")
        assert selected.semantic_dispatch_mode == "auto"
        assert selected.precision_mode == "float32"
        assert brain.tts_benchmark_selected == "semantic-auto-gpu-float32"
        assert brain.config.tts_device_mode == "gpu"
        assert brain.tts_runtime.selection_source == "benchmark"
        assert brain.tts_runtime.precision_mode == "float32"
        assert any("TTS benchmark selected semantic-auto-gpu-float32 with synth 100 ms" in item for item in brain.state.event_log)
    finally:
        brain.tts = original_tts
        brain.config = original_config
        brain.state.event_log = original_event_log
        brain.tts_benchmark_selected = None
        brain.tts_benchmark_results = []


def test_select_tts_runtime_skips_benchmark_when_requested(monkeypatch):
    original_tts = brain.tts
    original_config = brain.config.model_copy(deep=True)

    class FakeTTS:
        loaded = False
        device_backend = "cpu"
        device = "cpu"
        semantic_dispatch_mode = "single"
        precision_mode = "float32"

        def __init__(self, *args, **kwargs):
            self.device_mode = kwargs.get("device_mode", "auto")

    brain.config.tts_device_mode = "auto"
    monkeypatch.setattr("backend.app.should_skip_tts_benchmark", lambda: True)
    monkeypatch.setattr("backend.app.FishCloneTTS", FakeTTS)

    try:
        selected = brain._select_tts_runtime(selection_source="user")
        assert selected.semantic_dispatch_mode == "single"
        assert brain.tts_benchmark_selected is None
        assert brain.tts_runtime.selection_source == "user"
    finally:
        brain.tts = original_tts
        brain.config = original_config


def test_benchmark_auto_profiles_selects_best_isolated_candidate(monkeypatch):
    class FakeTTS(QwenCloneTTS):
        def __init__(self, *args, device_mode="auto", semantic_dispatch_mode="single", precision_mode=None, **kwargs):
            self.model_path = "model"
            self.ref_audio_path = "ref.wav"
            self.ref_text_path = "ref.txt"
            self.clone_voice_enabled = kwargs.get("clone_voice_enabled", True)
            self.semantic_dispatch_mode = semantic_dispatch_mode
            self.precision_mode = precision_mode or ("float16" if device_mode != "cpu" else "float32")
            self.loaded = False
            self.device = "cpu" if device_mode == "cpu" else "cuda:0"
            self.device_backend = "cpu" if device_mode == "cpu" else "gpu"

    monkeypatch.setattr(
        qwen_module,
        "benchmark_plans_for_current_host",
        lambda: [
            SemanticRuntimePlan(name="gpu", device_mode="gpu", semantic_dispatch_mode="single"),
            SemanticRuntimePlan(name="semantic-auto-gpu", device_mode="gpu", semantic_dispatch_mode="auto"),
            SemanticRuntimePlan(name="cpu", device_mode="cpu", semantic_dispatch_mode="single"),
        ],
    )

    def fake_runner(**kwargs):
        plan = kwargs["plan"]
        if plan.name == "gpu-float16":
            return SemanticBenchmarkResult(
                name="gpu-float16",
                device_mode="gpu",
                semantic_dispatch_mode="single",
                elapsed_ms=50,
                ok=False,
                preload_ms=None,
                synth_ms=None,
                precision_mode="float16",
                peak_vram_mb=None,
                detail="oom",
            )
        if plan.name == "gpu-float32":
            return SemanticBenchmarkResult(
                name="gpu-float32",
                device_mode="gpu",
                semantic_dispatch_mode="single",
                elapsed_ms=190,
                ok=True,
                preload_ms=120,
                synth_ms=70,
                precision_mode="float32",
                peak_vram_mb=900.0,
            )
        if plan.name == "semantic-auto-gpu-float16":
            return SemanticBenchmarkResult(
                name="semantic-auto-gpu-float16",
                device_mode="gpu",
                semantic_dispatch_mode="auto",
                elapsed_ms=120,
                ok=True,
                preload_ms=20,
                synth_ms=100,
                precision_mode="float16",
                peak_vram_mb=850.0,
            )
        if plan.name == "semantic-auto-gpu-float32":
            return SemanticBenchmarkResult(
                name="semantic-auto-gpu-float32",
                device_mode="gpu",
                semantic_dispatch_mode="auto",
                elapsed_ms=150,
                ok=True,
                preload_ms=30,
                synth_ms=120,
                precision_mode="float32",
                peak_vram_mb=930.0,
            )
        if plan.name == "cpu-bfloat16":
            return SemanticBenchmarkResult(
                name="cpu-bfloat16",
                device_mode="cpu",
                semantic_dispatch_mode="single",
                elapsed_ms=1000,
                ok=False,
                preload_ms=None,
                synth_ms=None,
                precision_mode="bfloat16",
                peak_vram_mb=None,
                detail="unsupported",
            )
        return SemanticBenchmarkResult(
            name="cpu-float32",
            device_mode="cpu",
            semantic_dispatch_mode="single",
            elapsed_ms=90,
            ok=True,
            preload_ms=10,
            synth_ms=80,
            precision_mode="float32",
            peak_vram_mb=0.0,
        )

    monkeypatch.setattr(qwen_module, "_run_benchmark_candidate_subprocess", fake_runner)

    selection = FakeTTS.benchmark_auto_profiles("model", "ref.wav", "ref.txt", clone_voice_enabled=False)

    assert selection is not None
    assert selection.result.name == "gpu-float32"
    assert selection.result.precision_mode == "float32"
    assert selection.result.synth_ms == 70
    assert selection.tts.device_backend == "gpu"
    assert selection.tts.semantic_dispatch_mode == "single"
    assert selection.tts.precision_mode == "float32"


def test_benchmark_candidate_subprocess_terminates_when_shutdown_requested(monkeypatch):
    plan = qwen_module.TTSBenchmarkCandidate(
        name="gpu-float16",
        device_mode="gpu",
        semantic_dispatch_mode="single",
        precision_mode="float16",
    )
    fake_process = type("FakeProcess", (), {"pid": 4321, "returncode": None})()

    def fake_poll():
        return fake_process.returncode

    fake_process.poll = fake_poll

    terminated = {"called": False}

    def fake_terminate(process):
        terminated["called"] = True
        process.returncode = -15

    monkeypatch.setattr(qwen_module.subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr(qwen_module, "shutdown_requested", lambda: True)
    monkeypatch.setattr(qwen_module, "_terminate_benchmark_process", fake_terminate)

    with pytest.raises(BenchmarkShutdownRequested, match="shutdown requested"):
        qwen_module._run_benchmark_candidate_subprocess(
            plan=plan,
            model_path="model",
            ref_audio_path="ref.wav",
            ref_text_path="ref.txt",
            clone_voice_enabled=False,
            sample_text="測試。",
        )

    assert terminated["called"] is True


def test_main_skip_tts_benchmark_sets_env_and_runs_uvicorn(monkeypatch):
    original = os.environ.get("MOMO_SKIP_TTS_BENCHMARK")
    called: dict[str, object] = {}

    def fake_run(app_target: str, **kwargs):
        called["app_target"] = app_target
        called["kwargs"] = kwargs

    monkeypatch.setattr(app_module.uvicorn, "run", fake_run)
    try:
        app_module.main(["--skip-tts-benchmark", "--host", "0.0.0.0", "--port", "9000"])
        assert os.environ["MOMO_SKIP_TTS_BENCHMARK"] == "1"
        assert called["app_target"] == "backend.app:app"
        assert called["kwargs"]["host"] == "0.0.0.0"
        assert called["kwargs"]["port"] == 9000
    finally:
        if original is None:
            os.environ.pop("MOMO_SKIP_TTS_BENCHMARK", None)
        else:
            os.environ["MOMO_SKIP_TTS_BENCHMARK"] = original


def test_get_config_reflects_latest_applied_value():
    client.post("/api/config", json={"camera_fps": 15})
    response = client.get("/api/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["camera_fps"] == 15


def test_status_snapshot_uses_latest_vision_servo_angles():
    original_get_snapshot = brain.vision.get_snapshot
    original_config = brain.config.model_copy(deep=True)
    brain.state.servo = ServoTelemetry(left_deg=83.5, right_deg=97.25, tracking_source="eye_midpoint")

    class FakeVisionState:
        features = AudienceFeatures(top_color="灰色", bbox_area_ratio=0.35, center_x_norm=0.5, eye_midpoint=[0.72, 0.5])
        servo = ServoTelemetry(left_deg=90, right_deg=90, tracking_source="eye_midpoint")
        frame_jpeg = None
        frame_shape = None
        target_seen_at = None

    brain.config = original_config.model_copy(
        update={
            "servo_left_zero_deg": 90.0,
            "servo_right_zero_deg": 90.0,
            "servo_output_inverted": False,
            "servo_left_trim_deg": 0.0,
            "servo_right_trim_deg": 0.0,
            "servo_left_gain": 1.0,
            "servo_right_gain": 1.0,
        }
    )
    brain.vision.get_snapshot = lambda: FakeVisionState()
    response = client.get("/api/status")
    brain.vision.get_snapshot = original_get_snapshot
    brain.config = original_config
    assert response.status_code == 200
    payload = response.json()
    assert payload["servo"]["left_deg"] > 90
    assert payload["servo"]["right_deg"] > 90
    assert payload["servo"]["tracking_source"] == "eye_midpoint"


def test_collect_startup_diagnostics_reports_expected_backends(monkeypatch):
    class DummyDetector:
        def warmup(self) -> str:
            return "gpu"

    class DummyPose:
        def warmup(self) -> str:
            return "gpu"

    class DummyVision:
        detector = DummyDetector()
        pose = DummyPose()

    class DummyTTS:
        device_backend = "gpu"
        loaded = True

    class DummyOllama:
        def __init__(self, *args, **kwargs):
            pass

        async def list_models(self) -> list[str]:
            return [brain.config.ollama_model]

        async def warmup_model(self, model: str) -> dict:
            return {"model": model}

        async def running_models(self) -> list[dict]:
            return [{"name": brain.config.ollama_model, "size_vram": 123}]

    original_vision = brain.vision
    original_tts = brain.tts
    brain.vision = DummyVision()
    brain.tts = DummyTTS()
    monkeypatch.setattr("backend.app.expected_accelerator_label", lambda: "gpu")
    monkeypatch.setattr("backend.app.OllamaClient", DummyOllama)

    try:
        lines = asyncio.run(brain._collect_startup_diagnostics())
        assert any("yolo person=gpu pose=gpu" in line for line in lines)
        assert any("tts backend=gpu" in line for line in lines)
        assert any("ollama backend=gpu" in line for line in lines)
    finally:
        brain.vision = original_vision
        brain.tts = original_tts


def test_reference_audio_loads_wav_without_ffmpeg(monkeypatch):
    tts = QwenCloneTTS("model", "voice.wav", "transcript.txt")
    called = {"ffmpeg": False, "loaded": None}

    def fake_load(path: str):
        called["loaded"] = path
        return [0.0], 24000

    monkeypatch.setattr(tts, "_load_audio_with_librosa", fake_load)
    monkeypatch.setattr(
        tts,
        "_convert_reference_audio_with_ffmpeg",
        lambda source: called.__setitem__("ffmpeg", True),
    )

    wav, sr = tts._load_reference_audio()
    assert wav == [0.0]
    assert sr == 24000
    assert called["loaded"] == "voice.wav"
    assert called["ffmpeg"] is False


def test_snapshot_recomputes_servo_from_latest_vision(monkeypatch):
    from backend.types import AudienceFeatures, ServoTelemetry

    original_snapshot = brain.vision.get_snapshot
    original_state_servo = brain.state.servo
    original_config = brain.config.model_copy(deep=True)

    brain.state.servo = ServoTelemetry(left_deg=10.0, right_deg=20.0, tracking_source="stale")

    def fake_snapshot():
        return VisionState(
            features=AudienceFeatures(
                track_id=1,
                bbox_area_ratio=0.35,
                center_x_norm=0.5,
                eye_midpoint=[0.72, 0.5],
            ),
            servo=ServoTelemetry(tracking_source="eye_midpoint"),
            frame_jpeg=None,
            frame_shape=(640, 480),
            target_seen_at=None,
        )

    brain.config = original_config.model_copy(
        update={
            "servo_left_zero_deg": 90.0,
            "servo_right_zero_deg": 90.0,
            "servo_output_inverted": False,
            "servo_left_trim_deg": 0.0,
            "servo_right_trim_deg": 0.0,
            "servo_left_gain": 1.0,
            "servo_right_gain": 1.0,
        }
    )
    brain.vision.get_snapshot = fake_snapshot
    try:
        snap = brain.snapshot()
        assert snap.servo.left_deg > 90
        assert snap.servo.right_deg > 90
        assert snap.servo.tracking_source == "eye_midpoint"
    finally:
        brain.vision.get_snapshot = original_snapshot
        brain.state.servo = original_state_servo
        brain.config = original_config


def test_compute_servo_can_invert_output():
    original_config = brain.config.model_copy(deep=True)
    try:
        features = AudienceFeatures(
            track_id=1,
            bbox_area_ratio=0.35,
            center_x_norm=0.5,
            eye_midpoint=[0.72, 0.5],
        )
        brain.config = original_config.model_copy(
            update={
                "servo_left_zero_deg": 90.0,
                "servo_right_zero_deg": 90.0,
                "servo_output_inverted": False,
                "servo_left_min_deg": 45.0,
                "servo_left_max_deg": 135.0,
                "servo_right_min_deg": 45.0,
                "servo_right_max_deg": 135.0,
            }
        )
        normal = brain._compute_servo_from_features(features, "eye_midpoint")
        brain.config = brain.config.model_copy(update={"servo_output_inverted": True})
        inverted = brain._compute_servo_from_features(features, "eye_midpoint")

        assert inverted.left_deg == round((2 * 90.0) - normal.left_deg, 2)
        assert inverted.right_deg == round((2 * 90.0) - normal.right_deg, 2)
    finally:
        brain.config = original_config


def test_compute_servo_trim_and_gain_affect_output():
    original_config = brain.config.model_copy(deep=True)
    try:
        features = AudienceFeatures(
            track_id=1,
            bbox_area_ratio=0.35,
            center_x_norm=0.5,
            eye_midpoint=[0.72, 0.5],
        )
        brain.config = original_config.model_copy(
            update={
                "servo_left_zero_deg": 90.0,
                "servo_right_zero_deg": 90.0,
                "servo_output_inverted": False,
                "servo_left_trim_deg": 0.0,
                "servo_right_trim_deg": 0.0,
                "servo_left_gain": 1.0,
                "servo_right_gain": 1.0,
                "servo_left_min_deg": 45.0,
                "servo_left_max_deg": 135.0,
                "servo_right_min_deg": 45.0,
                "servo_right_max_deg": 135.0,
            }
        )
        baseline = brain._compute_servo_from_features(features, "eye_midpoint")
        brain.config = brain.config.model_copy(
            update={
                "servo_left_trim_deg": 2.5,
                "servo_right_trim_deg": -1.5,
                "servo_left_gain": 1.5,
                "servo_right_gain": 0.5,
            }
        )
        adjusted = brain._compute_servo_from_features(features, "eye_midpoint")

        expected_left = round(90.0 + ((baseline.left_deg - 90.0) * 1.5) + 2.5, 2)
        expected_right = round(90.0 + ((baseline.right_deg - 90.0) * 0.5) - 1.5, 2)

        assert adjusted.left_deg == expected_left
        assert adjusted.right_deg == expected_right
        assert adjusted.left_deg != baseline.left_deg
        assert adjusted.right_deg != baseline.right_deg
    finally:
        brain.config = original_config


def test_snapshot_includes_serial_monitor(monkeypatch):
    from backend.types import SerialMonitorEntry, SerialMonitorSnapshot

    original_snapshot = brain.vision.get_snapshot
    original_serial = brain.serial

    class FakeSerial:
        connected = True

        def snapshot(self):
            return SerialMonitorSnapshot(
                port="/dev/cu.usbmodem-test",
                baud_rate=115200,
                last_tx="{\"type\":\"servo\"}",
                last_tx_at="2026-04-07T10:00:00Z",
                last_rx="{\"type\":\"ack\"}",
                last_rx_at="2026-04-07T10:00:01Z",
                entries=[
                    SerialMonitorEntry(ts="2026-04-07T10:00:01Z", direction="rx", message="{\"type\":\"ack\"}"),
                    SerialMonitorEntry(ts="2026-04-07T10:00:00Z", direction="tx", message="{\"type\":\"servo\"}"),
                ],
            )

        def close(self):
            return None

    brain.vision.get_snapshot = lambda: VisionState(
        features=AudienceFeatures(),
        servo=ServoTelemetry(tracking_source="none"),
        frame_jpeg=None,
        frame_shape=(640, 480),
        target_seen_at=None,
    )
    brain.serial = FakeSerial()

    try:
        snap = brain.snapshot()
        assert snap.serial_connected is True
        assert snap.serial_monitor.port == "/dev/cu.usbmodem-test"
        assert snap.serial_monitor.last_tx == "{\"type\":\"servo\"}"
        assert snap.serial_monitor.entries[0].direction == "rx"
    finally:
        brain.vision.get_snapshot = original_snapshot
        brain.serial = original_serial


def test_reference_audio_prefers_ffmpeg_for_mp3(monkeypatch):
    tts = QwenCloneTTS("model", "voice.mp3", "transcript.txt")
    loaded_paths: list[str] = []

    def fake_load(path: str):
        loaded_paths.append(path)
        return [1.0], 24000

    monkeypatch.setattr(tts, "_load_audio_with_librosa", fake_load)
    monkeypatch.setattr(
        tts,
        "_convert_reference_audio_with_ffmpeg",
        lambda source: Path("tmp/ref_voice_source.wav"),
    )

    wav, sr = tts._load_reference_audio()
    assert wav == [1.0]
    assert sr == 24000
    assert loaded_paths == ["tmp/ref_voice_source.wav"]


def test_reference_cache_path_changes_with_source_file(tmp_path):
    source_a = tmp_path / "voice-a.wav"
    source_b = tmp_path / "voice-b.wav"
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")

    tts_a = QwenCloneTTS("model", str(source_a), "transcript.txt")
    tts_b = QwenCloneTTS("model", str(source_b), "transcript.txt")

    assert tts_a._reference_cache_path(source_a) != tts_b._reference_cache_path(source_b)


def test_load_reference_text_reads_transcript(tmp_path):
    transcript = tmp_path / "ref.txt"
    transcript.write_text("新的參考逐字稿", encoding="utf-8")
    tts = QwenCloneTTS("model", "voice.wav", str(transcript))

    assert tts._load_reference_text() == "新的參考逐字稿"


def test_load_reference_text_skips_long_ascii_transcript(tmp_path):
    transcript = tmp_path / "ref.txt"
    transcript.write_text(
        (
            "great embodiment of Chaos hear me for ages Untold I studied your ways devoting my existence to you "
            "through every season and every silent night until the end of time itself"
        ),
        encoding="utf-8",
    )
    tts = QwenCloneTTS("model", "voice.wav", str(transcript))

    assert tts._load_reference_text() is not None


def test_load_reference_text_keeps_short_ascii_transcript(tmp_path):
    transcript = tmp_path / "ref.txt"
    transcript.write_text(
        "great embodiment of Chaos hear me, I studied your ways and devoted my existence to you",
        encoding="utf-8",
    )
    tts = QwenCloneTTS("model", "voice.wav", str(transcript))

    assert tts._load_reference_text() is not None


def test_qwen_voice_clone_tts_set_reference_paths_clears_cached_prompt():
    tts = QwenVoiceCloneTTS("model", "voice.wav", "transcript.txt")
    tts._clone_prompt = object()

    tts.set_reference_paths("next.wav", "next.txt")

    assert tts.ref_audio_path == "next.wav"
    assert tts.ref_text_path == "next.txt"
    assert tts._clone_prompt is None


def test_qwen_voice_clone_tts_retries_windows_cuda_numeric_failure_with_stable_gpu(monkeypatch, tmp_path):
    ref_audio = tmp_path / "voice.wav"
    ref_text = tmp_path / "transcript.txt"
    ref_audio.write_bytes(b"wav")
    ref_text.write_text("測試參考", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tts = QwenVoiceCloneTTS(str(model_dir), str(ref_audio), str(ref_text), device_mode="cpu")
    tts.device = "cuda:0"
    tts.device_backend = "gpu"

    class UnstableModel:
        def generate_voice_clone(self, **kwargs):
            raise RuntimeError("probability tensor contains either `inf`, `nan` or element < 0")

    class StableModel:
        def generate_voice_clone(self, **kwargs):
            return [np.zeros(8, dtype=np.float32)], 24000

    state = {"stable_model_requested": False}

    def fake_ensure_model():
        if tts._prefer_stable_cuda_profile:
            state["stable_model_requested"] = True
            tts._model = StableModel()
        else:
            tts._model = UnstableModel()
        tts._clone_prompt = object()
        tts.loaded = True

    writes: list[tuple[str, int]] = []

    monkeypatch.setattr("backend.tts.qwen_runtime.platform.system", lambda: "Windows")
    monkeypatch.setattr(tts, "_ensure_model", fake_ensure_model)
    monkeypatch.setattr("backend.tts.qwen_runtime.sf.write", lambda path, audio, sr: writes.append((str(path), sr)))

    output = tts.synthesize("測試。", str(tmp_path / "out.wav"))

    assert output.endswith("out.wav")
    assert state["stable_model_requested"] is True
    assert writes == [(str(tmp_path / "out.wav"), 24000)]


def test_fish_clone_tts_set_reference_paths_updates_runtime_values():
    tts = QwenCloneTTS(
        "model/huggingface/hf_snapshots/fishaudio__s1-mini",
        "voice.wav",
        "transcript.txt",
        clone_voice_enabled=False,
    )

    tts.set_reference_paths("next.wav", "next.txt")

    assert tts.ref_audio_path == "next.wav"
    assert tts.ref_text_path == "next.txt"


def test_apply_tts_emotion_wraps_text_with_supported_tag():
    tagged = brain._apply_tts_emotion("你好啊。", "excited")
    assert tagged == "(excited)你好啊。"


def test_build_request_skips_references_when_clone_voice_disabled():
    tts = QwenCloneTTS("model", "missing.wav", "missing.txt", clone_voice_enabled=False)

    request = tts._build_request("測試台詞。")

    assert request.text == "測試台詞。"
    assert request.references == []


def test_clean_tts_emotion_normalizes_raw_label():
    assert brain._clean_tts_emotion(" (soft_tone)。 ") == "soft tone"
    assert brain._clean_tts_emotion("   ") is None


def test_normalize_tts_emotion_accepts_supported_tag_only():
    assert brain._normalize_tts_emotion(" (curious) ") == "curious"
    assert brain._normalize_tts_emotion("neutral") is None


def test_fallback_tts_emotion_returns_supported_emotion():
    assert brain._fallback_tts_emotion("你為什麼還不來？") == "curious"
    assert brain._fallback_tts_emotion("我真的很孤獨。") == "sad"


def test_select_tts_reference_pair_fixed_uses_configured_paths():
    original_config = brain.config.model_copy(deep=True)

    brain.config.tts_clone_voice_enabled = True
    brain.config.tts_reference_mode = "fixed"
    brain.config.tts_ref_audio_path = "resource/voice/ref-voice3.wav"
    brain.config.tts_ref_text_path = "resource/voice/transcript3.txt"

    try:
        pair = asyncio.run(brain._select_tts_reference_pair("測試台詞。"))
        assert pair.audio_path == "resource/voice/ref-voice3.wav"
        assert pair.text_path == "resource/voice/transcript3.txt"
    finally:
        brain.config = original_config


def test_select_tts_reference_pair_random_uses_random_library_choice(monkeypatch):
    original_config = brain.config.model_copy(deep=True)
    expected = ReferencePair("冷靜且專業", "audio.mp3", "text.txt")

    brain.config.tts_clone_voice_enabled = True
    brain.config.tts_reference_mode = "random"
    monkeypatch.setattr("backend.app.choose_random_emotional_reference_pair", lambda *_: expected)

    try:
        pair = asyncio.run(brain._select_tts_reference_pair("測試台詞。"))
        assert pair == expected
    finally:
        brain.config = original_config


def test_classify_tts_reference_pair_uses_ollama_choice():
    original_config = brain.config.model_copy(deep=True)
    original_stream = brain._stream_text

    async def fake_stream(*args, **kwargs):
        return "懷疑嚴厲"

    brain.config.tts_clone_voice_enabled = True
    brain.config.tts_reference_mode = "ollama_emotion"
    brain._stream_text = fake_stream

    try:
        raw, pair = asyncio.run(brain._classify_tts_reference_pair("你最好老實說。"))
        assert raw == "懷疑嚴厲"
        assert pair.key == "懷疑且嚴厲"
        assert pair.audio_path.endswith(".MP3")
        assert pair.text_path.endswith(".txt")
    finally:
        brain.config = original_config
        brain._stream_text = original_stream


def test_select_tts_reference_pair_avoids_consecutive_duplicate_in_ollama_mode():
    original_config = brain.config.model_copy(deep=True)
    original_stream = brain._stream_text
    original_last_pair = brain._last_tts_reference_pair_key

    responses = iter(["悲慟自責絕望", "冷靜專業"])

    async def fake_stream(*args, **kwargs):
        return next(responses)

    brain.config.tts_clone_voice_enabled = True
    brain.config.tts_reference_mode = "ollama_emotion"
    brain._last_tts_reference_pair_key = "極度的悲慟和自責與絕望"
    brain._stream_text = fake_stream

    try:
        pair = asyncio.run(brain._select_tts_reference_pair("這句其實很平靜。"))
        assert pair.key == "冷靜且專業"
        assert brain.state.tts_reference_raw == "冷靜專業"
    finally:
        brain.config = original_config
        brain._stream_text = original_stream
        brain._last_tts_reference_pair_key = original_last_pair


def test_speak_line_applies_selected_reference_pair_before_synthesis():
    original_tts = brain.tts
    original_config = brain.config.model_copy(deep=True)
    original_select_reference = brain._select_tts_reference_pair
    original_set_output = brain.audio.set_output_device
    original_play = brain.audio.play

    captured: dict[str, object] = {}

    class FakeTTS:
        loaded = True
        model_profile = QWEN3_TTS_0_6B_BASE_PROFILE

        def set_reference_paths(self, audio_path: str, text_path: str) -> None:
            captured["ref_audio_path"] = audio_path
            captured["ref_text_path"] = text_path

        def synthesize(self, text: str, output_path: str) -> str:
            captured["tts_text"] = text
            return output_path

    async def fake_select_reference(_: str) -> ReferencePair:
        return ReferencePair("冷靜且專業", "resource/voice/emotional-ref/冷靜且專業.MP3", "resource/voice/emotional-ref/冷靜且專業.txt")

    brain.tts = FakeTTS()
    brain.config.tts_clone_voice_enabled = True
    brain.config.tts_reference_mode = "ollama_emotion"
    brain.config.tts_emotion_enabled = False
    brain._select_tts_reference_pair = fake_select_reference
    brain.audio.set_output_device = lambda *_: None
    brain.audio.play = lambda wav_path, volume=1.0: wav_path

    try:
        asyncio.run(brain._speak_line("測試台詞。"))
        assert captured["ref_audio_path"] == "resource/voice/emotional-ref/冷靜且專業.MP3"
        assert captured["ref_text_path"] == "resource/voice/emotional-ref/冷靜且專業.txt"
        assert captured["tts_text"] == "測試台詞。"
        assert brain.state.tts_reference_pair == "冷靜且專業"
        assert brain.state.tts_reference_audio_path == "resource/voice/emotional-ref/冷靜且專業.MP3"
        assert brain.state.tts_reference_text_path == "resource/voice/emotional-ref/冷靜且專業.txt"
    finally:
        brain.tts = original_tts
        brain.config = original_config
        brain._select_tts_reference_pair = original_select_reference
        brain.audio.set_output_device = original_set_output
        brain.audio.play = original_play


def test_v1_5_profile_uses_basic_emotion_set_only():
    assert brain._normalize_tts_emotion("disappointed", profile=FISH_SPEECH_V1_5_PROFILE) is None
    assert brain._normalize_tts_emotion("sad", profile=FISH_SPEECH_V1_5_PROFILE) == "sad"
    assert brain._fallback_tts_emotion("怎麼辦，我有點怕。", profile=FISH_SPEECH_V1_5_PROFILE) == "worried"


def test_resolve_tts_model_profile_by_model_path():
    assert resolve_tts_model_profile("model/huggingface/hf_snapshots/fishaudio__fish-speech-1.5") == FISH_SPEECH_V1_5_PROFILE
    assert resolve_tts_model_profile("model/huggingface/hf_snapshots/fishaudio__s1-mini") == FISH_AUDIO_S1_MINI_PROFILE
    assert resolve_tts_model_profile("model/huggingface/hf_snapshots/Qwen__Qwen3-TTS-12Hz-0.6B-Base") == QWEN3_TTS_0_6B_BASE_PROFILE
    assert resolve_tts_model_profile("model/huggingface/hf_snapshots/Qwen__Qwen3-TTS-12Hz-1.7B-Base") == QWEN3_TTS_1_7B_BASE_PROFILE


def test_speak_line_skips_structured_emotion_for_qwen_profile():
    original_tts = brain.tts
    original_set_output = brain.audio.set_output_device
    original_play = brain.audio.play
    original_raw = brain.state.tts_emotion_raw
    original_applied = brain.state.tts_emotion_applied
    original_used = brain.state.tts_emotion_used
    original_input_text = brain.state.tts_input_text

    captured: dict[str, object] = {}

    class FakeQwenTTS:
        loaded = True
        model_profile = QWEN3_TTS_0_6B_BASE_PROFILE

        def synthesize(self, text: str, output_path: str) -> str:
            captured["tts_text"] = text
            return output_path

    brain.tts = FakeQwenTTS()
    brain.audio.set_output_device = lambda *_: None
    brain.audio.play = lambda wav_path, volume=1.0: wav_path

    try:
        asyncio.run(brain._speak_line("測試台詞。"))
        assert captured["tts_text"] == "測試台詞。"
        assert brain.state.tts_emotion_raw is None
        assert brain.state.tts_emotion_applied is None
        assert brain.state.tts_emotion_used is False
        assert brain.state.tts_input_text == "測試台詞。"
    finally:
        brain.tts = original_tts
        brain.audio.set_output_device = original_set_output
        brain.audio.play = original_play
        brain.state.tts_emotion_raw = original_raw
        brain.state.tts_emotion_applied = original_applied
        brain.state.tts_emotion_used = original_used
        brain.state.tts_input_text = original_input_text


def test_polish_waveform_applies_fades_and_recenters() -> None:
    tts = QwenCloneTTS("model", "voice.wav", "transcript.txt")
    wav = np.ones(2400, dtype=np.float32) * 0.5

    polished = tts._polish_waveform(wav, 24000)

    assert abs(float(np.mean(polished))) < 0.05
    assert abs(float(polished[0])) < 1e-4
    assert abs(float(polished[-1])) < 1e-4


def test_suppress_transient_spikes_interpolates_impulses() -> None:
    tts = QwenCloneTTS("model", "voice.wav", "transcript.txt")
    wav = np.array([0.0, 0.02, 0.01, 0.72, 0.0, -0.01, 0.0], dtype=np.float32)

    repaired = tts._suppress_transient_spikes(wav)

    assert abs(float(repaired[3])) < 0.02


def test_suppress_transient_spikes_repairs_short_two_sample_burst() -> None:
    tts = QwenCloneTTS("model", "voice.wav", "transcript.txt")
    wav = np.array([0.0, 0.01, 0.0, 0.6, -0.55, 0.02, 0.01, 0.0], dtype=np.float32)

    repaired = tts._suppress_transient_spikes(wav)

    assert abs(float(repaired[3])) < 0.1
    assert abs(float(repaired[4])) < 0.1


def test_suppress_transient_spikes_repairs_short_three_sample_burst() -> None:
    tts = QwenCloneTTS("model", "voice.wav", "transcript.txt")
    wav = np.array([0.0, 0.01, 0.0, 0.55, -0.5, 0.48, 0.01, 0.0, 0.0], dtype=np.float32)

    repaired = tts._suppress_transient_spikes(wav)

    assert np.max(np.abs(repaired[3:6])) < 0.12


def test_select_best_waveform_prefers_lower_click_score(monkeypatch) -> None:
    tts = QwenCloneTTS("model", "voice.wav", "transcript.txt")
    direct = np.array([0.0, 0.01, 0.02], dtype=np.float32)
    repaired = np.array([0.0, 0.3, -0.3], dtype=np.float32)

    def fake_finalize(wav, sr, repair_spikes):
        return repaired if repair_spikes else direct

    monkeypatch.setattr(tts, "_finalize_waveform", fake_finalize)

    selected = tts._select_best_waveform(np.array([0.0], dtype=np.float32), 24000)

    assert np.array_equal(selected, direct)

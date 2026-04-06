import asyncio
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app import app, brain
from backend.tts.qwen_clone import QwenCloneTTS
from backend.types import AudienceFeatures, PipelineStage, ServoTelemetry
from backend.vision.runtime import VisionState


client = TestClient(app)


def test_status_endpoint_returns_pipeline_and_stats():
    response = client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert "pipeline" in payload
    assert "stats" in payload
    assert "tts_emotion_raw" in payload
    assert "tts_emotion_applied" in payload
    assert "tts_emotion_used" in payload
    assert "tts_input_text" in payload


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


def test_update_config_tts_path_no_server_error():
    response = client.post(
        "/api/config",
        json={"tts_model_path": "model/huggingface/hf_snapshots/fishaudio__s1-mini"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["validation_errors"] == []
    assert any(item["component"] == "tts" for item in payload["apply_checks"])


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
        assert captured["tts_text"] == "(excited) 測試台詞。"
        assert brain.state.tts_emotion_raw == "excited"
        assert brain.state.tts_emotion_applied == "excited"
        assert brain.state.tts_emotion_used is True
        assert brain.state.tts_input_text == "(excited) 測試台詞。"
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


def test_get_config_reflects_latest_applied_value():
    client.post("/api/config", json={"camera_fps": 15})
    response = client.get("/api/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["camera_fps"] == 15


def test_status_snapshot_uses_latest_vision_servo_angles():
    original_get_snapshot = brain.vision.get_snapshot
    brain.state.servo = ServoTelemetry(left_deg=83.5, right_deg=97.25, tracking_source="eye_midpoint")

    class FakeVisionState:
        features = AudienceFeatures(top_color="灰色", bbox_area_ratio=0.35, center_x_norm=0.5, eye_midpoint=[0.72, 0.5])
        servo = ServoTelemetry(left_deg=90, right_deg=90, tracking_source="eye_midpoint")
        frame_jpeg = None
        frame_shape = None
        target_seen_at = None

    brain.vision.get_snapshot = lambda: FakeVisionState()
    response = client.get("/api/status")
    brain.vision.get_snapshot = original_get_snapshot
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

    brain.vision.get_snapshot = fake_snapshot
    try:
        snap = brain.snapshot()
        assert snap.servo.left_deg > 90
        assert snap.servo.right_deg > 90
        assert snap.servo.tracking_source == "eye_midpoint"
    finally:
        brain.vision.get_snapshot = original_snapshot
        brain.state.servo = original_state_servo


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


def test_apply_tts_emotion_wraps_text_with_supported_tag():
    tagged = brain._apply_tts_emotion("你好啊。", "excited")
    assert tagged == "(excited) 你好啊。"


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

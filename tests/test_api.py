import asyncio
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app, brain
from backend.tts.qwen_clone import QwenCloneTTS
from backend.types import AudienceFeatures, PipelineStage, ServoTelemetry


client = TestClient(app)


def test_status_endpoint_returns_pipeline_and_stats():
    response = client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert "pipeline" in payload
    assert "stats" in payload


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
        json={"tts_model_path": "model/huggingface/hf_snapshots/Qwen__Qwen3-TTS-12Hz-1.7B-Base"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["validation_errors"] == []
    assert any(item["component"] == "tts" for item in payload["apply_checks"])


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

    monkeypatch.setattr("backend.app.QwenCloneTTS", FailingTTS)
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
    from backend.vision.runtime import VisionState

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

import platform

from backend.config import build_field_catalog, validate_runtime_config
from backend.types import RuntimeConfig
from backend.llm.ollama_client import OllamaClient
from backend.tts import semantic_runtime
from backend.tts.model_profiles import KOKORO_CHINESE_VOICES, supported_tts_model_paths


def test_default_ollama_timeout_is_600():
    config = RuntimeConfig()
    assert config.ollama_timeout_sec == 600
    assert config.ollama_model == "qwen3.5:2b"
    assert config.llm_liberation_mode is False
    assert config.tts_model_path == "model/huggingface/hf_snapshots/hexgrad__Kokoro-82M-v1.1-zh"
    assert config.tts_kokoro_voice == "zf_002"
    assert config.tts_reference_mode == "ollama_emotion"
    assert config.tts_route_via_virtual_device is False


def test_invalid_config_detected():
    config = RuntimeConfig(camera_width=100, history_max_sentences=9)
    errors = validate_runtime_config(config)
    assert any("camera_width" in item for item in errors)
    assert any("history_max_sentences" in item for item in errors)


def test_device_mode_fields_expose_os_specific_enum():
    config = RuntimeConfig()
    fields = {field.key: field for field in build_field_catalog(config)}
    accelerator = "mps" if platform.system() == "Darwin" else "gpu"

    assert fields["camera_flip_vertical"].type == "boolean"
    assert fields["yolo_device_mode"].enum == ["auto", "cpu", accelerator]
    assert fields["tts_device_mode"].enum == ["auto", "cpu", accelerator]
    assert fields["ollama_device_mode"].enum == ["auto", "cpu", accelerator]
    assert fields["llm_liberation_mode"].type == "boolean"
    assert fields["tts_reference_mode"].enum == ["fixed", "ollama_emotion", "random"]
    assert fields["tts_route_via_virtual_device"].type == "boolean"
    assert fields["led_min_brightness_pct"].type == "float"
    assert fields["led_max_brightness_pct"].type == "float"
    assert fields["led_signal_loss_fade_out_ms"].type == "int"
    assert fields["led_brightness_output_inverted"].type == "boolean"
    assert fields["led_left_right_inverted"].type == "boolean"


def test_tts_model_field_exposes_supported_model_options():
    config = RuntimeConfig()
    fields = {field.key: field for field in build_field_catalog(config)}

    assert fields["tts_model_path"].enum == supported_tts_model_paths()
    assert "model/huggingface/hf_snapshots/hexgrad__Kokoro-82M-v1.1-zh" in fields["tts_model_path"].enum
    assert "model/huggingface/hf_snapshots/myshell-ai__MeloTTS-Chinese" in fields["tts_model_path"].enum
    assert fields["tts_kokoro_voice"].enum == list(KOKORO_CHINESE_VOICES)


def test_kokoro_voice_field_is_hidden_for_non_kokoro_model():
    config = RuntimeConfig(tts_model_path="model/huggingface/hf_snapshots/Qwen__Qwen3-TTS-12Hz-0.6B-Base")
    fields = {field.key: field for field in build_field_catalog(config)}

    assert "tts_kokoro_voice" not in fields


def test_invalid_tts_reference_mode_detected():
    config = RuntimeConfig(tts_reference_mode="bad-mode")

    errors = validate_runtime_config(config)

    assert "tts_reference_mode must be one of ['fixed', 'ollama_emotion', 'random']" in errors


def test_invalid_kokoro_voice_detected():
    config = RuntimeConfig(tts_kokoro_voice="bad_voice")

    errors = validate_runtime_config(config)

    assert "tts_kokoro_voice must be one of the supported Kokoro Chinese voices" in errors


def test_invalid_led_brightness_config_detected():
    config = RuntimeConfig(led_min_brightness_pct=90, led_max_brightness_pct=10)

    errors = validate_runtime_config(config)

    assert "led_min_brightness_pct must be <= led_max_brightness_pct" in errors


def test_invalid_led_signal_loss_fade_out_config_detected():
    config = RuntimeConfig(led_signal_loss_fade_out_ms=-1)

    errors = validate_runtime_config(config)

    assert "led_signal_loss_fade_out_ms must be >= 0" in errors


def test_ollama_client_cpu_mode_sets_num_gpu_zero():
    client = OllamaClient("http://127.0.0.1:11434", 30, "cpu")
    assert client._ollama_options({})["num_gpu"] == 0


def test_ollama_client_auto_mode_leaves_num_gpu_unset():
    client = OllamaClient("http://127.0.0.1:11434", 30, "auto")
    assert "num_gpu" not in client._ollama_options({})


def test_benchmark_plans_prioritize_accelerator_before_cpu(monkeypatch):
    monkeypatch.setattr(semantic_runtime, "resolve_accelerator_mode", lambda: "gpu")
    monkeypatch.setattr(semantic_runtime, "accelerate_available", lambda: True)

    plans = semantic_runtime.benchmark_plans_for_current_host()

    assert [plan.name for plan in plans] == ["gpu", "semantic-auto-gpu", "cpu"]

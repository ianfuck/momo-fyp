import platform

from backend.config import build_field_catalog, validate_runtime_config
from backend.types import RuntimeConfig
from backend.llm.ollama_client import OllamaClient
from backend.tts import semantic_runtime


def test_default_ollama_timeout_is_600():
    config = RuntimeConfig()
    assert config.ollama_timeout_sec == 600
    assert config.ollama_model == "qwen3.5:2b"
    assert config.tts_model_path == "model/huggingface/hf_snapshots/fishaudio__fish-speech-1.5"


def test_invalid_config_detected():
    config = RuntimeConfig(camera_width=100, history_max_sentences=9)
    errors = validate_runtime_config(config)
    assert any("camera_width" in item for item in errors)
    assert any("history_max_sentences" in item for item in errors)


def test_device_mode_fields_expose_os_specific_enum():
    config = RuntimeConfig()
    fields = {field.key: field for field in build_field_catalog(config)}
    accelerator = "mps" if platform.system() == "Darwin" else "gpu"

    assert fields["yolo_device_mode"].enum == ["auto", "cpu", accelerator]
    assert fields["tts_device_mode"].enum == ["auto", "cpu", accelerator]
    assert fields["ollama_device_mode"].enum == ["auto", "cpu", accelerator]


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

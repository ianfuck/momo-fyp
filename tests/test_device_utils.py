import sys
from types import SimpleNamespace

from backend import device_utils


def test_get_tts_device_uses_cuda_on_windows(monkeypatch):
    calls: list[tuple[float, int]] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            set_per_process_memory_fraction=lambda fraction, index: calls.append((fraction, index)),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(device_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(device_utils, "_WINDOWS_CUDA_MEMORY_CONFIGURED", False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert device_utils.get_tts_device() == "cuda:0"
    assert calls == [(0.72, 0)]


def test_get_vision_device_uses_cpu_on_macos(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr(device_utils.platform, "system", lambda: "Darwin")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert device_utils.get_vision_device() == "cpu"

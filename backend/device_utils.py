from __future__ import annotations

import os
import platform

_WINDOWS_CUDA_MEMORY_CONFIGURED = False


def _configure_windows_cuda_memory_fraction(torch) -> None:
    global _WINDOWS_CUDA_MEMORY_CONFIGURED
    if _WINDOWS_CUDA_MEMORY_CONFIGURED:
        return
    if platform.system() != "Windows":
        return
    if not torch.cuda.is_available():
        return
    try:
        fraction = float(os.getenv("MOMO_CUDA_MEMORY_FRACTION", "0.72"))
        torch.cuda.set_per_process_memory_fraction(min(max(fraction, 0.1), 0.98), 0)
    except Exception:
        return
    _WINDOWS_CUDA_MEMORY_CONFIGURED = True


def _cuda_available(torch) -> bool:
    return bool(getattr(torch.cuda, "is_available", lambda: False)())


def _mps_available(torch) -> bool:
    mps = getattr(getattr(torch.backends, "mps", None), "is_available", None)
    return bool(callable(mps) and mps())


def _windows_low_vram_tts_auto(torch) -> bool:
    if platform.system() != "Windows" or not _cuda_available(torch):
        return False
    try:
        total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        threshold_gb = float(os.getenv("MOMO_TTS_AUTO_MIN_VRAM_GB", "6.0"))
        return total_bytes < threshold_gb * (1024 ** 3)
    except Exception:
        return False


def get_torch_device(mode: str = "auto", *, component: str = "general") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    normalized = (mode or "auto").strip().lower()
    if platform.system() == "Darwin":
        if normalized == "cpu":
            return "cpu"
        if normalized in {"mps", "auto"} and _mps_available(torch):
            return "mps"
        return "cpu"

    if normalized == "cpu":
        return "cpu"
    if normalized == "auto" and component == "tts" and _windows_low_vram_tts_auto(torch):
        return "cpu"
    if normalized in {"gpu", "auto"} and _cuda_available(torch):
        _configure_windows_cuda_memory_fraction(torch)
        return "cuda:0"
    return "cpu"


def get_vision_device(mode: str = "auto") -> str:
    override = os.getenv("MOMO_VISION_DEVICE")
    if override:
        return override
    if (mode or "auto").strip().lower() == "auto" and platform.system() == "Darwin":
        return "cpu"
    return get_torch_device(mode, component="vision")


def expected_accelerator_label() -> str:
    return "mps" if platform.system() == "Darwin" else "gpu"


def get_tts_device(mode: str = "auto") -> str:
    return get_torch_device(mode, component="tts")


def expected_tts_backend_label(mode: str = "auto") -> str:
    return backend_label_for_device(get_tts_device(mode))


def expected_vision_backend_label(mode: str = "auto") -> str:
    return backend_label_for_device(get_vision_device(mode))


def backend_label_for_device(device: str) -> str:
    if device.startswith("cuda"):
        return "gpu"
    if device == "mps":
        return "mps"
    return "cpu"

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


def get_torch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if platform.system() == "Darwin":
        mps = getattr(getattr(torch.backends, "mps", None), "is_available", None)
        if callable(mps) and mps():
            return "mps"
        return "cpu"

    if torch.cuda.is_available():
        _configure_windows_cuda_memory_fraction(torch)
        return "cuda:0"
    return "cpu"


def get_vision_device() -> str:
    override = os.getenv("MOMO_VISION_DEVICE")
    if override:
        return override
    if platform.system() == "Darwin":
        return "cpu"
    return get_torch_device()


def expected_accelerator_label() -> str:
    return "mps" if platform.system() == "Darwin" else "gpu"


def get_tts_device() -> str:
    return get_torch_device()


def expected_tts_backend_label() -> str:
    return backend_label_for_device(get_tts_device())


def expected_vision_backend_label() -> str:
    return backend_label_for_device(get_vision_device())


def backend_label_for_device(device: str) -> str:
    if device.startswith("cuda"):
        return "gpu"
    if device == "mps":
        return "mps"
    return "cpu"

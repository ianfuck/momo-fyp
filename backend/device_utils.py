from __future__ import annotations

import platform


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
        return "cuda:0"
    return "cpu"


def expected_accelerator_label() -> str:
    return "mps" if platform.system() == "Darwin" else "gpu"


def backend_label_for_device(device: str) -> str:
    if device.startswith("cuda"):
        return "gpu"
    if device == "mps":
        return "mps"
    return "cpu"

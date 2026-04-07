from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psutil

from backend.types import SystemStats


@dataclass(frozen=True)
class ProcessFootprint:
    ram_mb: float
    vram_mb: float | None


def capture_process_footprint(device: str | None = None) -> ProcessFootprint:
    process = psutil.Process()
    info = process.memory_info()
    return ProcessFootprint(
        ram_mb=round(info.rss / (1024 * 1024), 2),
        vram_mb=_device_memory_mb(device),
    )


def diff_process_footprint(before: ProcessFootprint, after: ProcessFootprint) -> tuple[float | None, float | None]:
    ram_delta = max(0.0, round(after.ram_mb - before.ram_mb, 2))
    if before.vram_mb is None or after.vram_mb is None:
        vram_delta = None
    else:
        vram_delta = max(0.0, round(after.vram_mb - before.vram_mb, 2))
    return ram_delta, vram_delta


def get_system_stats(tmp_dir: str = "tmp") -> SystemStats:
    process = psutil.Process()
    info = process.memory_info()
    temp_files = list(Path(tmp_dir).glob("*"))
    temp_size = sum(item.stat().st_size for item in temp_files if item.is_file())
    footprint = capture_process_footprint()
    return SystemStats(
        memory_rss_mb=round(info.rss / (1024 * 1024), 2),
        memory_vms_mb=round(info.vms / (1024 * 1024), 2),
        gpu_memory_mb=footprint.vram_mb,
        temp_file_count=len(temp_files),
        temp_file_size_mb=round(temp_size / (1024 * 1024), 2),
    )


def _device_memory_mb(device: str | None = None) -> float | None:
    try:
        import torch
    except ImportError:
        return None

    resolved = device or _preferred_runtime_device(torch)
    if resolved is None:
        return None
    if resolved.startswith("cuda") and torch.cuda.is_available():
        index = int(resolved.split(":", 1)[1]) if ":" in resolved else 0
        try:
            return round(float(torch.cuda.memory_reserved(index)) / (1024 * 1024), 2)
        except Exception:
            return None
    if resolved == "mps":
        mps = getattr(torch, "mps", None)
        if mps is None:
            return None
        for name in ("driver_allocated_memory", "current_allocated_memory"):
            fn = getattr(mps, name, None)
            if callable(fn):
                try:
                    return round(float(fn()) / (1024 * 1024), 2)
                except Exception:
                    continue
    return None


def _preferred_runtime_device(torch) -> str | None:
    if getattr(torch.cuda, "is_available", lambda: False)():
        return "cuda:0"
    mps = getattr(getattr(torch.backends, "mps", None), "is_available", None)
    if callable(mps) and mps():
        return "mps"
    return None

"""Best-effort GPU / accelerator snapshot for the dashboard (no extra deps)."""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any


def _parse_float_token(s: str) -> float | None:
    t = s.strip()
    if not t or t in ("N/A", "[N/A]", "[Not Supported]", "Unknown Error"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _nvidia_smi_snapshot() -> dict[str, Any] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=0.9,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0 or not (r.stdout or "").strip():
        return None
    line = (r.stdout or "").strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return None
    util = _parse_float_token(parts[0])
    mem_used = _parse_float_token(parts[1])
    mem_total = _parse_float_token(parts[2])
    pwr = _parse_float_token(parts[3])
    pwr_lim = _parse_float_token(parts[4])
    temp = _parse_float_token(parts[5]) if len(parts) > 5 else None
    return {
        "utilization_gpu_pct": util,
        "memory_used_mb": mem_used,
        "memory_total_mb": mem_total,
        "power_draw_w": pwr,
        "power_limit_w": pwr_lim,
        "temperature_c": temp,
    }


def _torch_cuda_snapshot() -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        d = torch.cuda.current_device()
        name = torch.cuda.get_device_name(d)
        alloc = torch.cuda.memory_allocated(d)
        reserved = torch.cuda.memory_reserved(d)
        return {
            "device_index": d,
            "device_name": name,
            "allocated_mb": round(alloc / 1024**2, 1),
            "reserved_mb": round(reserved / 1024**2, 1),
        }
    except Exception:
        return None


def _torch_mps_snapshot() -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        return None
    try:
        cur = torch.mps.current_allocated_memory()
        out: dict[str, Any] = {
            "current_allocated_mb": round(cur / 1024**2, 1),
        }
        drv_fn = getattr(torch.mps, "driver_allocated_memory", None)
        if callable(drv_fn):
            try:
                drv = drv_fn()
                out["driver_allocated_mb"] = round(drv / 1024**2, 1)
            except Exception:
                pass
        return out
    except Exception:
        return None


def collect_gpu_metrics() -> dict[str, Any]:
    """
    Merge system GPU (nvidia-smi) with in-process torch CUDA/MPS memory.
    Apple Silicon has no standard user-space GPU % without extra tools; MPS bytes are a useful proxy.
    """
    out: dict[str, Any] = {"ts": time.time(), "sources": []}
    nv = _nvidia_smi_snapshot()
    if nv:
        out["nvidia_smi"] = nv
        out["sources"].append("nvidia-smi")
    cu = _torch_cuda_snapshot()
    if cu:
        out["torch_cuda"] = cu
        out["sources"].append("torch.cuda")
    mp = _torch_mps_snapshot()
    if mp:
        out["torch_mps"] = mp
        out["sources"].append("torch.mps")
    if not out["sources"]:
        out["note"] = "未偵測到 nvidia-smi 或可用的 torch CUDA/MPS；若用 Apple GPU，可從「活動監視」看 GPU 歷程。"
    return out

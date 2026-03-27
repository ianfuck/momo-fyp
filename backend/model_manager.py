from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from backend.types import RuntimeConfig


ULTRALYTICS_ASSET_BASE = "https://github.com/ultralytics/assets/releases/latest/download"
DEFAULT_QWEN_TTS_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def ensure_runtime_models(config: RuntimeConfig) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    checks.append(_ensure_yolo_asset(config.yolo_model_path))
    checks.append(_ensure_yolo_asset(config.yolo_pose_model_path))
    checks.append(_ensure_tts_model(config.tts_model_path))
    return checks


def _ensure_yolo_asset(target_path: str) -> dict[str, str]:
    target = Path(target_path)
    if target.exists():
        return {
            "component": "vision-model",
            "status": "ok",
            "message": f"YOLO model ready at {target}.",
        }

    if target.suffix != ".pt":
        raise ValueError(f"YOLO model path must point to a .pt file: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    download_url = f"{ULTRALYTICS_ASSET_BASE}/{target.name}"
    _download_file(download_url, target)
    return {
        "component": "vision-model",
        "status": "ok",
        "message": f"Downloaded YOLO model to {target}.",
    }


def _ensure_tts_model(target_path: str) -> dict[str, str]:
    target = Path(target_path)
    if target.is_dir() and any(target.iterdir()):
        return {
            "component": "tts-model",
            "status": "ok",
            "message": f"TTS model ready at {target}.",
        }

    target.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the Qwen TTS model"
        ) from exc

    snapshot_download(
        repo_id=DEFAULT_QWEN_TTS_REPO,
        local_dir=str(target),
    )
    return {
        "component": "tts-model",
        "status": "ok",
        "message": f"Downloaded TTS model to {target}.",
    }


def _download_file(url: str, target: Path) -> None:
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)

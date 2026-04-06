from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from backend.types import RuntimeConfig


ULTRALYTICS_ASSET_BASE = "https://github.com/ultralytics/assets/releases/latest/download"
DEFAULT_FISH_TTS_REPO = "fishaudio/s1-mini"
REQUIRED_FISH_TTS_FILES = (
    "config.json",
    "model.pth",
    "codec.pth",
    "special_tokens.json",
    "tokenizer.tiktoken",
)


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
    if target.exists() and all((target / name).exists() for name in REQUIRED_FISH_TTS_FILES):
        return {
            "component": "tts-model",
            "status": "ok",
            "message": f"TTS model ready at {target}.",
        }

    target.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the Fish Audio S1 Mini model"
        ) from exc

    try:
        snapshot_download(
            repo_id=DEFAULT_FISH_TTS_REPO,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=list(REQUIRED_FISH_TTS_FILES),
            token=os.getenv("HF_TOKEN"),
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            "Fish Audio S1 Mini is a gated Hugging Face model. "
            "First accept the terms at https://huggingface.co/fishaudio/s1-mini, "
            "then run `hf auth login` or set `HF_TOKEN` before starting Momo."
        ) from exc
    return {
        "component": "tts-model",
        "status": "ok",
        "message": f"Downloaded TTS model to {target}.",
    }


def _download_file(url: str, target: Path) -> None:
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)

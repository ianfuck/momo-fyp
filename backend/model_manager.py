from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from backend.tts.model_profiles import resolve_tts_model_profile
from backend.types import RuntimeConfig


ULTRALYTICS_ASSET_BASE = "https://github.com/ultralytics/assets/releases/latest/download"


def ensure_runtime_models(config: RuntimeConfig, *, vision_only: bool = False) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    checks.append(_ensure_yolo_asset(config.yolo_model_path))
    checks.append(_ensure_yolo_asset(config.yolo_pose_model_path))
    if not vision_only:
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
    profile = resolve_tts_model_profile(target_path)
    if target.exists() and all((target / name).exists() for name in profile.required_model_files):
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
            "huggingface_hub is required to download TTS models"
        ) from exc

    try:
        snapshot_download(
            repo_id=profile.repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=list(profile.required_model_files),
            token=os.getenv("HF_TOKEN"),
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            f"{profile.display_name} is a gated Hugging Face model. "
            f"First accept the terms at {profile.huggingface_url}, "
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

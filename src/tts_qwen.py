"""TTS: Qwen3-TTS（torch + qwen-tts 為專案預設依賴）；僅在載入失敗時退回雜訊 stub。"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from src.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _ensure_wav_ref(ref_audio: Path) -> Path:
    """qwen-tts / librosa 對 m4a 支援不一，有 ffmpeg 時轉成單聲道 wav。"""
    suf = ref_audio.suffix.lower()
    if suf not in (".m4a", ".aac", ".mp4", ".mp3"):
        return ref_audio
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("未安裝 ffmpeg，參考音為 %s，若克隆異常請安裝 ffmpeg 或改放 .wav", suf)
        return ref_audio
    out = (PROJECT_ROOT / "tmp" / "ref_voice_for_tts.wav").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(ref_audio),
                "-ar",
                "24000",
                "-ac",
                "1",
                "-f",
                "wav",
                str(out),
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
        if r.returncode == 0 and out.exists() and out.stat().st_size > 256:
            logger.info("參考音已轉為 WAV：%s", out)
            return out
        logger.warning(
            "ffmpeg 轉檔失敗（code=%s），沿用原檔：%s",
            r.returncode,
            (r.stderr or b"").decode(errors="ignore")[:300],
        )
    except Exception as e:
        logger.warning("ffmpeg 轉檔例外，沿用原檔：%s", e)
    return ref_audio


def _pin_hf_env_to_project(cache: Path) -> None:
    """讓 huggingface_hub / transformers 預設快取落在專案下，避免讀到 ~/.cache 裡的半成品快照。"""
    root = cache.resolve()
    root.mkdir(parents=True, exist_ok=True)
    hub = root / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub)


def _ensure_full_local_snapshot(model_id: str, cache: Path) -> Path:
    """
    qwen_tts 的 Qwen3TTSModel.from_pretrained 只把 **kwargs 傳給 AutoModel，
    AutoProcessor.from_pretrained 卻沒帶 cache_dir，會改去 ~/.cache 找檔，容易對不上
    權重目錄而出現缺 speech_tokenizer/preprocessor_config.json。

    解法：snapshot_download 到專案內固定目錄，再以「本機路徑」載入，processor 與 weights 同源。
    """
    from huggingface_hub import get_token, snapshot_download

    safe = model_id.replace("/", "__")
    local = (cache / "hf_snapshots" / safe).resolve()
    marker = local / "speech_tokenizer" / "preprocessor_config.json"
    if marker.exists():
        logger.info("使用既有本機快照：%s", local)
        return local

    local.mkdir(parents=True, exist_ok=True)
    logger.info("正在下載 Qwen3-TTS 完整檔至 %s（首次較久）…", local)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=get_token(),
    )
    if not marker.exists():
        raise RuntimeError(
            f"下載後仍缺少 {marker}。請刪除目錄後重試：{local}，並確認網路與 HF_TOKEN（若需要）。"
        )
    return local


def _from_pretrained_attempts(model_id: str, cache: Path) -> Any:
    import torch
    from qwen_tts import Qwen3TTSModel

    cache.mkdir(parents=True, exist_ok=True)
    _pin_hf_env_to_project(cache)
    local_path = str(_ensure_full_local_snapshot(model_id, cache))

    attempts: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        attempts.append(
            {
                "device_map": "cuda:0",
                "dtype": torch.float16,
                "attn_implementation": "sdpa",
            }
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # float32 先試：少數環境 bfloat16 + MPS 會失敗或異常慢
        attempts.append(
            {
                "device_map": "mps",
                "dtype": torch.float32,
                "attn_implementation": "sdpa",
            }
        )
        attempts.append(
            {
                "device_map": {"": "mps"},
                "dtype": torch.float32,
                "attn_implementation": "sdpa",
            }
        )
        attempts.append(
            {
                "device_map": "mps",
                "dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
            }
        )
    attempts.append({"device_map": "cpu", "dtype": torch.float32, "attn_implementation": "sdpa"})
    attempts.append({"device_map": "cpu", "dtype": torch.float32})

    last_err: Exception | None = None
    for extra in attempts:
        try:
            return Qwen3TTSModel.from_pretrained(local_path, **extra)
        except Exception as e:
            last_err = e
            logger.warning("Qwen3TTSModel.from_pretrained 重試：%s 失敗：%s", extra, e)
    assert last_err is not None
    raise last_err


def _qwen_weight_devices(tts_model: Any) -> str:
    """從已載入的 Qwen3TTSModel 讀出參數實際所在 device（避免只憑 is_available 誤報 mps）。"""
    inner = getattr(tts_model, "model", None)
    if inner is None:
        return "unknown"
    devs: set[str] = set()
    try:
        for p in inner.parameters():
            devs.add(str(p.device))
    except Exception:
        return "unknown"
    if not devs:
        return "unknown"
    return "+".join(sorted(devs))


class TTSBackend:
    def __init__(self) -> None:
        self._model = None
        self._clone_prompt = None

    def ensure_model(self, model_id: str, cache_dir: str, ref_audio_rel: str, ref_text_rel: str) -> str:
        try:
            import torch
            import qwen_tts  # noqa: F401
        except Exception as e:
            self._model = None
            self._clone_prompt = None
            logger.error("無法匯入 torch / qwen_tts（請 uv sync）：%s", e)
            return f"stub(no_qwen:{e.__class__.__name__})"

        try:
            cache = (PROJECT_ROOT / cache_dir).resolve()
            ref_audio = (PROJECT_ROOT / ref_audio_rel).resolve()
            ref_text = (PROJECT_ROOT / ref_text_rel).read_text(encoding="utf-8")
            ref_audio = _ensure_wav_ref(ref_audio)
            self._model = _from_pretrained_attempts(model_id, cache)
            self._clone_prompt = self._model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text=ref_text,
            )
            actual = _qwen_weight_devices(self._model)
            logger.info(
                "Qwen3-TTS 已載入；權重實際所在裝置=%s（若為 cpu 但本機有 MPS，代表 MPS 載入失敗已回退）",
                actual,
            )
            return f"qwen3(weights={actual})"
        except Exception as e:
            self._model = None
            self._clone_prompt = None
            logger.exception("Qwen3-TTS 載入失敗，改用雜訊 stub：%s", e)
            return f"stub({e.__class__.__name__})"

    def synthesize_to_wav(self, text: str, out_path: Path) -> str:
        """回傳 \"qwen\" 或 \"stub\"，供上層寫 log。"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self._model is not None and self._clone_prompt is not None:
            import soundfile as sf

            try:
                wavs, sr = self._model.generate_voice_clone(
                    text=text,
                    language="Chinese",
                    voice_clone_prompt=self._clone_prompt,
                )
                raw = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
                if hasattr(raw, "detach"):
                    raw = raw.detach().cpu().float().numpy()
                arr = np.asarray(raw, dtype=np.float32)
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=-1) if arr.shape[-1] <= 4 else arr.reshape(-1)
                else:
                    arr = np.squeeze(arr)
                arr = np.clip(arr.astype(np.float32, copy=False), -1.0, 1.0)
                sr_i = int(np.asarray(sr).reshape(-1)[0])
                if arr.size < 256:
                    raise RuntimeError(f"TTS 輸出過短（{arr.size} samples），可能合成失敗")
                sf.write(str(out_path), arr, sr_i)
                logger.debug("TTS 已寫入 %s samples @ %s Hz", arr.size, sr_i)
                return "qwen"
            except Exception:
                logger.exception("generate_voice_clone 失敗，改用雜訊 stub")
        _write_noise_wav(out_path, duration_s=min(2.5, 0.06 * max(12, len(text))))
        return "stub"


def _write_noise_wav(path: Path, duration_s: float) -> None:
    sr = 22050
    n = int(sr * duration_s)
    rng = np.random.default_rng(0)
    x = (rng.random(n).astype(np.float32) * 2 - 1) * 0.04
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((x * 32767).astype(np.int16).tobytes())

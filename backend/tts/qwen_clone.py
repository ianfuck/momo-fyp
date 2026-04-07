from __future__ import annotations

import hashlib
import importlib
import importlib.util
import platform
import re
import shutil
import subprocess
import threading
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from backend.device_utils import backend_label_for_device, get_tts_device

_FISH_CONTROL_TOKEN_RE = re.compile(r"\([^()]{1,40}\)")
_REQUIRED_MODEL_FILES = (
    "config.json",
    "model.pth",
    "codec.pth",
    "special_tokens.json",
    "tokenizer.tiktoken",
)


def _fish_configs_dir() -> Path:
    spec = importlib.util.find_spec("fish_speech")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("fish_speech package is not installed")
    return Path(spec.submodule_search_locations[0]) / "configs"


def _load_fish_decoder_model(config_name: str, checkpoint_path: str, device: str):
    import hydra
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(_fish_configs_dir())):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(checkpoint_path, map_location=device, mmap=True, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in key for key in state_dict):
        state_dict = {
            key.replace("generator.", ""): value
            for key, value in state_dict.items()
            if "generator." in key
        }
    try:
        model.load_state_dict(state_dict, strict=False, assign=True)
    except TypeError:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


class FishCloneTTS:
    def __init__(
        self,
        model_path: str,
        ref_audio_path: str,
        ref_text_path: str,
        clone_voice_enabled: bool = True,
        device_mode: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path
        self.clone_voice_enabled = clone_voice_enabled
        self.loaded = False
        self._model_manager = None
        self._engine = None
        self._lock = threading.Lock()
        self.available = Path(model_path).exists() and (
            not clone_voice_enabled or (Path(ref_audio_path).exists() and Path(ref_text_path).exists())
        )
        self.device = get_tts_device(device_mode)
        self.device_backend = backend_label_for_device(self.device)

    def preload(self) -> None:
        with self._lock:
            self._ensure_model()

    def _ensure_model(self) -> None:
        if self._engine is not None:
            return
        if not self.available:
            raise FileNotFoundError("TTS model or reference files are missing")

        missing = [name for name in _REQUIRED_MODEL_FILES if not (Path(self.model_path) / name).exists()]
        if missing:
            raise FileNotFoundError(f"Fish Audio model is incomplete at {self.model_path}: missing {', '.join(missing)}")

        try:
            inference_engine_module = importlib.import_module("fish_speech.inference_engine")
            text2semantic_module = importlib.import_module("fish_speech.models.text2semantic.inference")
        except ImportError as exc:
            raise RuntimeError(
                "Fish Speech runtime dependencies are missing. Run `uv sync` to install the Fish Audio TTS stack."
            ) from exc

        import torch

        precision = torch.half if self.device.startswith("cuda") else torch.bfloat16
        llama_queue = text2semantic_module.launch_thread_safe_queue(
            checkpoint_path=self.model_path,
            device=self.device,
            precision=precision,
            compile=False,
        )
        decoder_model = _load_fish_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(Path(self.model_path) / "codec.pth"),
            device=self.device,
        )
        self._engine = inference_engine_module.TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=False,
        )
        self.loaded = True

    def synthesize(self, text: str, output_path: str) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._ensure_model()
            request = self._build_request(text)
            wav, sr = self._run_inference(request)
        wav = self._select_best_waveform(wav, sr)
        if self._looks_broken(wav, sr) and platform.system() == "Darwin":
            return self._fallback_say_tts(text, output)
        sf.write(output, wav, sr)
        return str(output)

    def _build_request(self, text: str):
        schema_module = importlib.import_module("fish_speech.utils.schema")
        references = []
        if self.clone_voice_enabled:
            ref_audio = self._prepare_reference_audio_bytes()
            ref_text = self._load_reference_text() or ""
            references = [schema_module.ServeReferenceAudio(audio=ref_audio, text=ref_text)]
        return schema_module.ServeTTSRequest(
            text=text,
            references=references,
            reference_id=None,
            use_memory_cache="on",
            chunk_length=200,
            max_new_tokens=1024,
            top_p=0.8,
            repetition_penalty=1.1,
            temperature=0.7,
            format="wav",
        )

    def _run_inference(self, request) -> tuple[np.ndarray, int]:
        final_audio: np.ndarray | None = None
        final_sr = 24000
        for result in self._engine.inference(request):
            if result.code == "error":
                raise RuntimeError(str(result.error))
            if result.code == "final" and isinstance(result.audio, tuple):
                final_sr, final_audio = result.audio
        if final_audio is None:
            raise RuntimeError("Fish Audio did not return synthesized audio")
        return np.asarray(final_audio, dtype=np.float32), int(final_sr)

    def _prepare_reference_audio_bytes(self) -> bytes:
        prepared = self._prepare_reference_audio_path()
        return prepared.read_bytes()

    def _prepare_reference_audio_path(self) -> Path:
        source = Path(self.ref_audio_path)
        output = self._reference_cache_path(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            return output
        wav, _ = self._load_reference_audio()
        if len(wav) > 24000 * 12:
            start = max(0, (len(wav) - (24000 * 12)) // 2)
            wav = wav[start:start + 24000 * 12]
        wav = self._normalize_waveform(wav)
        sf.write(output, wav, 24000)
        return output

    def _reference_cache_path(self, source: Path) -> Path:
        resolved = source.expanduser().resolve(strict=False)
        stat = resolved.stat() if resolved.exists() else None
        fingerprint = "|".join(
            [
                str(resolved),
                str(stat.st_size if stat else 0),
                str(stat.st_mtime_ns if stat else 0),
            ]
        )
        digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
        return Path("tmp") / f"ref_voice_{digest}_24k.wav"

    def _load_reference_audio(self) -> tuple[np.ndarray, int]:
        source = Path(self.ref_audio_path)
        if source.suffix.lower() != ".wav":
            converted = self._convert_reference_audio_with_ffmpeg(source)
            if converted is not None:
                return self._load_audio_with_librosa(str(converted))
        try:
            return self._load_audio_with_librosa(str(source))
        except Exception as exc:
            converted = self._convert_reference_audio_with_ffmpeg(source)
            if converted is None:
                raise RuntimeError(
                    "Failed to decode TTS reference audio. "
                    "On Windows, convert the reference file to WAV or install ffmpeg so .m4a can be decoded."
                ) from exc
            return self._load_audio_with_librosa(str(converted))

    def _load_audio_with_librosa(self, path: str) -> tuple[np.ndarray, int]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            return librosa.load(path, sr=24000, mono=True)

    def _load_reference_text(self) -> str | None:
        raw = Path(self.ref_text_path).read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return " ".join(raw.split())

    def _convert_reference_audio_with_ffmpeg(self, source: Path) -> Path | None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return None
        converted = Path("tmp/ref_voice_source.wav")
        converted.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "24000",
                str(converted),
            ],
            check=True,
            capture_output=True,
        )
        return converted if converted.exists() else None

    def _normalize_waveform(self, wav: np.ndarray) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 0:
            wav = wav / peak * 0.78
        return wav

    def _polish_waveform(self, wav: np.ndarray, sr: int) -> np.ndarray:
        return self._finalize_waveform(wav, sr, repair_spikes=True)

    def _select_best_waveform(self, wav: np.ndarray, sr: int) -> np.ndarray:
        direct = self._finalize_waveform(wav, sr, repair_spikes=False)
        repaired = self._finalize_waveform(wav, sr, repair_spikes=True)
        return repaired if self._click_score(repaired) <= self._click_score(direct) else direct

    def _finalize_waveform(self, wav: np.ndarray, sr: int, repair_spikes: bool) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            return wav

        wav = wav - float(np.mean(wav))
        wav = np.tanh(wav * 1.2)
        if repair_spikes:
            wav = self._suppress_transient_spikes(wav)
        wav = self._normalize_waveform(wav)
        if repair_spikes:
            wav = self._smooth_residual_clicks(wav, jump_threshold=0.16, dev_scale=0.06)

        fade_samples = min(max(1, int(sr * 0.012)), wav.size // 2)
        if fade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            wav[:fade_samples] *= fade_in
            wav[-fade_samples:] *= fade_out
        return wav

    def _click_score(self, wav: np.ndarray) -> tuple[int, int, float]:
        diffs = np.abs(np.diff(wav))
        severe = int(np.sum(diffs > 0.25))
        moderate = int(np.sum(diffs > 0.18))
        tail = float(np.quantile(diffs, 0.999)) if diffs.size else 0.0
        return severe, moderate, tail

    def _suppress_transient_spikes(self, wav: np.ndarray) -> np.ndarray:
        if wav.size < 5:
            return wav
        repaired = wav.copy()
        for _ in range(3):
            diffs = np.diff(repaired)
            median_abs = max(1e-5, float(np.median(np.abs(diffs))))
            jump_threshold = min(max(0.1, median_abs * 8.0), 0.22)
            dev_scale = max(0.08, median_abs * 8.0)

            repaired = self._repair_short_spike_spans(repaired, jump_threshold, dev_scale)

            for idx in range(2, repaired.size - 2):
                left = repaired[idx] - repaired[idx - 1]
                right = repaired[idx + 1] - repaired[idx]
                neighborhood = np.array(
                    [repaired[idx - 2], repaired[idx - 1], repaired[idx + 1], repaired[idx + 2]],
                    dtype=np.float32,
                )
                local_median = float(np.median(neighborhood))
                local_mad = max(1e-5, float(np.median(np.abs(neighborhood - local_median))))
                if abs(left) < jump_threshold or abs(right) < jump_threshold:
                    continue
                if left * right >= 0 and abs(repaired[idx] - local_median) < max(dev_scale, local_mad * 6.0):
                    continue
                repaired[idx] = (repaired[idx - 1] + repaired[idx + 1]) * 0.5

            for idx in range(2, repaired.size - 3):
                pair = repaired[idx:idx + 2]
                context = np.array(
                    [repaired[idx - 2], repaired[idx - 1], repaired[idx + 2], repaired[idx + 3]],
                    dtype=np.float32,
                )
                local_median = float(np.median(context))
                local_mad = max(1e-5, float(np.median(np.abs(context - local_median))))
                if (
                    abs(repaired[idx] - repaired[idx - 1]) > jump_threshold
                    and abs(repaired[idx + 2] - repaired[idx + 1]) > jump_threshold
                    and np.max(np.abs(pair - local_median)) > max(dev_scale, local_mad * 6.0)
                ):
                    repaired[idx] = repaired[idx - 1] * 0.67 + repaired[idx + 2] * 0.33
                    repaired[idx + 1] = repaired[idx - 1] * 0.33 + repaired[idx + 2] * 0.67
            repaired = self._smooth_residual_clicks(repaired, jump_threshold, dev_scale)
        return repaired

    def _repair_short_spike_spans(self, wav: np.ndarray, jump_threshold: float, dev_scale: float) -> np.ndarray:
        repaired = wav.copy()
        diff_idx = np.where(np.abs(np.diff(repaired)) > jump_threshold)[0]
        if diff_idx.size == 0:
            return repaired

        spans: list[tuple[int, int]] = []
        start = int(diff_idx[0])
        prev = int(diff_idx[0])
        for idx in diff_idx[1:]:
            idx = int(idx)
            if idx == prev + 1:
                prev = idx
                continue
            spans.append((start, prev))
            start = prev = idx
        spans.append((start, prev))

        for diff_start, diff_end in spans:
            sample_start = diff_start + 1
            sample_end = diff_end
            if sample_end < sample_start:
                continue
            span_len = sample_end - sample_start + 1
            if span_len > 3 or sample_start < 2 or sample_end >= repaired.size - 2:
                continue

            left_anchor = sample_start - 1
            right_anchor = sample_end + 1
            context = np.concatenate(
                [
                    repaired[max(0, left_anchor - 2):left_anchor + 1],
                    repaired[right_anchor:min(repaired.size, right_anchor + 3)],
                ]
            )
            local_median = float(np.median(context))
            local_mad = max(1e-5, float(np.median(np.abs(context - local_median))))
            segment = repaired[sample_start:sample_end + 1]
            if np.max(np.abs(segment - local_median)) < max(dev_scale, local_mad * 5.0):
                continue

            repaired[sample_start:sample_end + 1] = np.linspace(
                repaired[left_anchor],
                repaired[right_anchor],
                span_len + 2,
                dtype=np.float32,
            )[1:-1]
        return repaired

    def _smooth_residual_clicks(self, wav: np.ndarray, jump_threshold: float, dev_scale: float) -> np.ndarray:
        repaired = wav.copy()
        for idx in range(2, repaired.size - 2):
            local = repaired[idx - 2:idx + 3]
            local_median = float(np.median(local))
            if abs(repaired[idx] - local_median) < dev_scale:
                continue
            if max(abs(repaired[idx] - repaired[idx - 1]), abs(repaired[idx + 1] - repaired[idx])) < jump_threshold * 0.85:
                continue
            repaired[idx] = (
                repaired[idx - 1] * 0.2
                + repaired[idx] * 0.45
                + repaired[idx + 1] * 0.2
                + local_median * 0.15
            )
        return repaired

    def _looks_broken(self, wav: np.ndarray, sr: int) -> bool:
        duration = len(wav) / max(1, sr)
        mean_abs = float(np.mean(np.abs(wav))) if wav.size else 0.0
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        return duration < 0.8 or mean_abs < 0.01 or peak < 0.08

    def _fallback_say_tts(self, text: str, output: Path) -> str:
        aiff_path = output.with_suffix(".aiff")
        fallback_text = self._strip_control_tokens(text)
        subprocess.run(
            ["say", "-v", "Tingting", "-o", str(aiff_path), fallback_text],
            check=True,
            capture_output=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            wav, sr = librosa.load(str(aiff_path), sr=24000, mono=True)
        wav = self._normalize_waveform(wav)
        sf.write(output, wav, sr)
        aiff_path.unlink(missing_ok=True)
        return str(output)

    def _strip_control_tokens(self, text: str) -> str:
        stripped = _FISH_CONTROL_TOKEN_RE.sub("", text)
        return " ".join(stripped.split())


QwenCloneTTS = FishCloneTTS

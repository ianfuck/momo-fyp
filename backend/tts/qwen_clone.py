from __future__ import annotations

import json
import hashlib
import importlib
import importlib.util
import contextlib
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from backend.device_utils import backend_label_for_device, get_tts_device
from backend.runtime_shutdown import shutdown_requested
from backend.tts.model_profiles import resolve_tts_model_profile
from backend.tts.provider_runtimes import KokoroChineseTTS, MeloChineseTTS
from backend.tts.qwen_runtime import QwenVoiceCloneTTS
from backend.tts.semantic_runtime import (
    SemanticBenchmarkResult,
    benchmark_plans_for_current_host,
    cleanup_torch_memory,
    make_semantic_queue,
)

_FISH_CONTROL_TOKEN_RE = re.compile(r"\([^()]{1,40}\)")
_FISH_PUNCTUATION_RE = re.compile(r"[，。！？!?;；:,、…]")


@dataclass(frozen=True)
class TTSAutoBenchmarkSelection:
    tts: "FishCloneTTS"
    result: SemanticBenchmarkResult
    results: list[SemanticBenchmarkResult]


class BenchmarkShutdownRequested(RuntimeError):
    pass


@dataclass(frozen=True)
class TTSBenchmarkCandidate:
    name: str
    device_mode: str
    semantic_dispatch_mode: str
    precision_mode: str


_BENCHMARK_REQUEST_OVERRIDES = {
    "chunk_length": 32,
    "max_new_tokens": 24,
    "temperature": 0.35,
}
_BENCHMARK_POLL_INTERVAL_SEC = 0.2
_BENCHMARK_TEMP_DIR_CLEANUP_RETRIES = 5
_BENCHMARK_TEMP_DIR_CLEANUP_DELAY_SEC = 0.2


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

    config_path = Path(config_name)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    if config_path.exists():
        config_dir = config_path.parent
        resolved_name = config_path.stem
    else:
        config_dir = _fish_configs_dir()
        resolved_name = config_name
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.resolve())):
        cfg = compose(config_name=resolved_name)

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
        kokoro_voice: str | None = None,
        device_mode: str = "auto",
        semantic_dispatch_mode: str = "single",
        precision_mode: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path
        self.model_profile = resolve_tts_model_profile(model_path)
        self.clone_voice_enabled = clone_voice_enabled and self.model_profile.supports_voice_clone
        self.kokoro_voice = kokoro_voice or self.model_profile.default_voice
        self.semantic_dispatch_mode = semantic_dispatch_mode
        self.precision_mode = precision_mode or _default_precision_mode_for_device(get_tts_device(device_mode))
        self.loaded = False
        self._model_manager = None
        self._engine = None
        self._lock = threading.Lock()
        self._delegate = None
        self.available = Path(model_path).exists() and (
            not self.clone_voice_enabled or (Path(ref_audio_path).exists() and Path(ref_text_path).exists())
        )
        self.device = get_tts_device(device_mode)
        self.device_backend = backend_label_for_device(self.device)
        if self.model_profile.runtime_family == "qwen":
            self._delegate = QwenVoiceCloneTTS(
                model_path,
                ref_audio_path,
                ref_text_path,
                clone_voice_enabled=self.clone_voice_enabled,
                device_mode=device_mode,
                precision_mode=self.precision_mode,
            )
        elif self.model_profile.runtime_family == "kokoro":
            self._delegate = KokoroChineseTTS(
                model_path,
                ref_audio_path,
                ref_text_path,
                model_profile=self.model_profile,
                clone_voice_enabled=self.clone_voice_enabled,
                voice=self.kokoro_voice,
                device_mode=device_mode,
                precision_mode=self.precision_mode,
            )
        elif self.model_profile.runtime_family == "melo":
            self._delegate = MeloChineseTTS(
                model_path,
                ref_audio_path,
                ref_text_path,
                model_profile=self.model_profile,
                clone_voice_enabled=self.clone_voice_enabled,
                device_mode=device_mode,
                precision_mode=self.precision_mode,
            )
        if self._delegate is not None:
            self.device = self._delegate.device
            self.device_backend = self._delegate.device_backend
            self.precision_mode = self._delegate.precision_mode
            self.semantic_dispatch_mode = getattr(self._delegate, "semantic_dispatch_mode", self.semantic_dispatch_mode)
            self.available = self._delegate.available

    def set_reference_paths(self, ref_audio_path: str, ref_text_path: str) -> None:
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path
        if self._delegate is not None:
            self._delegate.set_reference_paths(ref_audio_path, ref_text_path)
            self.available = self._delegate.available
            return
        self.available = Path(self.model_path).exists() and (
            not self.clone_voice_enabled or (Path(ref_audio_path).exists() and Path(ref_text_path).exists())
        )

    def set_kokoro_voice(self, voice: str) -> None:
        self.kokoro_voice = voice
        if self._delegate is not None and hasattr(self._delegate, "set_voice"):
            self._delegate.set_voice(voice)

    def preload(self) -> None:
        if self._delegate is not None:
            self._delegate.preload()
            self.loaded = self._delegate.loaded
            return
        with self._lock:
            self._ensure_model()

    def unload(self) -> None:
        with self._lock:
            if self._delegate is not None:
                self._delegate.unload()
                self.loaded = False
                return

            engine = self._engine
            self._engine = None
            self._model_manager = None
            self.loaded = False
            if engine is not None:
                queue = getattr(engine, "llama_queue", None)
                if queue is not None:
                    try:
                        queue.put_nowait(None)
                    except Exception:
                        pass
                decoder_model = getattr(engine, "decoder_model", None)
                if decoder_model is not None and hasattr(decoder_model, "to"):
                    try:
                        decoder_model.to("cpu")
                    except Exception:
                        pass
            cleanup_torch_memory()

    def _ensure_model(self) -> None:
        if self._engine is not None:
            return
        if not self.available:
            raise FileNotFoundError("TTS model or reference files are missing")

        missing = [
            name
            for name in self.model_profile.required_model_files
            if not (Path(self.model_path) / name).exists()
        ]
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

        precision = _torch_dtype_for_precision_mode(torch, self.precision_mode)
        active_dispatch_mode = self.semantic_dispatch_mode
        try:
            llama_queue = make_semantic_queue(
                checkpoint_path=self.model_path,
                device=self.device,
                precision=precision,
                semantic_dispatch_mode=active_dispatch_mode,
                compile=False,
            )
        except Exception:
            if active_dispatch_mode != "auto":
                raise
            active_dispatch_mode = "single"
            llama_queue = text2semantic_module.launch_thread_safe_queue(
                checkpoint_path=self.model_path,
                device=self.device,
                precision=precision,
                compile=False,
            )
        self.semantic_dispatch_mode = active_dispatch_mode
        self.precision_mode = _precision_mode_name(precision)
        decoder_model = _load_fish_decoder_model(
            config_name=self.model_profile.decoder_config_name,
            checkpoint_path=str(Path(self.model_path) / self.model_profile.decoder_checkpoint_name),
            device=self.device,
        )
        self._engine = inference_engine_module.TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=False,
        )
        self.loaded = True

    def synthesize(self, text: str, output_path: str, *, request_overrides: dict | None = None) -> str:
        if self._delegate is not None:
            result = self._delegate.synthesize(text, output_path, request_overrides=request_overrides)
            self.loaded = self._delegate.loaded
            return result
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._ensure_model()
            request = self._build_request(text, request_overrides=request_overrides)
            wav, sr = self._run_inference(request)
        wav = self._select_best_waveform(wav, sr)
        if self._looks_broken(wav, sr) and platform.system() == "Darwin":
            return self._fallback_say_tts(text, output)
        sf.write(output, wav, sr)
        return str(output)

    @classmethod
    def benchmark_auto_profiles(
        cls,
        model_path: str,
        ref_audio_path: str,
        ref_text_path: str,
        *,
        clone_voice_enabled: bool,
        kokoro_voice: str | None = None,
        sample_text: str = "測試。",
    ) -> TTSAutoBenchmarkSelection | None:
        profile = resolve_tts_model_profile(model_path)
        if not profile.supports_startup_benchmark:
            tts = cls(
                model_path,
                ref_audio_path,
                ref_text_path,
                clone_voice_enabled=clone_voice_enabled,
                kokoro_voice=kokoro_voice,
                device_mode="auto",
                precision_mode=None,
            )
            result = SemanticBenchmarkResult(
                name=profile.key,
                device_mode="auto",
                semantic_dispatch_mode="single",
                elapsed_ms=0,
                ok=True,
                preload_ms=0,
                synth_ms=0,
                precision_mode=tts.precision_mode,
                peak_vram_mb=None,
            )
            return TTSAutoBenchmarkSelection(tts=tts, result=result, results=[result])
        plans = benchmark_plans_for_current_host()
        if profile.runtime_family in {"qwen", "kokoro", "melo"}:
            if profile.runtime_family == "qwen":
                plans = [plan for plan in plans if plan.semantic_dispatch_mode == "single"]
            else:
                plans = [plan for plan in plans if plan.semantic_dispatch_mode == "single"]
        if not plans:
            return None
        results: list[SemanticBenchmarkResult] = []
        best_result: SemanticBenchmarkResult | None = None
        for candidate in _benchmark_candidates_for_profile(profile.key, plans):
            print(
                f"[startup] tts benchmark running candidate={candidate.name} device={candidate.device_mode} "
                f"semantic={candidate.semantic_dispatch_mode} precision={candidate.precision_mode}",
                flush=True,
            )
            result = _run_benchmark_candidate_subprocess(
                plan=candidate,
                model_path=model_path,
                ref_audio_path=ref_audio_path,
                ref_text_path=ref_text_path,
                clone_voice_enabled=clone_voice_enabled,
                kokoro_voice=kokoro_voice,
                sample_text=sample_text,
            )
            detail = f" detail={result.detail[:200]}" if result.detail else ""
            print(
                f"[startup] tts benchmark candidate={result.name} status={'ok' if result.ok else 'error'} "
                f"preload_ms={result.preload_ms} synth_ms={result.synth_ms} total_ms={result.elapsed_ms} "
                f"peak_vram_mb={result.peak_vram_mb} semantic={result.semantic_dispatch_mode} "
                f"precision={result.precision_mode}{detail}",
                flush=True,
            )
            if result.ok and (best_result is None or _benchmark_sort_key(result) < _benchmark_sort_key(best_result)):
                best_result = result
            results.append(result)
        if best_result is None:
            return None
        best_tts = cls(
            model_path,
            ref_audio_path,
            ref_text_path,
            clone_voice_enabled=clone_voice_enabled,
            kokoro_voice=kokoro_voice,
            device_mode=best_result.device_mode,
            semantic_dispatch_mode=best_result.semantic_dispatch_mode,
            precision_mode=best_result.precision_mode,
        )
        return TTSAutoBenchmarkSelection(tts=best_tts, result=best_result, results=results)

    def format_emotion_text(self, text: str, emotion: str) -> str:
        if self._delegate is not None:
            return self._delegate.format_emotion_text(text, emotion)
        return self.model_profile.format_emotion_text(text, emotion)

    @property
    def emotion_tags(self) -> tuple[str, ...]:
        if self._delegate is not None:
            return self._delegate.emotion_tags
        return self.model_profile.emotion_tags

    def _build_request(self, text: str, *, request_overrides: dict | None = None):
        schema_module = importlib.import_module("fish_speech.utils.schema")
        references = []
        if self.clone_voice_enabled:
            ref_audio = self._prepare_reference_audio_bytes()
            ref_text = self._load_reference_text() or ""
            references = [schema_module.ServeReferenceAudio(audio=ref_audio, text=ref_text)]
        payload = {
            "text": text,
            "references": references,
            "reference_id": None,
            "use_memory_cache": "on",
            "chunk_length": 200,
            "max_new_tokens": self._estimate_max_new_tokens(text),
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "temperature": 0.7,
            "format": "wav",
        }
        if request_overrides:
            payload.update(request_overrides)
        return schema_module.ServeTTSRequest(
            **payload,
        )

    def _estimate_max_new_tokens(self, text: str) -> int:
        cleaned = self._strip_control_tokens(text)
        if not cleaned:
            return 96
        non_space_chars = sum(1 for char in cleaned if not char.isspace())
        punctuation_count = len(_FISH_PUNCTUATION_RE.findall(cleaned))
        ascii_word_count = len(re.findall(r"[A-Za-z0-9]+", cleaned))
        budget = 48 + (non_space_chars * 10) + (punctuation_count * 6) + (ascii_word_count * 4)
        return max(96, min(512, budget))

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


def _run_benchmark_candidate_subprocess(
    *,
    plan: TTSBenchmarkCandidate,
    model_path: str,
    ref_audio_path: str,
    ref_text_path: str,
    clone_voice_enabled: bool,
    kokoro_voice: str | None = None,
    sample_text: str,
) -> SemanticBenchmarkResult:
    timeout_sec = _benchmark_timeout_sec()
    tmp_dir = Path(tempfile.mkdtemp(prefix="momo_tts_bench_"))
    try:
        result_path = tmp_dir / "result.json"
        stdout_path = tmp_dir / "stdout.log"
        stderr_path = tmp_dir / "stderr.log"
        command = [
            sys.executable,
            "-m",
            "backend.tts.benchmark_worker",
            "--model-path",
            model_path,
            "--ref-audio-path",
            ref_audio_path,
            "--ref-text-path",
            ref_text_path,
            "--device-mode",
            plan.device_mode,
            "--plan-name",
            plan.name,
            "--semantic-dispatch-mode",
            plan.semantic_dispatch_mode,
            "--sample-text",
            sample_text,
            "--result-path",
            str(result_path),
            "--precision-mode",
            plan.precision_mode,
        ]
        if clone_voice_enabled:
            command.append("--clone-voice-enabled")
        if kokoro_voice:
            command.extend(["--kokoro-voice", kokoro_voice])
        popen_kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": None,
            "stderr": None,
            "text": True,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True
        with (
            stdout_path.open("w", encoding="utf-8", errors="ignore") as stdout_file,
            stderr_path.open("w", encoding="utf-8", errors="ignore") as stderr_file,
        ):
            popen_kwargs["stdout"] = stdout_file
            popen_kwargs["stderr"] = stderr_file
            process = subprocess.Popen(command, **popen_kwargs)
            try:
                deadline = time.monotonic() + timeout_sec
                while process.poll() is None:
                    if shutdown_requested():
                        _terminate_benchmark_process(process)
                        raise BenchmarkShutdownRequested("shutdown requested while benchmarking TTS")
                    if time.monotonic() >= deadline:
                        _terminate_benchmark_process(process)
                        return SemanticBenchmarkResult(
                            name=plan.name,
                            device_mode=plan.device_mode,
                            semantic_dispatch_mode=plan.semantic_dispatch_mode,
                            elapsed_ms=-1,
                            ok=False,
                            preload_ms=None,
                            synth_ms=None,
                            precision_mode=plan.precision_mode,
                            peak_vram_mb=None,
                            detail=f"benchmark timed out after {timeout_sec} seconds",
                        )
                    time.sleep(_BENCHMARK_POLL_INTERVAL_SEC)
                _wait_for_process_exit(process, timeout=1.0)
            except BaseException:
                _terminate_benchmark_process(process)
                raise
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            return SemanticBenchmarkResult(**payload)
        detail = _read_benchmark_logs(stdout_path, stderr_path)
        return_code = process.returncode
        if not detail:
            detail = f"benchmark subprocess exited with code {return_code}"
        return SemanticBenchmarkResult(
            name=plan.name,
            device_mode=plan.device_mode,
            semantic_dispatch_mode=plan.semantic_dispatch_mode,
            elapsed_ms=-1,
            ok=False,
            preload_ms=None,
            synth_ms=None,
            precision_mode=plan.precision_mode,
            peak_vram_mb=None,
            detail=detail[-1200:],
        )
    finally:
        _cleanup_benchmark_temp_dir(tmp_dir)


def _benchmark_timeout_sec() -> int:
    raw = os.getenv("MOMO_TTS_BENCHMARK_TIMEOUT_SEC", "120")
    try:
        return max(10, int(raw))
    except ValueError:
        return 120


def _benchmark_candidates_for_profile(profile_key: str, plans) -> list[TTSBenchmarkCandidate]:
    candidates: list[TTSBenchmarkCandidate] = []
    for plan in plans:
        if platform.system() == "Darwin" and profile_key.startswith(("kokoro-82m", "melotts")) and plan.device_mode != "cpu":
            continue
        for precision_mode in _benchmark_precision_modes(profile_key, plan.device_mode):
            candidates.append(
                TTSBenchmarkCandidate(
                    name=f"{plan.name}-{precision_mode}",
                    device_mode=plan.device_mode,
                    semantic_dispatch_mode=plan.semantic_dispatch_mode,
                    precision_mode=precision_mode,
                )
            )
    return candidates


def _benchmark_precision_modes(profile_key: str, device_mode: str) -> tuple[str, ...]:
    if profile_key.startswith("qwen3-tts"):
        if device_mode == "gpu":
            return ("float16", "float32")
        return ("float32",)
    if profile_key.startswith("kokoro-82m") or profile_key.startswith("melotts"):
        return ("float32",)
    if device_mode == "gpu":
        return ("float16", "float32")
    if device_mode == "cpu":
        return ("bfloat16", "float32")
    return ("float32",)


def _benchmark_sort_key(result: SemanticBenchmarkResult) -> tuple[float, int]:
    synth_ms = float(result.synth_ms) if result.synth_ms is not None and result.synth_ms >= 0 else float("inf")
    total_ms = result.elapsed_ms if result.elapsed_ms >= 0 else sys.maxsize
    return synth_ms, total_ms


def _default_precision_mode_for_device(device: str) -> str:
    return "float16" if device.startswith("cuda") else "bfloat16"


def _precision_mode_name(dtype) -> str:
    text = str(dtype)
    if text.startswith("torch."):
        return text.split(".", 1)[1]
    return text


def _torch_dtype_for_precision_mode(torch, precision_mode: str):
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[precision_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported TTS precision mode: {precision_mode}") from exc


def _cleanup_benchmark_temp_dir(path: Path) -> None:
    retries = _BENCHMARK_TEMP_DIR_CLEANUP_RETRIES if os.name == "nt" else 1
    last_error: OSError | None = None
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(_BENCHMARK_TEMP_DIR_CLEANUP_DELAY_SEC)
    if last_error is not None:
        print(
            f"[startup] tts benchmark cleanup skipped path={path} detail={last_error}",
            flush=True,
        )


def _terminate_benchmark_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        with contextlib.suppress(Exception):
            process.terminate()
        _wait_for_process_exit(process, timeout=3.0)
        if process.poll() is None:
            with contextlib.suppress(Exception):
                process.kill()
            _wait_for_process_exit(process, timeout=1.0)
        return

    with contextlib.suppress(Exception):
        os.killpg(process.pid, signal.SIGTERM)
    _wait_for_process_exit(process, timeout=3.0)
    if process.poll() is None:
        with contextlib.suppress(Exception):
            os.killpg(process.pid, signal.SIGKILL)
        _wait_for_process_exit(process, timeout=1.0)


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float) -> None:
    with contextlib.suppress(Exception):
        process.wait(timeout=timeout)


def _read_benchmark_logs(stdout_path: Path, stderr_path: Path) -> str:
    chunks: list[str] = []
    for path in (stderr_path, stdout_path):
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        if raw:
            chunks.append(raw)
    return "\n".join(chunks)

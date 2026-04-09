from __future__ import annotations

import os
import platform
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.device_utils import backend_label_for_device, get_tts_device
from backend.tts.model_profiles import TTSModelProfile
from backend.tts.semantic_runtime import cleanup_torch_memory

_KOKORO_SPLIT_RE = re.compile(r"\s+")


def _provider_device(device: str) -> str:
    return "cuda" if device.startswith("cuda") else device


def _safe_provider_device(device_mode: str) -> str:
    device = get_tts_device(device_mode)
    if platform.system() == "Darwin" and device == "mps":
        return "cpu"
    return device


def _concat_segments(segments: list[np.ndarray], *, sample_rate: int, gap_ms: int = 50) -> np.ndarray:
    if not segments:
        return np.zeros(0, dtype=np.float32)
    if len(segments) == 1:
        return np.asarray(segments[0], dtype=np.float32)
    gap = np.zeros(int(sample_rate * (gap_ms / 1000.0)), dtype=np.float32)
    merged: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        merged.append(np.asarray(segment, dtype=np.float32))
        if index != len(segments) - 1 and gap.size:
            merged.append(gap)
    return np.concatenate(merged).astype(np.float32, copy=False)


class KokoroChineseTTS:
    def __init__(
        self,
        model_path: str,
        ref_audio_path: str,
        ref_text_path: str,
        *,
        model_profile: TTSModelProfile,
        clone_voice_enabled: bool = True,
        voice: str | None = None,
        device_mode: str = "auto",
        precision_mode: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path
        self.clone_voice_enabled = False
        self.model_profile = model_profile
        self.voice = voice or self.model_profile.default_voice
        self.loaded = False
        self._pipeline = None
        self._model = None
        self._english_pipeline = None
        self.device = _safe_provider_device(device_mode)
        self.device_backend = backend_label_for_device(self.device)
        self.precision_mode = precision_mode or "float32"
        self.semantic_dispatch_mode = "single"
        self.available = Path(model_path).exists()

    def set_reference_paths(self, ref_audio_path: str, ref_text_path: str) -> None:
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path

    def set_voice(self, voice: str) -> None:
        self.voice = voice

    def preload(self) -> None:
        self._ensure_model()

    def unload(self) -> None:
        model = self._model
        self._pipeline = None
        self._english_pipeline = None
        self._model = None
        self.loaded = False
        if model is not None and hasattr(model, "to"):
            try:
                model.to("cpu")
            except Exception:
                pass
        cleanup_torch_memory()

    def synthesize(self, text: str, output_path: str, *, request_overrides: dict | None = None) -> str:
        del request_overrides
        self._ensure_model()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        segments: list[np.ndarray] = []
        for result in self._pipeline(self._clean_text(text), voice=self.voice, split_pattern=r"\n+"):
            audio = result.audio
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().float().numpy()
            segments.append(np.asarray(audio, dtype=np.float32))
        merged = _concat_segments(segments, sample_rate=self.model_profile.sample_rate)
        if merged.size == 0:
            raise RuntimeError("Kokoro did not return synthesized audio")
        sf.write(output, np.clip(merged, -1.0, 1.0), self.model_profile.sample_rate)
        return str(output)

    def format_emotion_text(self, text: str, emotion: str) -> str:
        del emotion
        return text

    @property
    def emotion_tags(self) -> tuple[str, ...]:
        return ()

    def _ensure_model(self) -> None:
        if self._pipeline is not None:
            return
        if not self.available:
            raise FileNotFoundError("Kokoro model files are missing")
        try:
            from kokoro import KModel, KPipeline
        except ImportError as exc:
            raise RuntimeError("kokoro and misaki[zh] are required for Kokoro TTS. Run `uv sync`.") from exc

        runtime_device = _provider_device(self.device)

        config_path = Path(self.model_path) / "config.json"
        checkpoint_path = Path(self.model_path) / "kokoro-v1_1-zh.pth"
        self._english_pipeline = KPipeline(lang_code="a", repo_id=self.model_profile.repo_id, model=False)

        def en_callable(text: str) -> str:
            if not text.strip():
                return ""
            try:
                return next(self._english_pipeline(text)).phonemes
            except StopIteration:
                return text

        self._model = KModel(
            repo_id=self.model_profile.repo_id,
            config=str(config_path),
            model=str(checkpoint_path),
        ).to(runtime_device).eval()
        self._pipeline = KPipeline(
            lang_code=self.model_profile.language_code,
            repo_id=self.model_profile.repo_id,
            model=self._model,
            en_callable=en_callable,
            device=runtime_device,
        )
        self.loaded = True
        self.precision_mode = "float32"

    def _clean_text(self, text: str) -> str:
        return _KOKORO_SPLIT_RE.sub(" ", text).strip()


class MeloChineseTTS:
    def __init__(
        self,
        model_path: str,
        ref_audio_path: str,
        ref_text_path: str,
        *,
        model_profile: TTSModelProfile,
        clone_voice_enabled: bool = True,
        device_mode: str = "auto",
        precision_mode: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path
        self.clone_voice_enabled = False
        self.model_profile = model_profile
        self.loaded = False
        self._model = None
        self._hps = None
        self._symbol_to_id = None
        self.device = _safe_provider_device(device_mode)
        self.device_backend = backend_label_for_device(self.device)
        self.precision_mode = precision_mode or "float32"
        self.semantic_dispatch_mode = "single"
        self.available = Path(model_path).exists()

    def set_reference_paths(self, ref_audio_path: str, ref_text_path: str) -> None:
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path

    def preload(self) -> None:
        self._ensure_model()

    def unload(self) -> None:
        model = self._model
        self._model = None
        self._hps = None
        self._symbol_to_id = None
        self.loaded = False
        if model is not None and hasattr(model, "to"):
            try:
                model.to("cpu")
            except Exception:
                pass
        cleanup_torch_memory()

    def synthesize(self, text: str, output_path: str, *, request_overrides: dict | None = None) -> str:
        self._ensure_model()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        options = {
            "sdp_ratio": 0.2,
            "noise_scale": 0.6,
            "noise_scale_w": 0.8,
            "speed": 1.0,
        }
        if request_overrides:
            options.update({key: value for key, value in request_overrides.items() if key in options})
        speaker_id = int(self._hps.data.spk2id[self.model_profile.default_voice])
        audio_segments: list[np.ndarray] = []
        from backend.tts.melo_vendor import utils as melo_utils
        from backend.tts.melo_vendor.split_utils import split_sentence as melo_split_sentence

        prepared_text = self._normalize_text(text)
        for segment in melo_split_sentence(prepared_text, language_str=self.model_profile.language_code):
            if not segment.strip():
                continue
            try:
                bert, ja_bert, phones, tones, lang_ids = melo_utils.get_text_for_tts_infer(
                    segment,
                    self.model_profile.language_code,
                    self._hps,
                    self.device,
                    self._symbol_to_id,
                )
            except AssertionError:
                fallback_segment = re.sub(r"\s+", "", segment)
                bert, ja_bert, phones, tones, lang_ids = melo_utils.get_text_for_tts_infer(
                    fallback_segment,
                    self.model_profile.language_code,
                    self._hps,
                    self.device,
                    self._symbol_to_id,
                )
            import torch

            with torch.no_grad():
                x_tst = phones.to(self.device).unsqueeze(0)
                tones = tones.to(self.device).unsqueeze(0)
                lang_ids = lang_ids.to(self.device).unsqueeze(0)
                bert = bert.to(self.device).unsqueeze(0)
                ja_bert = ja_bert.to(self.device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
                speakers = torch.LongTensor([speaker_id]).to(self.device)
                audio = self._model.infer(
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    sdp_ratio=float(options["sdp_ratio"]),
                    noise_scale=float(options["noise_scale"]),
                    noise_scale_w=float(options["noise_scale_w"]),
                    length_scale=1.0 / max(float(options["speed"]), 0.05),
                )[0][0, 0].data.cpu().float().numpy()
            audio_segments.append(audio)
        merged = _concat_segments(audio_segments, sample_rate=int(self._hps.data.sampling_rate))
        if merged.size == 0:
            raise RuntimeError("MeloTTS did not return synthesized audio")
        sf.write(output, np.clip(merged, -1.0, 1.0), int(self._hps.data.sampling_rate))
        return str(output)

    def format_emotion_text(self, text: str, emotion: str) -> str:
        del emotion
        return text

    @property
    def emotion_tags(self) -> tuple[str, ...]:
        return ()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise FileNotFoundError("MeloTTS model files are missing")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for MeloTTS runtime. Run `uv sync`.") from exc
        from backend.tts.melo_vendor import utils as melo_utils
        from backend.tts.melo_vendor.models import SynthesizerTrn

        config_path = Path(self.model_path) / "config.json"
        checkpoint_path = Path(self.model_path) / "checkpoint.pth"
        self._hps = melo_utils.get_hparams_from_file(str(config_path))
        self._hps.data.disable_bert = True
        symbols = self._hps.symbols
        self._model = SynthesizerTrn(
            len(symbols),
            self._hps.data.filter_length // 2 + 1,
            self._hps.train.segment_size // self._hps.data.hop_length,
            n_speakers=self._hps.data.n_speakers,
            num_tones=self._hps.num_tones,
            num_languages=self._hps.num_languages,
            **self._hps.model,
        ).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._model.load_state_dict(checkpoint["model"], strict=True)
        self._model.eval()
        self._symbol_to_id = {symbol: index for index, symbol in enumerate(symbols)}
        self.loaded = True
        self.precision_mode = "float32"

    def _normalize_text(self, text: str) -> str:
        normalized = text.strip()
        try:
            from opencc import OpenCC
        except ImportError:
            return normalized
        return OpenCC("t2s").convert(normalized)

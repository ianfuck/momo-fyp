from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


S1_EMOTION_TAGS = (
    "happy",
    "sad",
    "angry",
    "excited",
    "calm",
    "nervous",
    "confident",
    "surprised",
    "satisfied",
    "delighted",
    "scared",
    "worried",
    "upset",
    "frustrated",
    "depressed",
    "empathetic",
    "embarrassed",
    "disgusted",
    "moved",
    "proud",
    "relaxed",
    "grateful",
    "curious",
    "sarcastic",
    "disdainful",
    "unhappy",
    "anxious",
    "hysterical",
    "indifferent",
    "uncertain",
    "doubtful",
    "confused",
    "disappointed",
    "regretful",
    "guilty",
    "ashamed",
    "jealous",
    "envious",
    "hopeful",
    "optimistic",
    "pessimistic",
    "nostalgic",
    "lonely",
    "bored",
    "contemptuous",
    "sympathetic",
    "compassionate",
    "determined",
    "resigned",
)

V1_5_EMOTION_TAGS = (
    "happy",
    "sad",
    "angry",
    "excited",
    "calm",
    "nervous",
    "confident",
    "surprised",
    "satisfied",
    "delighted",
    "scared",
    "worried",
    "upset",
    "frustrated",
    "depressed",
    "empathetic",
    "embarrassed",
    "disgusted",
    "moved",
    "proud",
    "relaxed",
    "grateful",
    "curious",
    "sarcastic",
)


@dataclass(frozen=True)
class TTSModelProfile:
    key: str
    display_name: str
    repo_id: str
    local_dir_name: str
    required_model_files: tuple[str, ...]
    decoder_config_name: str
    decoder_checkpoint_name: str
    emotion_tags: tuple[str, ...]
    emotion_prompt_label: str
    runtime_family: str = "fish"
    supports_structured_emotion: bool = True
    supports_voice_clone: bool = True
    supports_startup_benchmark: bool = True
    default_voice: str = ""
    language_code: str = ""
    sample_rate: int = 24000

    def format_emotion_text(self, text: str, emotion: str) -> str:
        return f"({emotion}){text.lstrip()}"

    @property
    def huggingface_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"


FISH_SPEECH_V1_5_PROFILE = TTSModelProfile(
    key="fish-speech-1.5",
    display_name="Fish Speech V1.5",
    repo_id="fishaudio/fish-speech-1.5",
    local_dir_name="fishaudio__fish-speech-1.5",
    required_model_files=(
        "config.json",
        "model.pth",
        "special_tokens.json",
        "tokenizer.tiktoken",
        "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    ),
    decoder_config_name="backend/tts/decoder_configs/firefly_gan_vq.yaml",
    decoder_checkpoint_name="firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    emotion_tags=V1_5_EMOTION_TAGS,
    emotion_prompt_label="Fish Speech V1.5 basic emotion tags",
)

FISH_AUDIO_S1_MINI_PROFILE = TTSModelProfile(
    key="s1-mini",
    display_name="Fish Audio S1 Mini",
    repo_id="fishaudio/s1-mini",
    local_dir_name="fishaudio__s1-mini",
    required_model_files=(
        "config.json",
        "model.pth",
        "codec.pth",
        "special_tokens.json",
        "tokenizer.tiktoken",
    ),
    decoder_config_name="modded_dac_vq",
    decoder_checkpoint_name="codec.pth",
    emotion_tags=S1_EMOTION_TAGS,
    emotion_prompt_label="Fish Audio S1 emotion tags",
)

QWEN3_TTS_0_6B_BASE_PROFILE = TTSModelProfile(
    key="qwen3-tts-12hz-0.6b-base",
    display_name="Qwen3-TTS 0.6B Base",
    repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    local_dir_name="Qwen__Qwen3-TTS-12Hz-0.6B-Base",
    required_model_files=(
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/configuration.json",
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/preprocessor_config.json",
    ),
    decoder_config_name="",
    decoder_checkpoint_name="",
    emotion_tags=(),
    emotion_prompt_label="",
    runtime_family="qwen",
    supports_structured_emotion=False,
    supports_voice_clone=True,
    supports_startup_benchmark=True,
)

QWEN3_TTS_1_7B_BASE_PROFILE = TTSModelProfile(
    key="qwen3-tts-12hz-1.7b-base",
    display_name="Qwen3-TTS 1.7B Base",
    repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    local_dir_name="Qwen__Qwen3-TTS-12Hz-1.7B-Base",
    required_model_files=(
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/configuration.json",
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/preprocessor_config.json",
    ),
    decoder_config_name="",
    decoder_checkpoint_name="",
    emotion_tags=(),
    emotion_prompt_label="",
    runtime_family="qwen",
    supports_structured_emotion=False,
    supports_voice_clone=True,
    supports_startup_benchmark=True,
)

KOKORO_82M_ZH_PROFILE = TTSModelProfile(
    key="kokoro-82m-zh",
    display_name="Kokoro-82M",
    repo_id="hexgrad/Kokoro-82M-v1.1-zh",
    local_dir_name="hexgrad__Kokoro-82M-v1.1-zh",
    required_model_files=(
        "config.json",
        "kokoro-v1_1-zh.pth",
        "voices/zf_001.pt",
    ),
    decoder_config_name="",
    decoder_checkpoint_name="",
    emotion_tags=(),
    emotion_prompt_label="",
    runtime_family="kokoro",
    supports_structured_emotion=False,
    supports_voice_clone=False,
    supports_startup_benchmark=True,
    default_voice="zf_001",
    language_code="z",
    sample_rate=24000,
)

MELOTTS_CHINESE_PROFILE = TTSModelProfile(
    key="melotts-chinese",
    display_name="MeloTTS",
    repo_id="myshell-ai/MeloTTS-Chinese",
    local_dir_name="myshell-ai__MeloTTS-Chinese",
    required_model_files=(
        "config.json",
        "checkpoint.pth",
    ),
    decoder_config_name="",
    decoder_checkpoint_name="",
    emotion_tags=(),
    emotion_prompt_label="",
    runtime_family="melo",
    supports_structured_emotion=False,
    supports_voice_clone=False,
    supports_startup_benchmark=True,
    default_voice="ZH",
    language_code="ZH",
    sample_rate=44100,
)

DEFAULT_TTS_MODEL_PROFILE = FISH_SPEECH_V1_5_PROFILE
_ALL_TTS_MODEL_PROFILES = (
    QWEN3_TTS_0_6B_BASE_PROFILE,
    QWEN3_TTS_1_7B_BASE_PROFILE,
    FISH_SPEECH_V1_5_PROFILE,
    FISH_AUDIO_S1_MINI_PROFILE,
    KOKORO_82M_ZH_PROFILE,
    MELOTTS_CHINESE_PROFILE,
)


def supported_tts_model_paths() -> list[str]:
    return [f"model/huggingface/hf_snapshots/{profile.local_dir_name}" for profile in _ALL_TTS_MODEL_PROFILES]


def resolve_tts_model_profile(model_path: str) -> TTSModelProfile:
    path = Path(model_path)
    lowered = str(path).lower()
    file_names = {item.name.lower() for item in path.iterdir()} if path.exists() and path.is_dir() else set()
    if "qwen3-tts" in lowered or QWEN3_TTS_0_6B_BASE_PROFILE.local_dir_name.lower() in lowered:
        if QWEN3_TTS_1_7B_BASE_PROFILE.local_dir_name.lower() in lowered or "1.7b" in lowered:
            return QWEN3_TTS_1_7B_BASE_PROFILE
        return QWEN3_TTS_0_6B_BASE_PROFILE
    if "model.safetensors" in file_names and (path / "speech_tokenizer").exists():
        if "1.7b" in lowered:
            return QWEN3_TTS_1_7B_BASE_PROFILE
        return QWEN3_TTS_0_6B_BASE_PROFILE
    if "kokoro-v1_1-zh.pth" in file_names or "kokoro-82m" in lowered:
        return KOKORO_82M_ZH_PROFILE
    if "checkpoint.pth" in file_names or "melotts" in lowered:
        return MELOTTS_CHINESE_PROFILE
    if "firefly-gan-vq-fsq-8x1024-21hz-generator.pth" in file_names or "fish-speech-1.5" in lowered:
        return FISH_SPEECH_V1_5_PROFILE
    if "codec.pth" in file_names or "s1-mini" in lowered:
        return FISH_AUDIO_S1_MINI_PROFILE
    for profile in _ALL_TTS_MODEL_PROFILES:
        if profile.local_dir_name.lower() in lowered:
            return profile
    return DEFAULT_TTS_MODEL_PROFILE

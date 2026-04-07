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

DEFAULT_TTS_MODEL_PROFILE = FISH_SPEECH_V1_5_PROFILE
_ALL_TTS_MODEL_PROFILES = (
    FISH_SPEECH_V1_5_PROFILE,
    FISH_AUDIO_S1_MINI_PROFILE,
)


def resolve_tts_model_profile(model_path: str) -> TTSModelProfile:
    path = Path(model_path)
    lowered = str(path).lower()
    file_names = {item.name.lower() for item in path.iterdir()} if path.exists() and path.is_dir() else set()
    if "firefly-gan-vq-fsq-8x1024-21hz-generator.pth" in file_names or "fish-speech-1.5" in lowered:
        return FISH_SPEECH_V1_5_PROFILE
    if "codec.pth" in file_names or "s1-mini" in lowered:
        return FISH_AUDIO_S1_MINI_PROFILE
    for profile in _ALL_TTS_MODEL_PROFILES:
        if profile.local_dir_name.lower() in lowered:
            return profile
    return DEFAULT_TTS_MODEL_PROFILE

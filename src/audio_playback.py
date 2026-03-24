"""Cross-platform WAV playback (auto-detect backend)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path


_lock = threading.Lock()
_resolved: str | None = None


def resolve_backend(preference: str) -> str:
    global _resolved
    with _lock:
        if preference != "auto":
            _resolved = preference
            return _resolved
        if _resolved:
            return _resolved
        if sys.platform == "darwin" and shutil.which("afplay"):
            _resolved = "afplay"
        elif sys.platform == "win32":
            _resolved = "powershell"
        elif shutil.which("aplay"):
            _resolved = "aplay"
        else:
            try:
                import pygame  # noqa: F401

                _resolved = "pygame"
            except ImportError:
                _resolved = "none"
        return _resolved


def play_wav_blocking(path: Path, preference: str = "auto") -> None:
    path = path.resolve()
    if not path.exists():
        return
    backend = resolve_backend(preference)
    if backend == "afplay":
        subprocess.run(["afplay", str(path)], check=False, capture_output=True)
    elif backend == "aplay":
        subprocess.run(["aplay", str(path)], check=False, capture_output=True)
    elif backend == "powershell":
        ps = (
            f'(New-Object Media.SoundPlayer "{path.as_posix().replace(chr(92), chr(92)+chr(92))}").PlaySync()'
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            check=False,
            capture_output=True,
        )
    elif backend == "pygame":
        import pygame

        pygame.mixer.init()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.delay(50)
    else:
        pass


def play_wav_async(path: Path, preference: str, on_end: callable) -> None:
    def _run() -> None:
        try:
            play_wav_blocking(path, preference)
        finally:
            on_end()

    threading.Thread(target=_run, daemon=True).start()

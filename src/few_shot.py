"""Load CSV examples into chat messages for Ollama."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.paths import PROJECT_ROOT


def csv_to_few_shot_messages(
    relative_paths: list[str],
    max_rows_per_file: int,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for rel in relative_paths:
        path = (PROJECT_ROOT / rel).resolve()
        if not path.exists():
            continue
        rows: list[str] = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0 and row and ("順序" in row[0] or "編號" in row[0] or "階段" in row[0]):
                    continue
                if not row:
                    continue
                rows.append(" | ".join(cell.strip() for cell in row if cell.strip()))
                if len(rows) >= max_rows_per_file:
                    break
        if not rows:
            continue
        block = f"[範例檔 {path.name}]\n" + "\n".join(f"- {r}" for r in rows)
        messages.append({"role": "user", "content": f"以下是風格參考範例（不要複誦，僅模仿語氣與節奏）：\n{block}"})
        messages.append(
            {
                "role": "assistant",
                "content": "了解。我會依上述範例的語氣與節奏，只輸出一句符合規則的台詞。",
            }
        )
    return messages


_cache_key: tuple[Any, ...] | None = None
_cached_messages: list[dict[str, str]] | None = None


def get_cached_few_shot(
    paths: list[str],
    max_rows: int,
    mtimes: list[float],
) -> list[dict[str, str]]:
    global _cache_key, _cached_messages
    key = (tuple(paths), max_rows, tuple(mtimes))
    if key == _cache_key and _cached_messages is not None:
        return _cached_messages
    _cache_key = key
    _cached_messages = csv_to_few_shot_messages(paths, max_rows)
    return _cached_messages


def few_shot_mtimes(paths: list[str]) -> list[float]:
    out: list[float] = []
    for rel in paths:
        p = PROJECT_ROOT / rel
        if p.exists():
            out.append(p.stat().st_mtime)
        else:
            out.append(0.0)
    return out

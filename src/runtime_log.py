"""tmp/log.csv（事件）。"""

from __future__ import annotations

import csv
import json
import threading
import time
from typing import Any

from src.paths import PROJECT_ROOT

_lock = threading.Lock()
LOG_PATH = PROJECT_ROOT / "tmp" / "log.csv"


def append_log(category: str, message: str, detail: dict[str, Any] | None = None) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = [
        time.strftime("%Y-%m-%dT%H:%M:%S"),
        category,
        message,
        json.dumps(detail or {}, ensure_ascii=False),
    ]
    with _lock:
        new_file = not LOG_PATH.exists()
        with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp_iso", "category", "message", "detail_json"])
            w.writerow(row)

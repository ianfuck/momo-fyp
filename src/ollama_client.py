"""Ollama HTTP chat client：串流為主、總時限、自動重試、可選預熱。"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from src.few_shot import few_shot_mtimes, get_cached_few_shot
from src.paths import PROJECT_ROOT

_pull_lock = threading.Lock()
_warm_lock = threading.Lock()
_warmed: set[tuple[str, str]] = set()


def read_text(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8")


def _ollama_tags(client: httpx.Client, base: str) -> set[str]:
    r = client.get(f"{base}/api/tags")
    r.raise_for_status()
    data = r.json()
    return {m.get("name", "") for m in data.get("models", []) if m.get("name")}


def ensure_ollama_model(base_url: str, model: str, pull_timeout_s: float) -> None:
    """若本機 Ollama 尚未有該模型，呼叫 POST /api/pull 自動下載。"""
    base = base_url.rstrip("/")
    name = (model or "").strip()
    if not name:
        raise ValueError("ollama_model 不可為空")
    with httpx.Client(timeout=30.0) as client:
        if name in _ollama_tags(client, base):
            return
    with _pull_lock:
        with httpx.Client(timeout=30.0) as client:
            if name in _ollama_tags(client, base):
                return
        timeout = httpx.Timeout(pull_timeout_s, connect=60.0)
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{base}/api/pull", json={"name": name}) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    err = obj.get("error")
                    if err:
                        raise RuntimeError(f"Ollama pull 失敗：{err}")
        with httpx.Client(timeout=30.0) as client:
            if name not in _ollama_tags(client, base):
                raise RuntimeError(
                    f"Ollama pull 結束後仍找不到模型「{name}」，請確認名稱正確且 registry 可存取。"
                )


def _maybe_warm_ollama(base: str, model: str, keep_alive: str) -> None:
    """極短 generate，促進 Ollama 載入權重；逾時即放棄，不阻塞主請求太久。"""
    key = (base, model)
    with _warm_lock:
        if key in _warmed:
            return
    try:
        with httpx.Client(timeout=httpx.Timeout(25.0, connect=10.0, pool=10.0)) as client:
            client.post(
                f"{base}/api/generate",
                json={
                    "model": model,
                    "prompt": ".",
                    "stream": False,
                    "keep_alive": keep_alive,
                    "options": {"num_predict": 1},
                },
            )
    except Exception:
        return
    with _warm_lock:
        _warmed.add(key)


def preload_ollama_model(
    *,
    base_url: str,
    model: str,
    pull_timeout_s: float,
    keep_alive: str,
    generate_timeout_s: float = 120.0,
) -> None:
    """啟動時預載：ensure pull + 一次短 generate，並設定 keep_alive。"""
    base = base_url.rstrip("/")
    name = (model or "").strip()
    if not name:
        raise ValueError("ollama_model 不可為空")
    ensure_ollama_model(base_url, name, pull_timeout_s)
    body = {
        "model": name,
        "prompt": ".",
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"num_predict": 1},
    }
    with httpx.Client(timeout=httpx.Timeout(generate_timeout_s, connect=30.0, pool=30.0)) as client:
        r = client.post(f"{base}/api/generate", json=body)
        r.raise_for_status()
    with _warm_lock:
        _warmed.add((base, name))


def _parse_chat_json(data: dict[str, Any]) -> str:
    msg = data.get("message", {}) or {}
    text = (msg.get("content") or "").strip()
    if not text and isinstance(data.get("response"), str):
        text = (data.get("response") or "").strip()
    return text


def _chat_stream(
    base: str,
    model: str,
    messages: list[dict[str, str]],
    budget_s: float,
    keep_alive: str,
) -> tuple[str, int]:
    """
    串流 /api/chat：iter_bytes 手動切行；read=None 避免久未出換行被單次 read 判逾時。
    總時間僅由 ollama_timeout_s（deadline）限制。
    """
    url = f"{base}/api/chat"
    body = {"model": model, "messages": messages, "stream": True, "keep_alive": keep_alive}
    deadline = time.monotonic() + max(budget_s, 5.0)
    httpx_timeout = httpx.Timeout(connect=60.0, read=None, write=120.0, pool=60.0)
    t0 = time.perf_counter()
    parts: list[str] = []
    buf = b""
    done = False
    with httpx.Client(timeout=httpx_timeout) as client:
        with client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(chunk_size=4096):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Ollama 串流超過總時限 {budget_s:.0f}s（可調 ollama_timeout_s）"
                    )
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    err = obj.get("error")
                    if err:
                        raise RuntimeError(str(err))
                    m = obj.get("message") or {}
                    c = m.get("content") or ""
                    if c:
                        parts.append(c)
                    if obj.get("done"):
                        done = True
                        break
                if done:
                    break
    text = "".join(parts).strip()
    if text:
        text = text.splitlines()[0]
    ms = int((time.perf_counter() - t0) * 1000)
    return text, ms


def _chat_blocking(
    base: str,
    model: str,
    messages: list[dict[str, str]],
    read_timeout: float,
    keep_alive: str,
) -> tuple[str, int]:
    """非串流後備（同樣設 connect/pool）。"""
    url = f"{base}/api/chat"
    body = {"model": model, "messages": messages, "stream": False, "keep_alive": keep_alive}
    t0 = time.perf_counter()
    httpx_timeout = httpx.Timeout(read_timeout, connect=45.0, pool=45.0)
    with httpx.Client(timeout=httpx_timeout) as client:
        r = client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
    ms = int((time.perf_counter() - t0) * 1000)
    text = _parse_chat_json(data)
    if text:
        text = text.splitlines()[0]
    return text, ms


def generate_line_sync(
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    pull_timeout_s: float,
    max_retries: int,
    warmup: bool,
    keep_alive: str,
    system_persona_rel: str,
    few_shot_paths: list[str],
    few_shot_max_rows: int,
    user_payload: dict[str, Any],
    max_chars: int,
) -> tuple[str, int]:
    base = base_url.rstrip("/")
    ensure_ollama_model(base_url, model, pull_timeout_s)
    if warmup:
        _maybe_warm_ollama(base, model, keep_alive)

    system = read_text(system_persona_rel)
    mt = few_shot_mtimes(few_shot_paths)
    few = get_cached_few_shot(few_shot_paths, few_shot_max_rows, mt)
    user_task = (
        "請只輸出一句繁體中文台詞，不要引號、不要解釋。"
        f"字數上限 {max_chars} 字（含標點）。\n"
        f"當前情境（JSON）：{json.dumps(user_payload, ensure_ascii=False)}"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(few)
    messages.append({"role": "user", "content": user_task})

    budget = max(float(timeout_s), 30.0)
    attempts = max(1, int(max_retries))
    last_err: Exception | None = None

    for attempt in range(attempts):
        try:
            text, ms = _chat_stream(base, model, messages, budget, keep_alive)
            if text:
                return text[: max_chars + 20], ms
            last_err = RuntimeError("Ollama 回傳空字串")
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError, TimeoutError) as e:
            last_err = e
        except Exception as e:
            last_err = e

        if attempt + 1 < attempts:
            delay = min(8.0, 0.6 * (2**attempt))
            time.sleep(delay)
            continue

        # 最後一次再試非串流（少數環境串流異常）
        try:
            text, ms = _chat_blocking(base, model, messages, read_timeout=budget, keep_alive=keep_alive)
            if text:
                return text[: max_chars + 20], ms
        except httpx.ReadTimeout as e:
            last_err = e
        except Exception as e:
            last_err = e

    msg = f"Ollama 在 {attempts} 次嘗試後仍失敗：{last_err}"
    raise TimeoutError(msg) from last_err


from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator

import httpx


class OllamaClient:
    def __init__(self, base_url: str, timeout_sec: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def _ollama_options(self, options: dict | None = None) -> dict:
        merged = dict(options or {})
        merged.setdefault("num_gpu", 999)
        return merged

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
            return [item["name"] for item in payload.get("models", [])]

    async def running_models(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            response = await client.get(f"{self.base_url}/api/ps")
            response.raise_for_status()
            payload = response.json()
            return list(payload.get("models", []))

    async def warmup_model(self, model: str) -> dict:
        payload = {
            "model": model,
            "prompt": "ping",
            "stream": False,
            "options": self._ollama_options({"num_predict": 1, "temperature": 0}),
        }
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()

    async def generate_stream(
        self,
        model: str,
        system: str,
        prompt: str,
        options: dict | None = None,
        think: bool | None = False,
        images: list[bytes] | None = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": True,
            "options": self._ollama_options(options),
            "think": think,
        }
        if images:
            payload["images"] = [base64.b64encode(image).decode("ascii") for image in images]
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break

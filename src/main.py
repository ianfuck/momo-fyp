"""FastAPI app: REST, WebSocket, MJPEG, static Vite build."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import RuntimeConfig, build_config_schema, list_example_csvs
from src.paths import PROJECT_ROOT
from src.runtime import Orchestrator

orch = Orchestrator()

app = FastAPI(title="Momo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    orch.start()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/cameras")
def cameras() -> list[dict[str, Any]]:
    import cv2

    out: list[dict[str, Any]] = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            out.append({"index": i, "name": f"Camera {i}"})
            cap.release()
    if not out:
        out.append({"index": 0, "name": "Camera 0 (default)"})
    return out


@app.get("/api/resource/examples")
def api_examples() -> list[dict[str, Any]]:
    return list_example_csvs()


@app.get("/api/config/schema")
def api_schema() -> dict[str, Any]:
    return build_config_schema()


@app.get("/api/config")
def api_config_get() -> dict[str, Any]:
    return orch.get_config()


@app.patch("/api/config")
def api_config_patch(body: dict[str, Any]) -> dict[str, Any]:
    merged = {**orch.get_config(), **body}
    try:
        RuntimeConfig(**merged)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    orch.patch_config(body)
    return {"ok": True, "config": orch.get_config()}


@app.get("/api/stream/mjpeg")
def mjpeg() -> StreamingResponse:
    boundary = b"frame"

    def gen():
        while True:
            frame = orch.mjpeg_frame()
            if not frame:
                time.sleep(0.05)
                continue
            yield b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary.decode()}",
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            if orch.consume_round_reset():
                await ws.send_json({"type": "round_reset", "payload": {}})
            await ws.send_json({"type": "state", "payload": _public_dict()})
            await asyncio.sleep(0.12)
    except WebSocketDisconnect:
        pass


def _public_dict() -> dict[str, Any]:
    p = orch.public
    return {
        "state": p.state,
        "bbox_value": p.bbox_value,
        "bbox": p.bbox,
        "audience_features": p.audience_features,
        "sentence_index": p.sentence_index,
        "speech_phase": p.speech_phase,
        "speech_mode": p.speech_mode,
        "speech_sentence": p.speech_sentence,
        "llm_generating": p.llm_generating,
        "tts_generating": p.tts_generating,
        "last_llm_ms": p.last_llm_ms,
        "last_tts_ms": p.last_tts_ms,
        "angle_left_deg": p.angle_left_deg,
        "angle_right_deg": p.angle_right_deg,
        "mode_servo": p.mode_servo,
        "audio_backend_active": p.audio_backend_active,
        "tts_backend": p.tts_backend,
        "behavior_tags": p.behavior_tags,
        "vision_device": p.vision_device,
        "gpu_metrics": p.gpu_metrics,
    }


dist = PROJECT_ROOT / "frontend" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="static")


def run() -> None:
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()

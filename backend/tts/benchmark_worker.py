from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.telemetry.system_stats import (
    capture_process_footprint,
    diff_process_footprint,
    peak_device_memory_mb,
    reset_peak_device_memory,
)
from backend.tts.qwen_clone import FishCloneTTS, _BENCHMARK_REQUEST_OVERRIDES
from backend.tts.semantic_runtime import SemanticBenchmarkResult


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one isolated TTS benchmark candidate.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ref-audio-path", required=True)
    parser.add_argument("--ref-text-path", required=True)
    parser.add_argument("--device-mode", required=True)
    parser.add_argument("--plan-name", required=True)
    parser.add_argument("--semantic-dispatch-mode", required=True)
    parser.add_argument("--precision-mode", required=True)
    parser.add_argument("--sample-text", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--clone-voice-enabled", action="store_true")
    args = parser.parse_args(argv)

    result = run_candidate(
        model_path=args.model_path,
        ref_audio_path=args.ref_audio_path,
        ref_text_path=args.ref_text_path,
        device_mode=args.device_mode,
        plan_name=args.plan_name,
        semantic_dispatch_mode=args.semantic_dispatch_mode,
        precision_mode=args.precision_mode,
        clone_voice_enabled=args.clone_voice_enabled,
        sample_text=args.sample_text,
    )
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result.__dict__, ensure_ascii=False), encoding="utf-8")


def run_candidate(
    *,
    model_path: str,
    ref_audio_path: str,
    ref_text_path: str,
    device_mode: str,
    plan_name: str,
    semantic_dispatch_mode: str,
    precision_mode: str,
    clone_voice_enabled: bool,
    sample_text: str,
) -> SemanticBenchmarkResult:
    candidate = FishCloneTTS(
        model_path,
        ref_audio_path,
        ref_text_path,
        clone_voice_enabled=clone_voice_enabled,
        device_mode=device_mode,
        semantic_dispatch_mode=semantic_dispatch_mode,
        precision_mode=precision_mode,
    )
    started = time.monotonic()
    before = capture_process_footprint(candidate.device)
    try:
        reset_peak_device_memory(candidate.device)
        preload_started = time.monotonic()
        candidate.preload()
        _synchronize_device(candidate.device)
        preload_ms = int((time.monotonic() - preload_started) * 1000)
        after_preload = capture_process_footprint(candidate.device)
        synth_started = time.monotonic()
        candidate.synthesize(
            sample_text,
            f"tmp/tts_benchmark_{device_mode}_{semantic_dispatch_mode}.wav",
            request_overrides=_BENCHMARK_REQUEST_OVERRIDES,
        )
        _synchronize_device(candidate.device)
        synth_ms = int((time.monotonic() - synth_started) * 1000)
        after = capture_process_footprint(candidate.device)
        ram_mb, vram_mb = diff_process_footprint(before, after)
        peak_vram_mb = peak_device_memory_mb(candidate.device)
        if peak_vram_mb is None:
            snapshots = [after_preload.vram_mb, after.vram_mb]
            peak_vram_mb = max((value for value in snapshots if value is not None), default=None)
        return SemanticBenchmarkResult(
            name=plan_name,
            device_mode=device_mode,
            semantic_dispatch_mode=candidate.semantic_dispatch_mode,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            ok=True,
            preload_ms=preload_ms,
            synth_ms=synth_ms,
            precision_mode=candidate.precision_mode,
            peak_vram_mb=peak_vram_mb,
            ram_mb=ram_mb,
            vram_mb=vram_mb,
        )
    except Exception as exc:
        return SemanticBenchmarkResult(
            name=plan_name,
            device_mode=device_mode,
            semantic_dispatch_mode=semantic_dispatch_mode,
            elapsed_ms=-1,
            ok=False,
            preload_ms=None,
            synth_ms=None,
            precision_mode=precision_mode,
            peak_vram_mb=peak_device_memory_mb(candidate.device),
            detail=str(exc),
        )


def _synchronize_device(device: str | None) -> None:
    try:
        import torch
    except ImportError:
        return

    if device is None:
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        try:
            torch.cuda.synchronize(index)
        except Exception:
            return
        return
    if device == "mps":
        synchronize = getattr(getattr(torch, "mps", None), "synchronize", None)
        if callable(synchronize):
            try:
                synchronize()
            except Exception:
                return


if __name__ == "__main__":
    main()

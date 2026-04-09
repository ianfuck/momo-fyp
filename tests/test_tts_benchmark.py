from types import SimpleNamespace

from backend.tts import qwen_clone
from backend.tts.model_profiles import resolve_tts_model_profile
from backend.tts.qwen_clone import FishCloneTTS
from backend.tts.semantic_runtime import SemanticBenchmarkResult


class DummyBenchmarkTTS(FishCloneTTS):
    def __init__(
        self,
        model_path: str,
        ref_audio_path: str,
        ref_text_path: str,
        clone_voice_enabled: bool = True,
        device_mode: str = "auto",
        semantic_dispatch_mode: str = "single",
        precision_mode: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path
        self.clone_voice_enabled = clone_voice_enabled
        self.device = device_mode
        self.device_backend = device_mode
        self.semantic_dispatch_mode = semantic_dispatch_mode
        self.precision_mode = precision_mode or ("float16" if device_mode == "gpu" else "float32")
        self.model_profile = resolve_tts_model_profile(model_path)
        self.loaded = False


def test_benchmark_auto_profiles_logs_running_candidate(monkeypatch, capsys):
    model_path = "model/huggingface/hf_snapshots/fishaudio__fish-speech-1.5"
    plans = [
        SimpleNamespace(name="gpu", device_mode="gpu", semantic_dispatch_mode="single"),
    ]
    results = {
        "gpu-float16": SemanticBenchmarkResult(
            name="gpu-float16",
            device_mode="gpu",
            semantic_dispatch_mode="single",
            elapsed_ms=1600,
            ok=True,
            preload_ms=200,
            synth_ms=1400,
            precision_mode="float16",
            peak_vram_mb=512.0,
        ),
        "gpu-float32": SemanticBenchmarkResult(
            name="gpu-float32",
            device_mode="gpu",
            semantic_dispatch_mode="single",
            elapsed_ms=900,
            ok=True,
            preload_ms=500,
            synth_ms=400,
            precision_mode="float32",
            peak_vram_mb=640.0,
        ),
    }

    monkeypatch.setattr("backend.tts.qwen_clone.benchmark_plans_for_current_host", lambda: plans)
    monkeypatch.setattr(
        "backend.tts.qwen_clone._run_benchmark_candidate_subprocess",
        lambda *, plan, **kwargs: results[plan.name],
    )

    selection = DummyBenchmarkTTS.benchmark_auto_profiles(
        model_path,
        "resource/voice/ref-voice3.wav",
        "resource/voice/transcript3.txt",
        clone_voice_enabled=True,
    )

    captured = capsys.readouterr()
    assert selection is not None
    assert selection.result.name == "gpu-float32"
    assert selection.result.precision_mode == "float32"
    assert selection.tts.precision_mode == "float32"
    assert "[startup] tts benchmark running candidate=gpu-float16 device=gpu semantic=single precision=float16" in captured.out
    assert (
        "[startup] tts benchmark candidate=gpu-float32 status=ok preload_ms=500 synth_ms=400 total_ms=900 "
        "peak_vram_mb=640.0 semantic=single precision=float32"
    ) in captured.out


def test_qwen_benchmark_auto_profiles_uses_device_only_candidates(monkeypatch):
    model_path = "model/huggingface/hf_snapshots/Qwen__Qwen3-TTS-12Hz-0.6B-Base"
    plans = [
        SimpleNamespace(name="gpu", device_mode="gpu", semantic_dispatch_mode="single"),
        SimpleNamespace(name="semantic-auto-gpu", device_mode="gpu", semantic_dispatch_mode="auto"),
        SimpleNamespace(name="cpu", device_mode="cpu", semantic_dispatch_mode="single"),
    ]
    seen_plan_names: list[str] = []
    results = {
        "gpu-float16": SemanticBenchmarkResult(
            name="gpu-float16",
            device_mode="gpu",
            semantic_dispatch_mode="single",
            elapsed_ms=800,
            ok=True,
            preload_ms=100,
            synth_ms=700,
            precision_mode="float16",
        ),
        "gpu-float32": SemanticBenchmarkResult(
            name="gpu-float32",
            device_mode="gpu",
            semantic_dispatch_mode="single",
            elapsed_ms=1400,
            ok=True,
            preload_ms=1100,
            synth_ms=300,
            precision_mode="float32",
        ),
        "cpu-float32": SemanticBenchmarkResult(
            name="cpu-float32",
            device_mode="cpu",
            semantic_dispatch_mode="single",
            elapsed_ms=900,
            ok=True,
            preload_ms=100,
            synth_ms=800,
            precision_mode="float32",
        ),
    }

    monkeypatch.setattr("backend.tts.qwen_clone.benchmark_plans_for_current_host", lambda: plans)

    def fake_runner(*, plan, **kwargs):
        seen_plan_names.append(plan.name)
        return results[plan.name]

    monkeypatch.setattr("backend.tts.qwen_clone._run_benchmark_candidate_subprocess", fake_runner)

    selection = DummyBenchmarkTTS.benchmark_auto_profiles(
        model_path,
        "resource/voice/ref-voice3.wav",
        "resource/voice/transcript3.txt",
        clone_voice_enabled=True,
    )

    assert selection is not None
    assert selection.result.name == "gpu-float32"
    assert selection.result.precision_mode == "float32"
    assert selection.tts.precision_mode == "float32"
    assert seen_plan_names == ["gpu-float16", "gpu-float32", "cpu-float32"]


def test_cleanup_benchmark_temp_dir_retries_windows_cleanup(monkeypatch, tmp_path, capsys):
    bench_dir = tmp_path / "momo_tts_bench_case"
    bench_dir.mkdir()
    (bench_dir / "stderr.log").write_text("busy", encoding="utf-8")

    original_rmtree = qwen_clone.shutil.rmtree
    attempts: list[object] = []

    def flaky_rmtree(path):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError(32, "The process cannot access the file because it is being used by another process")
        original_rmtree(path)

    monkeypatch.setattr("backend.tts.qwen_clone.shutil.rmtree", flaky_rmtree)
    monkeypatch.setattr("backend.tts.qwen_clone.time.sleep", lambda _: None)
    monkeypatch.setattr("backend.tts.qwen_clone.os.name", "nt")

    qwen_clone._cleanup_benchmark_temp_dir(bench_dir)

    captured = capsys.readouterr()
    assert len(attempts) == 2
    assert not bench_dir.exists()
    assert "tts benchmark cleanup skipped" not in captured.out


def test_cleanup_benchmark_temp_dir_logs_and_swallows_failure(monkeypatch, tmp_path, capsys):
    bench_dir = tmp_path / "momo_tts_bench_case"
    bench_dir.mkdir()
    monkeypatch.setattr(
        "backend.tts.qwen_clone.shutil.rmtree",
        lambda path: (_ for _ in ()).throw(PermissionError(32, "still busy")),
    )
    monkeypatch.setattr("backend.tts.qwen_clone.time.sleep", lambda _: None)
    monkeypatch.setattr("backend.tts.qwen_clone.os.name", "nt")

    qwen_clone._cleanup_benchmark_temp_dir(bench_dir)

    captured = capsys.readouterr()
    assert bench_dir.exists()
    assert "[startup] tts benchmark cleanup skipped" in captured.out


def test_benchmark_precision_modes_keeps_windows_fish_float16(monkeypatch):
    monkeypatch.setattr("backend.tts.qwen_clone.platform.system", lambda: "Windows")

    assert qwen_clone._benchmark_precision_modes("fish-speech-1.5", "gpu") == ("float16", "float32")
    assert qwen_clone._benchmark_precision_modes("s1-mini", "gpu") == ("float16", "float32")
    assert qwen_clone._benchmark_precision_modes("qwen3-tts-12hz-0.6b-base", "gpu") == ("float16", "float32")


def test_benchmark_precision_modes_keep_kokoro_and_melo_float32_only():
    assert qwen_clone._benchmark_precision_modes("kokoro-82m-zh", "gpu") == ("float32",)
    assert qwen_clone._benchmark_precision_modes("kokoro-82m-zh", "cpu") == ("float32",)
    assert qwen_clone._benchmark_precision_modes("melotts-chinese", "gpu") == ("float32",)
    assert qwen_clone._benchmark_precision_modes("melotts-chinese", "cpu") == ("float32",)


def test_benchmark_candidates_skip_mps_for_kokoro_and_melo_on_macos(monkeypatch):
    monkeypatch.setattr("backend.tts.qwen_clone.platform.system", lambda: "Darwin")
    plans = [
        SimpleNamespace(name="mps", device_mode="mps", semantic_dispatch_mode="single"),
        SimpleNamespace(name="cpu", device_mode="cpu", semantic_dispatch_mode="single"),
    ]

    kokoro = qwen_clone._benchmark_candidates_for_profile("kokoro-82m-zh", plans)
    melo = qwen_clone._benchmark_candidates_for_profile("melotts-chinese", plans)

    assert [item.name for item in kokoro] == ["cpu-float32"]
    assert [item.name for item in melo] == ["cpu-float32"]

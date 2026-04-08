from __future__ import annotations

import gc
import hashlib
import platform
import queue
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger


@dataclass(frozen=True)
class SemanticRuntimePlan:
    name: str
    device_mode: str
    semantic_dispatch_mode: str


@dataclass(frozen=True)
class SemanticBenchmarkResult:
    name: str
    device_mode: str
    semantic_dispatch_mode: str
    elapsed_ms: int
    ok: bool
    preload_ms: int | None = None
    synth_ms: int | None = None
    precision_mode: str | None = None
    peak_vram_mb: float | None = None
    detail: str = ""
    ram_mb: float | None = None
    vram_mb: float | None = None


@dataclass(frozen=True)
class SemanticBenchmarkOption:
    name: str
    plan: SemanticRuntimePlan | None
    skip_reason: str = ""


def accelerate_available() -> bool:
    try:
        import accelerate  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_accelerator_mode() -> str | None:
    if torch.cuda.is_available():
        return "gpu"
    mps = getattr(getattr(torch.backends, "mps", None), "is_available", None)
    if callable(mps) and mps():
        return "mps"
    return None


def benchmark_options_for_current_host() -> list[SemanticBenchmarkOption]:
    accelerator = resolve_accelerator_mode()
    accelerator_option = "mps" if platform.system() == "Darwin" else "gpu"
    options: list[SemanticBenchmarkOption] = []

    if accelerator:
        options.append(
            SemanticBenchmarkOption(
                name=accelerator,
                plan=SemanticRuntimePlan(name=accelerator, device_mode=accelerator, semantic_dispatch_mode="single"),
            )
        )
        if accelerate_available():
            options.append(
                SemanticBenchmarkOption(
                    name="auto",
                    plan=SemanticRuntimePlan(
                        name=f"semantic-auto-{accelerator}",
                        device_mode=accelerator,
                        semantic_dispatch_mode="auto",
                    ),
                )
            )
        else:
            options.append(
                SemanticBenchmarkOption(
                    name="auto",
                    plan=None,
                    skip_reason="accelerate is not installed, so semantic auto dispatch is unavailable",
                )
            )
    else:
        options.append(
            SemanticBenchmarkOption(
                name=accelerator_option,
                plan=None,
                skip_reason=f"{accelerator_option} backend is not available on this host",
            )
        )
        options.append(
            SemanticBenchmarkOption(
                name="auto",
                plan=None,
                skip_reason=f"auto benchmark requires an available {accelerator_option} backend",
            )
        )

    options.append(
        SemanticBenchmarkOption(
            name="cpu",
            plan=SemanticRuntimePlan(name="cpu", device_mode="cpu", semantic_dispatch_mode="single"),
        )
    )
    return options


def benchmark_plans_for_current_host() -> list[SemanticRuntimePlan]:
    return [item.plan for item in benchmark_options_for_current_host() if item.plan is not None]


def make_semantic_queue(
    checkpoint_path: str,
    device: str,
    precision: torch.dtype,
    semantic_dispatch_mode: str,
    compile: bool = False,
):
    input_queue: queue.Queue = queue.Queue()
    init_event = threading.Event()
    init_error: list[Exception] = []

    def worker() -> None:
        try:
            model, decode_one_token, input_device = load_semantic_runtime(
                checkpoint_path,
                device=device,
                precision=precision,
                semantic_dispatch_mode=semantic_dispatch_mode,
                compile=compile,
            )
        except Exception as exc:  # pragma: no cover - startup path
            init_error.append(exc)
            init_event.set()
            return

        init_event.set()
        inference_module = __import__(
            "fish_speech.models.text2semantic.inference",
            fromlist=["GenerateRequest", "WrappedGenerateResponse", "generate_long"],
        )
        GenerateRequest = getattr(inference_module, "GenerateRequest")
        WrappedGenerateResponse = getattr(inference_module, "WrappedGenerateResponse")
        generate_long = getattr(inference_module, "generate_long")

        while True:
            item = input_queue.get()
            if item is None:
                break
            if not isinstance(item, GenerateRequest):
                continue
            kwargs = dict(item.request)
            kwargs["device"] = input_device
            response_queue = item.response_queue
            try:
                for chunk in generate_long(model=model, decode_one_token=decode_one_token, **kwargs):
                    response_queue.put(WrappedGenerateResponse(status="success", response=chunk))
            except Exception as exc:  # pragma: no cover - runtime path
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=exc))

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()
    if init_error:
        raise init_error[0]
    return input_queue


def load_semantic_runtime(
    checkpoint_path: str,
    device: str,
    precision: torch.dtype,
    semantic_dispatch_mode: str,
    compile: bool = False,
) -> tuple[Any, Any, str]:
    inference_module = __import__(
        "fish_speech.models.text2semantic.inference",
        fromlist=["decode_one_token_ar"],
    )
    llama_module = __import__(
        "fish_speech.models.text2semantic.llama",
        fromlist=["BaseModelArgs", "DualARTransformer", "FishTokenizer", "KVCache", "TransformerBlock", "find_multiple"],
    )
    decode_one_token = getattr(inference_module, "decode_one_token_ar")
    BaseModelArgs = getattr(llama_module, "BaseModelArgs")
    DualARTransformer = getattr(llama_module, "DualARTransformer")
    FishTokenizer = getattr(llama_module, "FishTokenizer")
    KVCache = getattr(llama_module, "KVCache")
    find_multiple = getattr(llama_module, "find_multiple")

    if semantic_dispatch_mode != "auto":
        model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)
        model = model.to(device=device, dtype=precision).eval()
        _setup_semantic_caches(model, device, next(model.parameters()).dtype, KVCache, find_multiple)
        return model, decode_one_token, device

    if not accelerate_available():
        raise RuntimeError("accelerate is required for semantic auto dispatch")

    from accelerate import init_empty_weights, load_checkpoint_and_dispatch

    config = BaseModelArgs.from_pretrained(str(checkpoint_path))
    tokenizer = FishTokenizer.from_pretrained(checkpoint_path)
    with init_empty_weights():
        model = DualARTransformer(config, tokenizer=tokenizer)

    filtered_checkpoint = _filtered_semantic_checkpoint(Path(checkpoint_path) / "model.pth")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(filtered_checkpoint),
        device_map="auto",
        no_split_module_classes=["TransformerBlock"],
        dtype=precision,
        offload_buffers=False,
        force_hooks=True,
        strict=False,
    ).eval()
    input_device = _dispatch_input_device(model, preferred=device)
    _move_shared_buffers(model, input_device)
    _setup_semantic_caches(model, input_device, precision, KVCache, find_multiple, per_layer_device=True)
    return model, decode_one_token, input_device


def _filtered_semantic_checkpoint(model_file: Path) -> Path:
    stat = model_file.stat()
    digest = hashlib.sha1(f"{model_file}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")).hexdigest()[:12]
    target = Path("tmp") / f"semantic_filtered_{digest}.pth"
    if target.exists():
        return target

    weights = torch.load(model_file, map_location="cpu", mmap=True, weights_only=True)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    if next(iter(weights.keys())).startswith("model."):
        weights = {key.replace("model.", ""): value for key, value in weights.items()}
    weights = {key: value for key, value in weights.items() if not key.startswith("audio_")}
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, target)
    return target


def _dispatch_input_device(model: Any, preferred: str) -> str:
    device_map = getattr(model, "hf_device_map", {}) or {}
    for value in device_map.values():
        if isinstance(value, str) and value not in {"cpu", "disk"}:
            return value
        if isinstance(value, int):
            return f"cuda:{value}"
    return preferred


def _move_shared_buffers(model: Any, device: str) -> None:
    target = torch.device(device)
    for name in ("freqs_cis", "causal_mask", "fast_freqs_cis"):
        buffer = getattr(model, name, None)
        if buffer is not None:
            setattr(model, name, buffer.to(target))


def _setup_semantic_caches(
    model: Any,
    main_device: str,
    dtype: torch.dtype,
    kv_cache_cls: Any,
    find_multiple: Any,
    *,
    per_layer_device: bool = False,
) -> None:
    max_seq_len = find_multiple(model.config.max_seq_len, 8)
    model.max_seq_len = max_seq_len
    model.max_batch_size = 1

    for layer in model.layers:
        device = _module_device(layer.attention) if per_layer_device else torch.device(main_device)
        layer.attention.kv_cache = kv_cache_cls(
            1,
            max_seq_len,
            model.config.n_local_heads,
            model.config.head_dim,
            dtype=dtype,
        ).to(device)

    if hasattr(model, "fast_layers"):
        for layer in model.fast_layers:
            device = _module_device(layer.attention) if per_layer_device else torch.device(main_device)
            layer.attention.kv_cache = kv_cache_cls(
                1,
                model.config.num_codebooks,
                model.config.fast_n_local_heads,
                model.config.fast_head_dim,
                dtype=dtype,
            ).to(device)


def _module_device(module: Any) -> torch.device:
    for param in module.parameters():
        return param.device
    return torch.device("cpu")


def cleanup_torch_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except Exception:
            pass
    gc.collect()

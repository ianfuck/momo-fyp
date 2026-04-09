from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.device_utils import get_tts_device
from backend.model_manager import _ensure_tts_model
from backend.tts.model_profiles import GPT_SOVITS_V4_PROFILE
from backend.tts.provider_runtimes import _patch_gpt_sovits_frontend


def utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot GPT-SoVITS V4 trainer for a single reference clip.")
    parser.add_argument(
        "--base-model-path",
        default=f"model/huggingface/hf_snapshots/{GPT_SOVITS_V4_PROFILE.local_dir_name}",
        help="Path to the downloaded GPT-SoVITS V4 base model folder.",
    )
    parser.add_argument("--reference-audio", required=True, help="Path to the reference audio clip.")
    parser.add_argument(
        "--reference-transcript",
        required=True,
        help="Transcript text or a path to a UTF-8 transcript file for the reference audio clip.",
    )
    parser.add_argument("--name", default="", help="Output model folder name. Defaults to a timestamped name.")
    parser.add_argument(
        "--output-root",
        default="model/huggingface/hf_snapshots",
        help="Root directory where the trained model folder will be created.",
    )
    parser.add_argument(
        "--work-root",
        default="tmp/gpt_sovits_v4_fewshot",
        help="Scratch workspace for dataset prep and training logs.",
    )
    parser.add_argument("--speaker-name", default="speaker0", help="Speaker id written into the GPT-SoVITS list file.")
    parser.add_argument("--language", default="zh", choices=["zh"], help="Training language.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"], help="Training device preference.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for both S1 and S2 stages.")
    parser.add_argument("--s1-epochs", type=int, default=2, help="S1 GPT fine-tune epochs.")
    parser.add_argument("--s2-epochs", type=int, default=2, help="S2 SoVITS fine-tune epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare configs and print commands without training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_name = args.name.strip() or f"momo__gpt-sovits-v4-fewshot-{utc_now_slug()}"
    reference_transcript = _load_reference_transcript(args.reference_transcript)
    base_model_path = Path(args.base_model_path)
    if not base_model_path.exists():
        print(f"[setup] downloading base model into {base_model_path}", flush=True)
        _ensure_tts_model(str(base_model_path))
    base_model_path = base_model_path.resolve()
    work_root = Path(args.work_root).resolve() / model_name
    exp_dir = work_root / "exp"
    output_dir = Path(args.output_root).resolve() / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    _patch_gpt_sovits_frontend(base_model_path)
    metadata_path = work_root / "metadata.list"
    metadata_path.write_text(
        f"{Path(args.reference_audio).resolve()}|{args.speaker_name}|{args.language}|{reference_transcript}\n",
        encoding="utf-8",
    )
    _prepare_dataset(base_model_path, work_root, exp_dir, metadata_path, args)
    if args.dry_run:
        print(f"[dry-run] prepared workspace at {work_root}", flush=True)
        return 0
    gpt_weight = _run_s1_training(base_model_path, work_root, exp_dir, args, model_name)
    sovits_weight = _run_s2_training(base_model_path, work_root, exp_dir, args, model_name)
    packaged = _package_output(base_model_path, output_dir, Path(args.reference_audio), reference_transcript, gpt_weight, sovits_weight)
    print(f"[done] packaged model at {packaged}", flush=True)
    return 0


def _load_reference_transcript(value: str) -> str:
    candidate = Path(value).expanduser()
    raw_text = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    normalized = "".join(line.strip() for line in raw_text.splitlines() if line.strip()).strip()
    if not normalized:
        raise ValueError("Reference transcript is empty.")
    return normalized


def _normalize_semantic_tsv(path: Path) -> None:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No semantic tokens were generated in {path}")
    if lines[0].startswith("item_name\tsemantic_audio"):
        return
    path.write_text("item_name\tsemantic_audio\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _prepare_dataset(base_model_path: Path, work_root: Path, exp_dir: Path, metadata_path: Path, args: argparse.Namespace) -> None:
    print("[stage] prepare dataset", flush=True)
    env = _base_env(base_model_path, args)
    env.update(
        {
            "inp_text": str(metadata_path),
            "inp_wav_dir": "",
            "exp_name": work_root.name,
            "i_part": "0",
            "all_parts": "1",
            "opt_dir": str(exp_dir),
            "bert_pretrained_dir": str(base_model_path / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
            "cnhubert_base_dir": str(base_model_path / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
            "pretrained_s2G": str(base_model_path / "GPT_SoVITS" / "pretrained_models" / "gsv-v4-pretrained" / "s2Gv4.pth"),
        }
    )
    s2_config_path = work_root / "s2_prep.json"
    _write_s2_config(base_model_path, exp_dir, work_root, args, s2_config_path, include_model_version=False)
    env["s2config_path"] = str(s2_config_path)
    commands = [
        [sys.executable, "GPT_SoVITS/prepare_datasets/1-get-text.py"],
        [sys.executable, "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
        [sys.executable, "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
    ]
    for index, command in enumerate(commands, start=1):
        _run_command(command, cwd=base_model_path, env=env, prefix=f"prep{index}")
    shutil.move(exp_dir / "2-name2text-0.txt", exp_dir / "2-name2text.txt")
    semantic_path = exp_dir / "6-name2semantic.tsv"
    shutil.move(exp_dir / "6-name2semantic-0.tsv", semantic_path)
    _normalize_semantic_tsv(semantic_path)


def _run_s1_training(
    base_model_path: Path,
    work_root: Path,
    exp_dir: Path,
    args: argparse.Namespace,
    model_name: str,
) -> Path:
    print("[stage] train s1", flush=True)
    config_path = work_root / "s1.yaml"
    _write_s1_config(base_model_path, exp_dir, work_root, args, config_path, model_name)
    _run_command(
        [sys.executable, "GPT_SoVITS/s1_train.py", "-c", str(config_path)],
        cwd=base_model_path,
        env=_base_env(base_model_path, args),
        prefix="s1",
    )
    weights_dir = work_root / "weights" / "gpt"
    candidates = sorted(weights_dir.glob("*.ckpt"))
    if not candidates:
        raise RuntimeError(f"S1 training did not produce any checkpoint in {weights_dir}")
    return candidates[-1]


def _run_s2_training(
    base_model_path: Path,
    work_root: Path,
    exp_dir: Path,
    args: argparse.Namespace,
    model_name: str,
) -> Path:
    print("[stage] train s2", flush=True)
    config_path = work_root / "s2.json"
    _write_s2_config(base_model_path, exp_dir, work_root, args, config_path)
    _run_command(
        [sys.executable, "GPT_SoVITS/s2_train_v3.py", "-c", str(config_path)],
        cwd=base_model_path,
        env=_base_env(base_model_path, args),
        prefix="s2",
    )
    weights_dir = work_root / "weights" / "sovits"
    candidates = sorted(weights_dir.glob("*.pth"))
    if not candidates:
        raise RuntimeError(f"S2 training did not produce any checkpoint in {weights_dir}")
    return candidates[-1]


def _package_output(
    base_model_path: Path,
    output_dir: Path,
    reference_audio: Path,
    reference_transcript: str,
    gpt_weight: Path,
    sovits_weight: Path,
) -> Path:
    print("[stage] package model", flush=True)
    assets_dir = output_dir / "assets"
    weights_dir = output_dir / "weights"
    assets_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    packaged_ref_audio = assets_dir / reference_audio.name
    packaged_ref_text = assets_dir / "reference.txt"
    packaged_gpt = weights_dir / gpt_weight.name
    packaged_sovits = weights_dir / sovits_weight.name
    shutil.copy2(reference_audio, packaged_ref_audio)
    packaged_ref_text.write_text(reference_transcript.strip() + "\n", encoding="utf-8")
    shutil.copy2(gpt_weight, packaged_gpt)
    shutil.copy2(sovits_weight, packaged_sovits)
    manifest = {
        "format": "momo-gpt-sovits-v4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model_path": os.path.relpath(base_model_path, output_dir),
        "t2s_weights_path": os.path.relpath(packaged_gpt, output_dir),
        "vits_weights_path": os.path.relpath(packaged_sovits, output_dir),
        "reference_audio_path": os.path.relpath(packaged_ref_audio, output_dir),
        "reference_text_path": os.path.relpath(packaged_ref_text, output_dir),
        "supports_voice_clone": False,
    }
    (output_dir / "gpt_sovits_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def _write_s1_config(
    base_model_path: Path,
    exp_dir: Path,
    work_root: Path,
    args: argparse.Namespace,
    config_path: Path,
    model_name: str,
) -> None:
    import torch

    config = torch.load(
        base_model_path / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt",
        map_location="cpu",
        weights_only=False,
    )["config"]
    config["output_dir"] = str(work_root / "logs" / "s1")
    config["train_semantic_path"] = str(exp_dir / "6-name2semantic.tsv")
    config["train_phoneme_path"] = str(exp_dir / "2-name2text.txt")
    config.setdefault("train", {})
    config["train"]["epochs"] = args.s1_epochs
    config["train"]["batch_size"] = args.batch_size
    config["train"]["save_every_n_epoch"] = 1
    config["train"]["precision"] = 16 if _use_half_precision(args) else 32
    config["train"]["if_save_latest"] = True
    config["train"]["if_save_every_weights"] = True
    config["train"]["half_weights_save_dir"] = str(work_root / "weights" / "gpt")
    config["train"]["exp_name"] = model_name
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["train"]["half_weights_save_dir"]).mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_s2_config(
    base_model_path: Path,
    exp_dir: Path,
    work_root: Path,
    args: argparse.Namespace,
    config_path: Path,
    *,
    include_model_version: bool = True,
) -> None:
    import torch

    config = torch.load(
        base_model_path / "GPT_SoVITS" / "pretrained_models" / "gsv-v4-pretrained" / "s2Gv4.pth",
        map_location="cpu",
        weights_only=False,
    )["config"]
    config.setdefault("train", {})
    config.setdefault("data", {})
    config.setdefault("model", {})
    config["train"]["epochs"] = args.s2_epochs
    config["train"]["batch_size"] = args.batch_size
    config["train"]["fp16_run"] = _use_half_precision(args)
    config["train"]["log_interval"] = 1
    config["train"]["eval_interval"] = 1
    config["train"]["pretrained_s2G"] = str(base_model_path / "GPT_SoVITS" / "pretrained_models" / "gsv-v4-pretrained" / "s2Gv4.pth")
    config["train"]["pretrained_s2D"] = ""
    config["train"]["if_save_every_weights"] = True
    config["data"]["exp_dir"] = str(exp_dir)
    if include_model_version:
        config["model"]["version"] = "v4"
    else:
        config["model"].pop("version", None)
    config["save_weight_dir"] = str(work_root / "weights" / "sovits")
    config["s2_ckpt_dir"] = str(work_root / "logs" / "s2")
    Path(config["save_weight_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["s2_ckpt_dir"]).mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str], prefix: str) -> None:
    print(f"[run:{prefix}] {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{prefix}] {line.rstrip()}", flush=True)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"{prefix} failed with exit code {code}")


def _base_env(base_model_path: Path, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    bootstrap_dir = _ensure_bootstrap_dir(base_model_path)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(bootstrap_dir),
            str(base_model_path),
            str(base_model_path / "GPT_SoVITS"),
            env.get("PYTHONPATH", ""),
        ]
    ).strip(os.pathsep)
    env["version"] = "v4"
    env["is_half"] = "True" if _use_half_precision(args) else "False"
    if _resolve_training_device(args) == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["_CUDA_VISIBLE_DEVICES"] = ""
    return env


def _resolve_training_device(args: argparse.Namespace) -> str:
    resolved = get_tts_device(args.device)
    if resolved == "mps":
        return "cpu"
    return resolved


def _use_half_precision(args: argparse.Namespace) -> bool:
    return _resolve_training_device(args).startswith("cuda")


def _ensure_bootstrap_dir(base_model_path: Path) -> Path:
    bootstrap_dir = base_model_path / ".momo_bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize_path = bootstrap_dir / "sitecustomize.py"
    sitecustomize_path.write_text(
        "\n".join(
            [
                "import contextlib",
                "import sys",
                "",
                "with contextlib.suppress(Exception):",
                "    import transformers.modeling_utils as modeling_utils",
                "    import transformers.utils.import_utils as import_utils",
                "    import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None",
                "    modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None",
                "",
                "with contextlib.suppress(Exception):",
                "    import jieba",
                "    import jieba.posseg as posseg",
                "    import types",
                "    proxy = types.ModuleType('jieba_fast')",
                "    proxy.__dict__.update(jieba.__dict__)",
                "    proxy.posseg = posseg",
                "    sys.modules['jieba_fast'] = proxy",
                "    sys.modules['jieba_fast.posseg'] = posseg",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return bootstrap_dir


if __name__ == "__main__":
    raise SystemExit(main())

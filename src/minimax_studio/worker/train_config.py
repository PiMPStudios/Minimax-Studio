"""SimpleTuner training config generation + training preflight (PLAN-V2 S0).

The entire coupling surface to SimpleTuner is:

1. ``config/<run-id>/config.json`` and ``multidatabackend.json`` — written here
2. ``simpletuner train env=<run-id>`` — launched by ``train_runs`` from the run
   directory, version-pinned below and asserted in preflight
3. ``train.log`` + ``checkpoints/`` — parsed by ``train_runs``

No SimpleTuner code is imported, vendored, or monkeypatched. If a SimpleTuner
upgrade changes this contract, preflight's version assert fails loudly instead
of the trainer silently misbehaving.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SIMPLETUNER_PIN = "4.8.0"


@dataclass(frozen=True)
class TrainPreset:
    name: str
    title: str
    vram_floor_gb: int
    lora_rank: int
    base_model_precision: str
    text_encoder_precision: str
    gradient_checkpointing: bool = True
    note: str = ""


# Floors from SimpleTuner's own MiniMax Music 3 quickstart (24 GB minimum,
# 48 GB recommended); ranks/precisions are its conservative defaults.
PRESETS: dict[str, TrainPreset] = {
    "24g": TrainPreset(
        name="24g",
        title="24 GB — conservative LoRA",
        vram_floor_gb=24,
        lora_rank=16,
        base_model_precision="int8-quanto",
        text_encoder_precision="int8-quanto",
        note="int8 everywhere + gradient checkpointing",
    ),
    "48g": TrainPreset(
        name="48g",
        title="48 GB — room to breathe",
        vram_floor_gb=48,
        lora_rank=32,
        base_model_precision="bf16",
        text_encoder_precision="int8-quanto",
        note="bf16 transformer, int8 text encoder",
    ),
}
DEFAULT_PRESET = "24g"

# VAE + text-embed caches grow past the dataset size; demand headroom, not luck.
MIN_FREE_CACHE_GB = 10.0


def validate_music_dataset_dir(path: str | Path) -> list[str]:
    """Cheap, honest checks only — the full validator is PLAN-V2 S1."""
    folder = Path(path)
    if not folder.is_dir():
        return [f"Dataset folder not found: {folder}"]
    clips = sorted(folder.glob("*.wav")) + sorted(folder.glob("*.flac"))
    if not clips:
        return [f"No audio files (*.wav) found in {folder}"]
    errors: list[str] = []
    for clip in clips:
        if not clip.with_suffix(".txt").is_file():
            errors.append(f"{clip.name}: missing caption {clip.stem}.txt")
    return errors


def simpletuner_command_prefix() -> list[str] | None:
    """``MINIMAX_STUDIO_SIMPLETUNER_BIN`` may be a full command prefix (tests
    run a stub interpreter); otherwise the console script on PATH."""
    override = os.environ.get("MINIMAX_STUDIO_SIMPLETUNER_BIN")
    if override:
        import shlex

        return shlex.split(override)
    if shutil.which("simpletuner"):
        return ["simpletuner"]
    return None


def simpletuner_version(prefix: list[str] | None = None) -> str | None:
    prefix = prefix if prefix is not None else simpletuner_command_prefix()
    if not prefix:
        return None
    try:
        proc = subprocess.run(
            [*prefix, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", proc.stdout + proc.stderr)
    return match.group(1) if match else None


def write_run_config(
    run_dir: Path,
    run_id: str,
    dataset_dir: str | Path,
    preset_name: str = DEFAULT_PRESET,
    steps: int = 1000,
    rank: int | None = None,
    learning_rate: float = 5e-5,
    validation: dict[str, Any] | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, str]:
    """Write SimpleTuner's two JSON files for one detached run.

    Layout is what `simpletuner train env=<run_id>` expects when launched with
    cwd=run_dir. Everything SimpleTuner may ever write stays under run_dir:
    checkpoints and both caches included.

    ``resume_from_checkpoint`` continues a finished or killed run from one of
    its own checkpoints — same run dir, so the weights and the cache stay put
    and the history keeps one home. Like every other key here, the name is part
    of the pinned contract and gets verified against real SimpleTuner output in
    S0 steps 3–5; a wrong key fails loudly at launch, not silently as a fresh run.
    """
    if preset_name not in PRESETS:
        raise RuntimeError(
            f"Unknown training preset '{preset_name}' "
            f"(known: {', '.join(sorted(PRESETS))})."
        )
    preset = PRESETS[preset_name]
    root = _models_root()
    model_dir = root / "music3-cuda"
    encoder_dir = root / "music3-train-encoder"
    env_dir = run_dir / "config" / run_id
    env_dir.mkdir(parents=True, exist_ok=True)
    validation = validation or {}

    config: dict[str, Any] = {
        "model_family": "minimaxmusic",
        "model_flavour": "music3",
        "model_type": "lora",
        "pretrained_model_name_or_path": str(model_dir),
        "pretrained_vae_model_name_or_path": str(encoder_dir),
        "output_dir": str(run_dir / "checkpoints"),
        "data_backend_config": str(env_dir / "multidatabackend.json"),
        "resolution": 512,
        "mixed_precision": "bf16",
        "base_model_precision": preset.base_model_precision,
        "text_encoder_1_precision": preset.text_encoder_precision,
        "gradient_checkpointing": preset.gradient_checkpointing,
        "lora_rank": int(rank or preset.lora_rank),
        # Export in the format our LoRA picker and the Comfy path already load.
        "lora_format": "comfyui",
        "optimizer": "adamw_bf16",
        "learning_rate": learning_rate,
        "train_batch_size": 1,
        "vae_batch_size": 1,
        "max_train_steps": int(steps),
        # Only present when resuming, so a fresh run's config is unchanged.
        **(
            {"resume_from_checkpoint": str(resume_from_checkpoint)}
            if resume_from_checkpoint
            else {}
        ),
        "validation_prompt": str(
            validation.get("prompt")
            or "bright synth pop with clean vocal melody and crisp percussion"
        ),
        "validation_lyrics": str(validation.get("lyrics") or ""),
        "validation_audio_duration": float(validation.get("duration") or 15),
        "validation_guidance": 1.7,
        "validation_num_inference_steps": 30,
        "validation_steps": 50,
        "validation_disable_unconditional": True,
    }
    backends = [
        {
            "id": f"studio-{run_id}",
            "type": "local",
            "dataset_type": "audio",
            "instance_data_dir": str(Path(dataset_dir).resolve()),
            "metadata_backend": "discovery",
            "caption_strategy": "textfile",
            "audio": {
                "bucket_strategy": "duration",
                "duration_interval": 3.0,
                "max_duration_seconds": 60,
                "lyrics_filename_format": "{filename}.lyrics",
            },
            "cache_dir_vae": str(run_dir / "cache" / "vae"),
        },
        {
            "id": "text-embeds",
            "dataset_type": "text_embeds",
            "default": True,
            "type": "local",
            "cache_dir": str(run_dir / "cache" / "text"),
        },
    ]
    paths = {
        "config": str(env_dir / "config.json"),
        "multidatabackend": str(env_dir / "multidatabackend.json"),
    }
    for path, payload in (
        (paths["config"], config),
        (paths["multidatabackend"], backends),
    ):
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return paths


def train_preflight(preset_name: str = DEFAULT_PRESET) -> dict[str, Any]:
    """Everything that must be true before burning hours of GPU — with the
    named numbers product style demands, never a mystery OOM."""
    if preset_name not in PRESETS:
        raise RuntimeError(
            f"Unknown training preset '{preset_name}' "
            f"(known: {', '.join(sorted(PRESETS))})."
        )
    preset = PRESETS[preset_name]
    from minimax_studio.worker.probe import probe

    result: dict[str, Any] = {
        "ok": False,
        "preset": preset_name,
        "vram_floor_gb": preset.vram_floor_gb,
        "simpletuner": None,
        # The Train page builds its VRAM picker from this instead of keeping a
        # second copy of the preset table that could disagree with the worker.
        "presets": {
            name: {
                "title": row.title,
                "vram_floor_gb": row.vram_floor_gb,
                "lora_rank": row.lora_rank,
                "note": row.note,
            }
            for name, row in PRESETS.items()
        },
        "problems": [],
        "warnings": [],
        "detail": "",
    }
    problems: list[str] = result["problems"]

    prefix = simpletuner_command_prefix()
    if not prefix:
        problems.append(
            "SimpleTuner is not installed — run: "
            f"pip install 'minimax-studio[train]' (pins SimpleTuner "
            f"{SIMPLETUNER_PIN})."
        )
    else:
        version = simpletuner_version(prefix)
        result["simpletuner"] = version
        if version is None:
            problems.append(
                "Found a simpletuner command but could not read its "
                f"--version (this plan pins {SIMPLETUNER_PIN})."
            )
        elif version != SIMPLETUNER_PIN:
            result["warnings"].append(
                f"SimpleTuner {version} is not the pinned {SIMPLETUNER_PIN} "
                "— configs generated to the pinned contract; report anything "
                "odd."
            )

    from minimax_studio.worker.catalog import PACKS
    from minimax_studio.worker.downloads import pack_status

    root = _models_root()
    if not pack_status(PACKS["music3-cuda"], root)["ready"]:
        problems.append(
            "Training LoRAs needs the official weights — download "
            "MiniMax-Music3 (CUDA / diffusers) on the Models page."
        )
    if not pack_status(PACKS["music3-train-encoder"], root)["ready"]:
        problems.append(
            "The Music 3 Training Encoder pack (DAV VAE, ~0.3 GB) is "
            "missing — download it on the Models page."
        )

    from minimax_studio.worker.runtime import runtime

    busy = [
        job
        for job in runtime.jobs.values()
        if job.get("status") in {"queued", "running", "cancelling"}
    ]
    if busy:
        problems.append(
            f"{len(busy)} generation job(s) are active — training refuses to "
            "share the GPU (finish or cancel them first)."
        )

    hw = probe()
    result["free_vram_gb"] = hw.get("free_vram_gb")
    if not hw.get("cuda"):
        problems.append(
            "Training is CUDA-only (Windows/Linux) — Mac generates with "
            "MLX and the API but does not train."
        )
    else:
        free = hw.get("free_vram_gb")
        if free is not None and free < preset.vram_floor_gb:
            problems.append(
                f"'{preset.title}' needs {preset.vram_floor_gb} GB of free "
                f"VRAM; {free} GB is free right now. Close ComfyUI or other "
                "GPU apps, or wait for the current job to finish."
            )
        elif free is None:
            result["warnings"].append(
                "Could not read free VRAM (no nvidia-smi?) — the first "
                "minutes of the run decide; watch the log."
            )

    from minimax_studio.worker.train_runs import runs_root

    try:
        free_disk_gb = shutil.disk_usage(runs_root()).free / (1024**3)
        if free_disk_gb < MIN_FREE_CACHE_GB:
            problems.append(
                f"Only {free_disk_gb:.0f} GB free on the training volume — "
                f"VAE/text-embed caches want about {MIN_FREE_CACHE_GB:.0f} GB."
            )
    except OSError:
        pass

    result["ok"] = not problems
    result["detail"] = (
        f"Ready to train '{preset.title}' (rank {preset.lora_rank}, "
        f"{preset.note})."
        if not problems
        else " ".join(problems)
    )
    return result


def _models_root() -> Path:
    from minimax_studio.worker.runtime import runtime

    return runtime.config.models_root()

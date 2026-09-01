"""SimpleTuner training config generation + training preflight (PLAN-V2 S0).

The entire coupling surface to SimpleTuner is:

1. ``config/<run-id>/config.json`` and ``multidatabackend.json`` — written here
2. ``simpletuner train env=<run-id>`` — launched by ``train_runs`` from the run
   directory, version-pinned below and asserted in preflight
3. ``train.log`` + ``checkpoints/`` — parsed by ``train_runs``
4. ``st_startup/sitecustomize.py`` — PYTHONPATH hook in the *trainer*
   process only (Dynamo skip + SDNQ ConvRot shim). The GUI never imports it.

No SimpleTuner code is imported by the GUI or the worker. If a SimpleTuner
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

# VAE + text-embed caches grow past the dataset size; demand headroom, not luck.
# Each preset can raise it — video caches are another order of magnitude, and
# "40 GB wanted" beats a full disk at step 400.
MIN_FREE_CACHE_GB = 10.0


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
    #: which trainer this preset belongs to — a Music dataset and an H3 preset
    #: must never meet, and the only way to guarantee that is to say it here.
    family: str = "music"
    #: SimpleTuner's CPU-offload: how a 24 GB card reaches H3 at all.
    ram_torch: bool = False
    #: the pixel bucket the trainer fills (stills and frames alike).
    resolution: int = 512
    #: packs that must be on disk, all of them…
    packs: tuple[str, ...] = ()
    #: …and packs where any one will do (the H3 weights share one folder).
    packs_any: tuple[str, ...] = ()
    min_free_cache_gb: float = MIN_FREE_CACHE_GB
    #: SimpleTuner ``model_flavour``. Music is ``music3``; H3 INT8 tiers are
    #: ``convrot-int8`` and the 80 GB tier is official ``fl2va``. ``h3`` is not
    #: a flavour the pinned trainer knows.
    flavour: str = "music3"


# Floors from SimpleTuner's own MiniMax Music 3 quickstart (24 GB minimum,
# 48 GB recommended); ranks/precisions are its conservative defaults. The H3
# tiers are the four SimpleTuner names for LoRA on the video model (24G with
# RamTorch CPU-offload, 32G, 48G, 80G).
#
# H3 keys were written against SimpleTuner's docs until S0 metal (0.2.37)
# confirmed them on a 32 GB Blackwell card. The RamTorch 24 GB warning stays:
# that run is memory-bound even when the keys are right.
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
    "h3-24g": TrainPreset(
        name="h3-24g",
        title="24 GB — H3 LoRA with RamTorch",
        vram_floor_gb=24,
        lora_rank=16,
        # Flavour already loads ConvRot INT8; quanto on top is the wrong 8-bit.
        # SimpleTuner's own 24G example is ``base_model_precision: no_change``.
        base_model_precision="no_change",
        text_encoder_precision="int8-quanto",
        note="RamTorch CPU-offload, ConvRot INT8, 480p buckets",
        family="h3",
        ram_torch=True,
        resolution=480,
        # Official audio VAE + Qwen3-VL (~63 GB), not the 130 GB transformer.
        packs=("h3-train", "h3-fl2va"),
        min_free_cache_gb=40.0,
        flavour="convrot-int8",
    ),
    "h3-32g": TrainPreset(
        name="h3-32g",
        title="32 GB — H3 LoRA",
        vram_floor_gb=32,
        lora_rank=16,
        base_model_precision="no_change",
        text_encoder_precision="int8-quanto",
        note="ConvRot INT8 + gradient checkpointing, 768p buckets",
        family="h3",
        resolution=768,
        packs=("h3-train", "h3-fl2va"),
        min_free_cache_gb=40.0,
        flavour="convrot-int8",
    ),
    "h3-48g": TrainPreset(
        name="h3-48g",
        title="48 GB — H3 LoRA, rank 32",
        vram_floor_gb=48,
        lora_rank=32,
        base_model_precision="no_change",
        text_encoder_precision="int8-quanto",
        note="ConvRot INT8 at 768p, rank 32",
        family="h3",
        resolution=768,
        packs=("h3-train", "h3-fl2va"),
        min_free_cache_gb=60.0,
        flavour="convrot-int8",
    ),
    "h3-80g": TrainPreset(
        name="h3-80g",
        title="80 GB — H3 LoRA, full precision transformer",
        vram_floor_gb=80,
        lora_rank=32,
        base_model_precision="no_change",
        text_encoder_precision="int8-quanto",
        note="official FL2VA bf16 transformer at 1080p buckets",
        family="h3",
        resolution=1080,
        packs_any=("h3-diffusers-fl2va", "h3-diffusers-ref2va"),
        min_free_cache_gb=80.0,
        flavour="fl2va",
    ),
}
DEFAULT_PRESET = "24g"

#: Retired 0.2.37: S0 metal confirmed minimax_h3_target_mode, ramtorch,
#: resolution, and validation_prompt on SimpleTuner 4.8.0's own stdout.
#: Left as an empty tuple so ``if H3_UNVERIFIED_KEYS:`` stays the gate.
H3_UNVERIFIED_KEYS: tuple[str, ...] = ()


def validate_video_dataset_dir(path: str | Path) -> list[str]:
    """Cheap, honest checks for an H3 dataset — the full validator is
    :func:`datasets._validate_video_dir`; this is the preflight-shaped version
    that answers in milliseconds without measuring a single frame."""
    from minimax_studio.worker.datasets import MEDIA_BY_KIND, caption_body_problems

    folder = Path(path)
    if not folder.is_dir():
        return [f"Dataset folder not found: {folder}"]
    media = [
        item
        for item in sorted(folder.iterdir())
        if item.suffix.lower() in MEDIA_BY_KIND["video"]
    ]
    if not media:
        return [
            f"No stills or clips in {folder} — H3 wants .png/.jpg stills or "
            "short .mp4 clips"
        ]
    errors: list[str] = []
    for item in media:
        txt = item.with_suffix(".txt")
        if not txt.is_file():
            errors.append(f"{item.name}: missing caption {item.stem}.txt")
            continue
        errors.extend(caption_body_problems(txt))
    return errors


def validate_music_dataset_dir(path: str | Path) -> list[str]:
    """Cheap, honest checks only — the full validator is PLAN-V2 S1."""
    from minimax_studio.worker.datasets import MEDIA_BY_KIND, caption_body_problems

    folder = Path(path)
    if not folder.is_dir():
        return [f"Dataset folder not found: {folder}"]
    exts = MEDIA_BY_KIND["music"]
    clips = [
        item
        for item in sorted(folder.iterdir())
        if item.suffix.lower() in exts and not item.name.startswith(".")
    ]
    if not clips:
        return [f"No audio files ({', '.join(exts)}) found in {folder}"]
    errors: list[str] = []
    for clip in clips:
        txt = clip.with_suffix(".txt")
        if not txt.is_file():
            errors.append(f"{clip.name}: missing caption {clip.stem}.txt")
            continue
        errors.extend(caption_body_problems(txt))
    return errors


def _default_prompt(preset: TrainPreset) -> str:
    """Something worth rendering every 50 steps, per family."""
    if preset.family == "h3":
        return (
            "a steady camera push across a neon-lit street at dusk, shallow "
            "depth of field, film grain"
        )
    return "bright synth pop with clean vocal melody and crisp percussion"


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
    dataset_spec: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write SimpleTuner's two JSON files for one detached run.

    Layout is what `simpletuner train env=<run_id>` expects when launched with
    cwd=run_dir. Everything SimpleTuner may ever write stays under run_dir:
    checkpoints and both caches included.

    ``dataset_spec`` is what :func:`datasets.dataset_spec` measured — kind,
    whether the set holds stills or clips, and the chosen H3 target mode. The
    preset family and the dataset kind have to agree: an H3 preset pointed at a
    Music dataset would train the wrong model and call it provenance, so it is
    refused here rather than discovered an hour in.

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
    dataset_spec = dataset_spec or {"kind": "music"}
    kind = str(dataset_spec.get("kind") or "music")
    wanted_family = "h3" if kind == "video" else "music"
    if preset.family != wanted_family:
        raise RuntimeError(
            f"Preset '{preset.title}' trains "
            f"{'MiniMax H3 (video/stills)' if preset.family == 'h3' else 'MiniMax Music 3 (audio)'}, "
            f"but this dataset is a {kind} one. Pick the matching trainer on the "
            "preset row."
        )
    root = _models_root()
    env_dir = run_dir / "config" / run_id
    env_dir.mkdir(parents=True, exist_ok=True)
    validation = validation or {}
    common = {
        "model_type": "lora",
        "output_dir": str(run_dir / "checkpoints"),
        "data_backend_config": str(env_dir / "multidatabackend.json"),
        "resolution": preset.resolution,
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
        "max_train_steps": int(steps),
        # SimpleTuner 4.8.0: max_train_steps without this is a ValueError
        # ("you must set --num_train_epochs to 0"). Official H3/Music examples
        # all write it; metal on 2026-08-30 confirmed the refusal.
        "num_train_epochs": 0,
        # Only present when resuming, so a fresh run's config is unchanged.
        **(
            {"resume_from_checkpoint": str(resume_from_checkpoint)}
            if resume_from_checkpoint
            else {}
        ),
        "validation_prompt": str(validation.get("prompt") or _default_prompt(preset)),
        "validation_steps": 50,
        # Official 24G examples write 50. A 200-step smoke with the default
        # (often 500) would finish without a checkpoint to Install.
        "checkpoint_step_interval": 50,
    }
    if preset.family == "h3":
        config: dict[str, Any] = {
            **common,
            # SimpleTuner 4.8.0 registers the family as ``minimaxh3`` (see
            # ``simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora``).
            # ``minimax_h3`` is not a family the pinned trainer knows.
            "model_family": "minimaxh3",
            "model_flavour": preset.flavour,
            "pretrained_model_name_or_path": str(root / "h3-diffusers"),
            **(
                {
                    "pretrained_transformer_model_name_or_path": transformer,
                    "pretrained_transformer_subfolder": None,
                }
                if (transformer := _local_h3_int8_transformer(root))
                and preset.flavour == "convrot-int8"
                else {}
            ),
            # Flavour convrot-int8 otherwise pulls Kijai's INT8 ConvRot VAE and
            # SimpleTuner 4.8.0 + sdnq 0.2.6 crash in SDNQDequantizer (metal
            # 2026-08-30). The Comfy fp16 VAE is already on disk and loads.
            **(
                {"pretrained_vae_model_name_or_path": vae}
                if (vae := _local_h3_video_vae(root))
                else {}
            ),
            # JSON keys match CLI flags: ``--ramtorch``, not ``ram_torch``.
            # Official 24G writes transformer_percent=100 + text-encoder
            # offload; 32G writes percent=0. Metal 2026-08-30: omitting
            # percent still defaulted to 100 and peaked the 32 GB card.
            **(
                {
                    "ramtorch": True,
                    "ramtorch_text_encoder": True,
                    "ramtorch_transformer_percent": 100,
                }
                if preset.ram_torch
                else {}
            ),
            # Official 24G/32G ConvRot examples. Metal 2026-08-30: without
            # these the INT8 DiT forward fragmented 32 GB and died at step 9
            # trying to allocate 316 MiB.
            "attention_mechanism": "native-efficient",
            "offload_during_startup": True,
            "vae_enable_slicing": True,
            "minimax_h3_target_mode": str(
                dataset_spec.get("h3_target_mode") or "video"
            ),
            "resolution_type": "pixel_area",
            "flow_schedule_shift": 12.0,
            "audio_flow_schedule_shift": 3.0,
            "validation_guidance": 1.0,
            "validation_disable_unconditional": True,
            # Official H3 LoRA examples all write this block. Leaving it out
            # does not "leave the default on" — it trains without the audio-head
            # safety net PLAN-V2 wanted kept.
            "distillation_method": "h3_drift",
            "distillation_config": {
                "h3_drift": {
                    "audio_weight": 1.0,
                    "balance": "token",
                    "loss_weight": 0.5,
                    "sft_loss_weight": 1.0,
                    "video_weight": 1.0,
                }
            },
        }
        backend_type = "image" if (
            dataset_spec.get("has_stills") and not dataset_spec.get("has_clips")
        ) else "video"
        # No bucket/aspect-ratio block for H3: those keys are the ones we have
        # not seen SimpleTuner accept for this family, and an invented key is
        # worse than the trainer's own default. Stills and clips both ride the
        # resolution buckets the preset names.
    else:
        config = {
            **common,
            "model_family": "minimaxmusic",
            "model_flavour": "music3",
            "pretrained_model_name_or_path": str(root / "music3-cuda"),
            "pretrained_vae_model_name_or_path": str(root / "music3-train-encoder"),
            "vae_batch_size": 1,
            "validation_lyrics": str(validation.get("lyrics") or ""),
            "validation_audio_duration": float(validation.get("duration") or 15),
            "validation_guidance": 1.7,
            "validation_num_inference_steps": 30,
            "validation_disable_unconditional": True,
        }
        backend_type = "audio"
        from minimax_studio.worker.datasets import MAX_SECONDS

        media_block: dict[str, Any] = {
            "bucket_strategy": "duration",
            "duration_interval": 3.0,
            # Same cap the validator advertises — a "ready" 90 s clip must not
            # be skipped silently at SimpleTuner discovery.
            "max_duration_seconds": MAX_SECONDS,
            "lyrics_filename_format": "{filename}.lyrics",
        }
    backend: dict[str, Any] = {
        "id": f"studio-{run_id}",
        "type": "local",
        "dataset_type": backend_type,
        "instance_data_dir": str(Path(dataset_dir).resolve()),
        "metadata_backend": "discovery",
        "caption_strategy": "textfile",
        "cache_dir_vae": str(run_dir / "cache" / "vae"),
    }
    if backend_type == "audio":
        # Music 3 buckets by duration; the key shape is the one the pinned
        # trainer has been writing since S0.
        backend["audio"] = media_block
    backends = [
        backend,
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


def train_preflight(
    preset_name: str = DEFAULT_PRESET, dataset_dir: str | Path | None = None
) -> dict[str, Any]:
    """Everything that must be true before burning hours of GPU — with the
    named numbers product style demands, never a mystery OOM.

    ``dataset_dir`` is optional because the Train page checks requirements while
    the user is still choosing; once a dataset is picked, passing it here is what
    catches "that preset is for the other model" before the run exists.
    """
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
        "family": preset.family,
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
                # The Train page filters these by the dataset's kind, so the
                # only place the tier list lives is this table.
                "family": row.family,
                "ram_torch": row.ram_torch,
                "resolution": row.resolution,
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
    if preset.family == "music":
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
    for pack_id in preset.packs:
        if not pack_status(PACKS[pack_id], root)["ready"]:
            problems.append(
                f"'{preset.title}' needs {PACKS[pack_id].title} — download it "
                "on the Models page."
            )
    if preset.packs_any and not any(
        pack_status(PACKS[pack_id], root)["ready"] for pack_id in preset.packs_any
    ):
        titles = " or ".join(
            PACKS[pack_id].title for pack_id in preset.packs_any
        )
        problems.append(
            f"Training H3 LoRAs needs the H3 weights in the diffusers layout — "
            f"download {titles} on the Models page. That folder is the audio "
            "VAE and Qwen3-VL text encoder; the Comfy generate pack is not a "
            "substitute for it."
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
    from minimax_studio.worker.device import selected_free_vram_gb

    free = selected_free_vram_gb(hw)
    result["free_vram_gb"] = free
    if not hw.get("cuda"):
        problems.append(
            "Training is CUDA-only (Windows/Linux) — Mac generates with "
            "MLX and the API but does not train."
        )
    else:
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

    if dataset_dir:
        from minimax_studio.worker.datasets import dataset_spec

        spec = dataset_spec(dataset_dir)
        result["dataset_kind"] = spec["kind"]
        wanted = "video" if preset.family == "h3" else "music"
        if spec["kind"] != wanted:
            problems.append(
                f"'{preset.title}' trains "
                f"{'MiniMax H3 — stills and short clips' if preset.family == 'h3' else 'MiniMax Music 3'}"
                f", and this dataset holds "
                f"{'clips and stills' if spec['kind'] == 'video' else 'audio'}"
                f" ({spec['stills'] + spec['clips'] or spec['audio_files']} "
                "file(s)). Pick the preset for the other model."
            )
        elif spec["kind"] == "video" and spec["has_stills"] and spec["has_clips"]:
            result["warnings"].append(
                f"This set mixes {spec['stills']} still(s) with {spec['clips']} "
                "clip(s) in one run. Nothing forbids it, and nothing proves it "
                "either — two runs, one per kind, is the comparison you can "
                "actually read."
            )

    try:
        free_disk_gb = shutil.disk_usage(runs_root()).free / (1024**3)
        if free_disk_gb < preset.min_free_cache_gb:
            problems.append(
                f"Only {free_disk_gb:.0f} GB free on the training volume — "
                f"'{preset.title}' wants about {preset.min_free_cache_gb:.0f} GB "
                "for its VAE/text-embed caches."
            )
    except OSError:
        pass

    if preset.family == "h3":
        if H3_UNVERIFIED_KEYS:
            result["warnings"].append(
                "H3 LoRA training has not run on this build yet: "
                f"{', '.join(H3_UNVERIFIED_KEYS)} are written from SimpleTuner's "
                "documentation, not its output. Watch the first minutes of the log "
                "and report anything odd — that session is what retires this "
                "warning."
            )
        if preset.ram_torch:
            result["warnings"].append(
                "RamTorch keeps layers in system RAM: expect the run to be "
                "memory-bound and slower than the same tier on 32 GB."
            )

    result["ok"] = not problems
    result["detail"] = (
        f"Ready to train '{preset.title}' (rank {preset.lora_rank}, "
        f"{preset.note})."
        if not problems
        else " ".join(problems)
    )
    return result


def _first_named_weight(models_root: Path, name: str) -> str | None:
    try:
        from minimax_studio.worker.model_paths import search_roots

        roots = search_roots(models_root, None)
    except Exception:
        roots = [models_root]
    for base in roots:
        for candidate in (base / name, base / "diffusion_models" / name, base / "vae" / name):
            if candidate.is_file():
                return str(candidate.resolve())
    return None


def _local_h3_int8_transformer(models_root: Path) -> str | None:
    """Comfy-Org ConvRot INT8 DiT if it is already on disk (PLAN-V2 S0 step 5)."""
    return _first_named_weight(models_root, "minimax_h3_fl2va_pruned_int8_convrot.safetensors")


def _local_h3_video_vae(models_root: Path) -> str | None:
    return _first_named_weight(models_root, "minimax_h3_video_vae_fp16.safetensors")


def _models_root() -> Path:
    from minimax_studio.worker.runtime import runtime

    return runtime.config.models_root()

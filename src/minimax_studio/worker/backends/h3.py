from __future__ import annotations

from pathlib import Path
from typing import Any

from minimax_studio.worker.catalog import PACKS
from minimax_studio.worker.downloads import pack_status
from minimax_studio.worker.jobs import JobRequest, update_job
from minimax_studio.worker.runtime import runtime


def duration_to_frames(seconds: float) -> int:
    """Snap duration to H3's 17n+5 frame grid at 24 fps, clamped to 5–15 s."""
    raw = max(5.0, min(15.0, float(seconds)))
    target = raw * 24.0
    n = max(0, round((target - 5) / 17))
    frames = 17 * n + 5
    while frames / 24.0 < 5.0:
        n += 1
        frames = 17 * n + 5
    while frames / 24.0 > 15.0 and n > 0:
        n -= 1
        frames = 17 * n + 5
    return frames


def generate_h3(job_id: str, request: JobRequest) -> dict[str, Any]:
    backend = _resolve_backend(request.backend)
    if backend == "api":
        from minimax_studio.worker.backends.h3_api import generate_h3_api

        return generate_h3_api(job_id, request)
    if backend == "stub":
        raise RuntimeError("H3 has no stub renderer.")

    dest = runtime.config.history_root() / job_id
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "video.mp4"

    pack_dir = runtime.config.models_root() / PACKS["h3-diffusers-fl2va"].local_dir
    if not pack_status(PACKS["h3-diffusers-fl2va"], runtime.config.models_root())["ready"]:
        raise RuntimeError(
            "Download “MiniMax H3 FL2VA (official diffusers)” on the Models page. "
            "The Comfy-Org INT8 pack is for Comfy / later loaders, not this generate path yet."
        )

    update_job(job_id, message="Loading MiniMax H3", progress=0.1)
    try:
        import torch
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.utils import load_image
        from diffusers.utils.export_utils import encode_video
    except ImportError as exc:
        raise RuntimeError(
            "Local H3 needs torch and a MiniMax-H3-capable diffusers. "
            "pip install torch diffusers accelerate"
        ) from exc

    workflow = "ref2va" if request.mode == "ref2va" else "fl2va"
    if workflow == "ref2va" and not (
        pack_dir / "transformer_ref" / "config.json"
    ).exists():
        raise RuntimeError("Download the official Ref2VA transformer pack for reference mode.")

    cache_key = f"{pack_dir}:{workflow}"
    pipe = runtime.h3_pipe if runtime.h3_pipe_path == cache_key else None
    if pipe is None:
        manager = ComponentsManager()
        if torch.cuda.is_available():
            manager.enable_auto_cpu_offload(device="cuda")
        pipe = ModularPipeline.from_pretrained(
            str(pack_dir),
            workflow=workflow,
            components_manager=manager,
        )
        pipe.load_components(workflow=workflow, dtype=torch.bfloat16)
        runtime.h3_pipe = pipe
        runtime.h3_pipe_path = cache_key

    steps = int(request.steps)
    loras = list(request.loras)
    if request.speed == "fast":
        turbo = _find_turbo_lora()
        if turbo and not any("turbo" in (item.get("id") or "").lower() for item in loras):
            loras.append({"id": turbo, "strength": 1.0})
        if steps >= 16:
            steps = 8
    num_frames = duration_to_frames(request.duration_s)
    width = int(request.width or 960)
    height = int(request.height or 544)
    generator = None
    if request.seed >= 0:
        generator = torch.Generator().manual_seed(int(request.seed))

    kwargs: dict[str, Any] = {
        "prompt": request.prompt,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "generator": generator,
        "output": ["videos", "audio", "sampling_rate"],
        "width": width,
        "height": height,
    }
    _attach_media(request, kwargs, load_image)

    update_job(
        job_id,
        message=f"Sampling {num_frames} frames ({num_frames / 24:.1f}s)",
        progress=0.35,
    )
    for item in loras:
        path = item.get("id")
        if path and hasattr(pipe, "load_lora_weights"):
            try:
                pipe.load_lora_weights(path)
            except Exception:
                pass
    results = pipe(**kwargs)
    update_job(job_id, message="Muxing MP4", progress=0.9)
    encode_video(
        results["videos"][0],
        fps=24,
        output_path=str(out_path),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    return {"output_path": str(out_path), "backend": "cuda", "media_type": "video"}


def _resolve_backend(requested: str) -> str:
    name = requested.lower()
    if name in {"api", "local", "cuda", "stub"}:
        if name == "local":
            return "cuda"
        return name
    # auto
    root = runtime.config.models_root()
    local_ready = pack_status(PACKS["h3-diffusers-fl2va"], root)["ready"]
    from minimax_studio.worker.probe import probe

    hw = probe()
    if local_ready and hw.get("cuda"):
        return "cuda"
    if runtime.config.minimax_api_key:
        return "api"
    if local_ready:
        return "cuda"
    raise RuntimeError(
        "No local H3 diffusers pack and no MiniMax API key. "
        "Download the official FL2VA pack or add a key in Settings."
    )


def _find_turbo_lora() -> str | None:
    root = runtime.config.models_root()
    candidates = [
        root
        / "h3-comfy"
        / "loras"
        / "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
    ]
    extra = root / "loras"
    if extra.is_dir():
        candidates.extend(sorted(extra.glob("*turbo*.safetensors")))
    for path in candidates:
        if Path(path).is_file():
            return str(path)
    return None


def _attach_media(request: JobRequest, kwargs: dict[str, Any], load_image) -> None:
    assets = {item.get("role"): item.get("path") for item in request.assets if item.get("path")}
    if request.mode in {"i2va", "fl2va"} and assets.get("first_frame"):
        kwargs["image"] = load_image(assets["first_frame"])
    if request.mode in {"l2va", "fl2va"} and assets.get("last_frame"):
        kwargs["last_image"] = load_image(assets["last_frame"])
    if request.mode == "ref2va":
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3AudioReference,
            MiniMaxH3ImageReference,
            MiniMaxH3VideoReference,
        )

        refs = []
        for item in request.assets:
            path = item.get("path")
            if not path:
                continue
            suffix = Path(path).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                refs.append(MiniMaxH3ImageReference.from_file(path))
            elif suffix in {".mp4", ".mov", ".webm", ".mkv"}:
                refs.append(MiniMaxH3VideoReference.from_file(path))
            elif suffix in {".wav", ".mp3", ".flac", ".m4a"}:
                refs.append(MiniMaxH3AudioReference.from_file(path))
        if not refs:
            raise RuntimeError("Reference mode needs at least one image, video, or audio file.")
        kwargs["references"] = refs

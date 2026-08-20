from __future__ import annotations

from pathlib import Path
from typing import Any

from minimax_studio.h3_timing import duration_to_frames, resolve_dims
from minimax_studio.worker.catalog import PACKS
from minimax_studio.worker.downloads import pack_status
from minimax_studio.worker.jobs import JobRequest, update_job
from minimax_studio.worker.runtime import runtime


INT8_NEEDS_COMFY = (
    "Comfy-Org INT8 H3 is on disk, but those files use Comfy convrot kernels — "
    "diffusers cannot load them. Start ComfyUI (Settings → ComfyUI URL, default "
    "http://127.0.0.1:8188) and generate again, or download the official FL2VA "
    "diffusers pack on the Models page for in-process generate."
)


def generate_h3(job_id: str, request: JobRequest) -> dict[str, Any]:
    backend = resolve_h3_backend(request.backend)
    if backend == "api":
        from minimax_studio.worker.backends.h3_api import generate_h3_api

        return generate_h3_api(job_id, request)
    if backend == "comfy":
        from minimax_studio.worker.backends.h3_comfy import generate_h3_comfy

        return generate_h3_comfy(job_id, request)
    if backend == "stub":
        raise RuntimeError("H3 has no stub renderer.")

    dest = runtime.config.history_root() / job_id
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "video.mp4"

    pack_dir = runtime.config.models_root() / PACKS["h3-diffusers-fl2va"].local_dir
    official = pack_status(PACKS["h3-diffusers-fl2va"], runtime.config.models_root())
    if not official["ready"]:
        int8 = pack_status(PACKS["h3-fl2va"], runtime.config.models_root())
        if int8["ready"]:
            raise RuntimeError(INT8_NEEDS_COMFY + f" Found at {int8['path']}.")
        raise RuntimeError(
            "Download MiniMax H3 on the Models page (Comfy-Org INT8 or official "
            "diffusers FL2VA), or add a MiniMax API key in Settings."
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
        from minimax_studio.worker.device import select_cuda_device

        device = select_cuda_device()
        if torch.cuda.is_available():
            manager.enable_auto_cpu_offload(device=device)
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
    if request.attention.strip().lower() in {"sage", "sageattention"}:
        update_job(
            job_id,
            message="SageAttention is used on the Comfy path; official diffusers uses PyTorch attention",
        )
    if request.speed == "fast":
        turbo = _find_turbo_lora(request.mode)
        if turbo and not any("turbo" in (item.get("id") or "").lower() for item in loras):
            loras.append({"id": turbo, "strength": 1.0})
        if steps >= 16:
            steps = 8
    num_frames = duration_to_frames(request.duration_s)
    width, height = resolve_dims(
        request.resolution,
        request.ratio,
        request.width,
        request.height,
        request.quality,
    )
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
    _apply_loras(pipe, loras)
    from minimax_studio.worker.jobs import is_cancelled, step_cancel_callback

    if is_cancelled(job_id):
        raise RuntimeError("Cancelled")
    try:
        results = pipe(
            **kwargs, callback_on_step_end=step_cancel_callback(job_id, steps)
        )
    except TypeError:
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


def resolve_h3_backend(requested: str) -> str:
    name = requested.lower()
    root = runtime.config.models_root()
    official = pack_status(PACKS["h3-diffusers-fl2va"], root)["ready"]
    int8 = pack_status(PACKS["h3-fl2va"], root)

    def _comfy_up() -> bool:
        from minimax_studio.worker.backends.h3_comfy import comfy_reachable

        return comfy_reachable()

    if name == "api":
        return "api"
    if name == "stub":
        return "stub"
    if name == "cuda":
        from minimax_studio.worker.probe import probe as _probe

        hw = _probe()
        if hw.get("apple_silicon") and not hw.get("cuda"):
            raise RuntimeError(
                "Local MiniMax H3 on Apple Silicon is gated in v1. Use the MiniMax API."
            )
        return "cuda"
    if name == "comfy":
        from minimax_studio.worker.probe import probe as _probe_comfy

        hw = _probe_comfy()
        if hw.get("apple_silicon") and not hw.get("cuda"):
            raise RuntimeError(
                "Local MiniMax H3 on Apple Silicon is gated in v1. Use the MiniMax API."
            )
        if not int8["ready"]:
            raise RuntimeError(
                "Comfy backend needs the Comfy-Org INT8 FL2VA files. "
                "Download that pack or point Settings at your ComfyUI models folder."
            )
        if not _comfy_up():
            raise RuntimeError(INT8_NEEDS_COMFY + f" Found at {int8['path']}.")
        return "comfy"
    from minimax_studio.worker.probe import probe

    hw = probe()
    if hw.get("apple_silicon") and not hw.get("cuda"):
        if name in {"local", "cuda", "comfy"}:
            raise RuntimeError(
                "Local MiniMax H3 on Apple Silicon is gated in v1 (slow and RAM-hungry). "
                "Use the MiniMax API, or generate on an NVIDIA CUDA machine."
            )
        if name in {"auto", ""}:
            if runtime.config.minimax_api_key:
                return "api"
            raise RuntimeError(
                "Local MiniMax H3 on Apple Silicon is gated in v1. "
                "Add a MiniMax API key in Settings, or use a CUDA machine."
            )
    torch_ok = bool(hw.get("torch_available"))
    if name == "local":
        if official and torch_ok:
            return "cuda"
        if int8["ready"] and _comfy_up():
            return "comfy"
        if int8["ready"]:
            raise RuntimeError(INT8_NEEDS_COMFY + f" Found at {int8['path']}.")
        return "cuda"

    from minimax_studio.worker.device import selected_vram_gb

    vram = selected_vram_gb(hw)
    comfy_ok = int8["ready"] and _comfy_up()
    official_ok = official and hw.get("cuda") and torch_ok and (vram >= 24 or vram == 0)
    if vram and vram < 24 and comfy_ok:
        return "comfy"
    if official_ok:
        return "cuda"
    if comfy_ok:
        return "comfy"
    if runtime.config.minimax_api_key:
        return "api"
    if official and torch_ok:
        return "cuda"
    if official and not torch_ok:
        raise RuntimeError(
            "Official H3 diffusers is on disk but PyTorch is not in the Studio venv. "
            "pip install torch, or start ComfyUI to use INT8 packs."
        )
    if int8["ready"]:
        raise RuntimeError(INT8_NEEDS_COMFY + f" Found at {int8['path']}.")
    raise RuntimeError(
        "No local H3 pack and no MiniMax API key. "
        "Download Comfy-Org INT8 or official FL2VA on Models, or add a key in Settings."
    )


def _apply_loras(pipe: Any, loras: list[dict[str, Any]]) -> None:
    if not loras or not hasattr(pipe, "load_lora_weights"):
        return
    names: list[str] = []
    weights: list[float] = []
    for index, item in enumerate(loras):
        path = item.get("id") or item.get("path")
        if not path:
            continue
        name = f"adapter{index}"
        try:
            pipe.load_lora_weights(path, adapter_name=name)
        except TypeError:
            pipe.load_lora_weights(path)
            names = []
            break
        except Exception:
            continue
        names.append(name)
        weights.append(float(item.get("strength") or 1.0))
    setter = getattr(pipe, "set_adapters", None)
    if setter and names:
        try:
            setter(names, adapter_weights=weights)
        except Exception:
            try:
                setter(names)
            except Exception:
                pass


def _find_turbo_lora(mode: str = "t2va") -> str | None:
    from minimax_studio.worker.model_paths import search_roots

    names = [
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    ]
    if mode == "ref2va":
        names = [
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            *names,
        ]
    root = runtime.config.models_root()
    roots = search_roots(root, runtime.config.comfy_models_dir)
    candidates: list[Path] = []
    for folder in roots:
        for name in names:
            candidates.extend(
                [
                    folder / "h3-comfy" / "loras" / name,
                    folder / "minimax-h3" / "loras" / name,
                    folder / "loras" / name,
                ]
            )
        lora_dir = folder / "loras"
        if lora_dir.is_dir():
            if mode == "ref2va":
                candidates.extend(sorted(lora_dir.glob("*ref2*turbo*.safetensors")))
            candidates.extend(sorted(lora_dir.glob("*turbo*.safetensors")))
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
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

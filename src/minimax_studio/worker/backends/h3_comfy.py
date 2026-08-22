from __future__ import annotations

import random
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from minimax_studio.worker.jobs import JobRequest, is_cancelled, update_job
from minimax_studio.worker.runtime import runtime

UNET_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
UNET_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def comfy_base() -> str:
    return (runtime.config.comfy_url or "http://127.0.0.1:8188").rstrip("/")


_REACH_AT = 0.0
_REACH_OK = False
_REACH_TTL_S = 2.0


def reset_comfy_reach_cache() -> None:
    global _REACH_AT, _REACH_OK
    _REACH_AT = 0.0
    _REACH_OK = False


def comfy_reachable() -> bool:
    global _REACH_AT, _REACH_OK
    now = time.monotonic()
    if now - _REACH_AT < _REACH_TTL_S:
        return _REACH_OK
    ok = False
    try:
        response = httpx.get(f"{comfy_base()}/system_stats", timeout=0.6)
        ok = response.status_code < 400
    except httpx.HTTPError:
        try:
            response = httpx.get(f"{comfy_base()}/queue", timeout=0.6)
            ok = response.status_code < 400
        except httpx.HTTPError:
            ok = False
    _REACH_AT = now
    _REACH_OK = ok
    return ok


def generate_h3_comfy(job_id: str, request: JobRequest) -> dict[str, Any]:
    if not comfy_reachable():
        raise RuntimeError(
            "Comfy-Org INT8 H3 needs a running ComfyUI. "
            f"Start it at {comfy_base()} (Settings → ComfyUI URL), "
            "or download the official FL2VA diffusers pack for in-process generate."
        )

    from minimax_studio.worker.backends.h3 import duration_to_frames, resolve_dims
    from minimax_studio.worker.catalog import PACKS
    from minimax_studio.worker.downloads import pack_status

    dest = runtime.config.history_root() / job_id
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "video.mp4"
    frames = duration_to_frames(request.duration_s)
    width, height = resolve_dims(
        request.resolution,
        request.ratio,
        request.width,
        request.height,
        request.quality,
    )
    seed = int(request.seed) if request.seed >= 0 else random.randint(0, 2_147_483_647)
    steps = int(request.steps)
    mode = request.mode
    unet_name = UNET_FL2VA
    if mode == "ref2va":
        ref_pack = pack_status(PACKS["h3-ref2va"], runtime.config.models_root())
        if not ref_pack["ready"]:
            raise RuntimeError(
                "Reference mode on Comfy needs the Comfy-Org Ref2VA INT8 file. "
                "Download that pack on Models, or use official Ref2VA / the MiniMax API."
            )
        unet_name = UNET_REF2VA

    stack: list[dict[str, Any]] = []
    if request.speed == "fast":
        from minimax_studio.worker.backends.h3 import _find_turbo_lora

        turbo_path = _find_turbo_lora(mode)
        if not turbo_path:
            raise RuntimeError(
                "Fast needs the MiniMax H3 Turbo LoRA. Download that pack on Models, "
                "or switch Inspector Speed to Quality."
            )
        stack.append({"id": turbo_path, "strength": 1.0})
        if mode == "ref2va" and steps >= 16:
            steps = 4
        elif steps >= 16:
            steps = 8
    for item in request.loras or []:
        path = item.get("id") or item.get("path")
        if not path:
            continue
        name = Path(str(path)).name.lower()
        if any(Path(str(existing.get("id") or "")).name.lower() == name for existing in stack):
            continue
        stack.append({"id": path, "strength": float(item.get("strength") or 1.0)})
    lora_name = Path(str(stack[0]["id"])).name if stack else None
    lora_strength = float(stack[0]["strength"]) if stack else 1.0
    extra_loras = [
        {"id": Path(str(item["id"])).name, "strength": item["strength"]}
        for item in stack[1:]
    ]

    use_sage = request.attention.strip().lower() in {"sage", "sageattention"}
    refs = _split_assets(request) if mode == "ref2va" else {"images": [], "videos": [], "audios": []}
    if mode == "ref2va" and not (refs["images"] or refs["videos"] or refs["audios"]):
        raise RuntimeError("Reference mode needs at least one image, video, or audio file.")

    update_job(job_id, message="Uploading to ComfyUI", progress=0.12)
    first_name = last_name = None
    uploaded_images: list[str] = []
    uploaded_videos: list[str] = []
    uploaded_audios: list[str] = []
    if mode == "ref2va":
        uploaded_images = [_upload_file(path) for path in refs["images"][:9]]
        uploaded_videos = [_upload_file(path) for path in refs["videos"][:3]]
        uploaded_audios = [_upload_file(path) for path in refs["audios"][:3]]
    else:
        first_name = _maybe_upload(request, "first_frame")
        last_name = _maybe_upload(request, "last_frame")

    graph_kwargs: dict[str, Any] = dict(
        prompt=request.prompt,
        width=width,
        height=height,
        length=frames,
        seed=seed,
        steps=steps,
        mode=mode,
        first_image=first_name,
        last_image=last_name,
        ref_images=uploaded_images,
        ref_videos=uploaded_videos,
        ref_audios=uploaded_audios,
        lora_name=lora_name,
        lora_strength=lora_strength,
        extra_loras=extra_loras,
        unet_name=unet_name,
        sage=use_sage,
        scheduler="beta" if mode == "ref2va" else "simple",
        ref_image_size=request.ref_image_size or "match",
        prefix=f"minimax_studio/{job_id}",
    )
    graph = build_h3_comfy_graph(**graph_kwargs)
    client_id = uuid.uuid4().hex
    update_job(job_id, message="Queued on ComfyUI", progress=0.2)
    with httpx.Client(timeout=30.0) as client:
        submitted = _submit_graph(client, graph, client_id)
        if (
            submitted.status_code >= 400
            and use_sage
            and "PathchSageAttentionKJ" in (submitted.text or "")
        ):
            graph_kwargs["sage"] = False
            graph = build_h3_comfy_graph(**graph_kwargs)
            update_job(
                job_id,
                message="SageAttention node missing; retrying without it",
                progress=0.22,
            )
            submitted = _submit_graph(client, graph, client_id)
        if submitted.status_code >= 400:
            detail = submitted.text
            try:
                detail = str(submitted.json().get("error") or submitted.json())
            except Exception:
                pass
            raise RuntimeError(f"ComfyUI rejected the graph: {detail}")
        prompt_id = submitted.json().get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI returned no prompt_id: {submitted.text}")
        outputs = _poll_history(client, prompt_id, job_id)
        media = _first_media(outputs)
        if not media:
            raise RuntimeError(f"ComfyUI finished without a video: {outputs}")
        update_job(job_id, message="Downloading ComfyUI result", progress=0.92)
        query = urlencode(
            {
                "filename": media.get("filename") or "",
                "subfolder": media.get("subfolder") or "",
                "type": media.get("type") or "output",
            }
        )
        video = client.get(f"{comfy_base()}/view?{query}", timeout=120.0)
        video.raise_for_status()
        out_path.write_bytes(video.content)
    return {"output_path": str(out_path), "backend": "comfy", "media_type": "video"}


def build_h3_comfy_graph(
    *,
    prompt: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    steps: int,
    mode: str = "t2va",
    first_image: str | None = None,
    last_image: str | None = None,
    ref_images: list[str] | None = None,
    ref_videos: list[str] | None = None,
    ref_audios: list[str] | None = None,
    lora_name: str | None = None,
    lora_strength: float = 1.0,
    extra_loras: list[dict[str, Any]] | None = None,
    unet_name: str = UNET_FL2VA,
    sage: bool = False,
    scheduler: str = "simple",
    ref_image_size: str = "match",
    prefix: str = "minimax_studio/h3",
) -> dict[str, Any]:
    use_lora = bool(lora_name)
    model_src: list[Any] = ["unet", 0]
    graph: dict[str, Any] = {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": CLIP_NAME,
                "type": "minimax",
                "device": "default",
            },
        },
        "vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VIDEO_VAE},
        },
        "avae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE},
        },
        "noise": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": int(seed)},
        },
        "sampler": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "vdec": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]},
        },
        "adec": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]},
        },
        "video": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["vdec", 0],
                "audio": ["adec", 0],
                "fps": 24,
                "bit_depth": 8,
            },
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["video", 0],
                "filename_prefix": prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }
    stack = []
    if use_lora and lora_name:
        stack.append((lora_name, float(lora_strength)))
    for item in extra_loras or []:
        name = item.get("id") or item.get("path") or item.get("lora_name")
        if name:
            stack.append((Path(str(name)).name, float(item.get("strength") or 1.0)))
    prev = ["unet", 0]
    for index, (name, strength) in enumerate(stack):
        key = "lora" if index == 0 else f"lora{index}"
        graph[key] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": prev,
                "lora_name": name,
                "strength_model": strength,
            },
        }
        prev = [key, 0]
    if stack:
        model_src = prev
    if sage:
        graph["sage"] = {
            "class_type": "PathchSageAttentionKJ",
            "inputs": {
                "model": model_src,
                "sage_attention": "auto",
                "allow_compile": False,
            },
        }
        model_src = ["sage", 0]
    graph["shift"] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {
            "model": model_src,
            "shift_video": 12.0,
            "shift_audio": 3.0,
        },
    }
    graph["sched"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": ["shift", 0],
            "scheduler": scheduler,
            "steps": int(steps),
            "denoise": 1.0,
        },
    }
    graph["guide"] = {
        "class_type": "BasicGuider",
        "inputs": {
            "model": ["shift", 0],
            "conditioning": ["cond", 0],
        },
    }
    graph["sample"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise", 0],
            "guider": ["guide", 0],
            "sampler": ["sampler", 0],
            "sigmas": ["sched", 0],
            "latent_image": ["cond", 1],
        },
    }

    if mode == "ref2va":
        cond_inputs: dict[str, Any] = {
            "clip": ["clip", 0],
            "vae": ["vae", 0],
            "audio_vae": ["avae", 0],
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "length": int(length),
            "ref_image_size": ref_image_size if ref_image_size in {"match", "max"} else "match",
        }
        for index, name in enumerate(ref_images or []):
            node_id = f"img{index}"
            graph[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": name},
            }
            cond_inputs[f"ref_images.ref_image_{index}"] = [node_id, 0]
        for index, name in enumerate(ref_videos or []):
            load_id = f"vid{index}"
            split_id = f"vcomp{index}"
            graph[load_id] = {
                "class_type": "LoadVideo",
                "inputs": {"file": name},
            }
            graph[split_id] = {
                "class_type": "GetVideoComponents",
                "inputs": {"video": [load_id, 0]},
            }
            cond_inputs[f"ref_videos.ref_video_{index}"] = [split_id, 0]
            cond_inputs[f"ref_video_audios.ref_video_audio_{index}"] = [split_id, 1]
        for index, name in enumerate(ref_audios or []):
            node_id = f"aud{index}"
            graph[node_id] = {
                "class_type": "LoadAudio",
                "inputs": {"audio": name},
            }
            cond_inputs[f"ref_audios.ref_audio_{index}"] = [node_id, 0]
        graph["cond"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": cond_inputs,
        }
        return graph

    graph["cond"] = {
        "class_type": "MiniMaxH3ImageToVideo",
        "inputs": {
            "clip": ["clip", 0],
            "vae": ["vae", 0],
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "length": int(length),
        },
    }
    if first_image:
        graph["first"] = {
            "class_type": "LoadImage",
            "inputs": {"image": first_image},
        }
        graph["cond"]["inputs"]["first_frame"] = ["first", 0]
    if last_image:
        graph["last"] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_image},
        }
        graph["cond"]["inputs"]["last_frame"] = ["last", 0]
    return graph


def _split_assets(request: JobRequest) -> dict[str, list[str]]:
    images: list[str] = []
    videos: list[str] = []
    audios: list[str] = []
    for item in request.assets:
        path = item.get("path")
        if not path:
            continue
        suffix = Path(path).suffix.lower()
        if suffix in IMAGE_EXT:
            images.append(path)
        elif suffix in VIDEO_EXT:
            videos.append(path)
        elif suffix in AUDIO_EXT:
            audios.append(path)
    return {"images": images, "videos": videos, "audios": audios}


def _maybe_upload(request: JobRequest, role: str) -> str | None:
    for item in request.assets:
        if item.get("role") == role and item.get("path"):
            return _upload_file(item["path"])
    return None


def _upload_file(path: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"Missing file: {path}")
    with source.open("rb") as handle:
        response = httpx.post(
            f"{comfy_base()}/upload/image",
            files={"image": (source.name, handle, "application/octet-stream")},
            data={"overwrite": "true"},
            timeout=120.0,
        )
    response.raise_for_status()
    body = response.json()
    name = body.get("name") or source.name
    sub = body.get("subfolder") or ""
    return f"{sub}/{name}" if sub else str(name)


def _submit_graph(client: httpx.Client, graph: dict[str, Any], client_id: str) -> httpx.Response:
    return client.post(
        f"{comfy_base()}/prompt",
        json={"prompt": graph, "client_id": client_id},
    )


def _poll_history(client: httpx.Client, prompt_id: str, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 45 * 60
    started = time.monotonic()
    while time.monotonic() < deadline:
        if is_cancelled(job_id):
            try:
                client.post(f"{comfy_base()}/interrupt")
            except httpx.HTTPError:
                pass
            raise RuntimeError("Cancelled")
        history = client.get(f"{comfy_base()}/history/{prompt_id}", timeout=15.0)
        if history.status_code == 404:
            time.sleep(1.5)
            continue
        history.raise_for_status()
        payload = history.json()
        record = payload.get(prompt_id) if isinstance(payload, dict) else None
        if record:
            status = record.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(
                    "ComfyUI job failed: " + _comfy_error_text(status, record)
                )
            outputs = record.get("outputs") or {}
            done = status.get("status_str") == "success" or status.get("completed") is True
            if done:
                return outputs
        elapsed = time.monotonic() - started
        progress = 0.25 + min(0.62, elapsed / 480.0)
        update_job(
            job_id,
            message=_comfy_progress_message(client, elapsed),
            progress=progress,
        )
        time.sleep(1.5)
    raise RuntimeError("Timed out waiting for ComfyUI.")


def _comfy_error_text(status: dict[str, Any], record: dict[str, Any] | None = None) -> str:
    messages = status.get("messages") or (record or {}).get("messages") or []
    for item in messages:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        kind, body = item[0], item[1]
        if kind != "execution_error" or not isinstance(body, dict):
            continue
        node = body.get("node_type") or body.get("node_id") or "node"
        msg = (body.get("exception_message") or body.get("exception_type") or "error")
        return f"{node}: {str(msg).strip()}"
    return str(messages or status)


def _comfy_progress_message(client: httpx.Client, elapsed: float) -> str:
    total = max(0, int(elapsed))
    mins, secs = divmod(total, 60)
    clock = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
    queue = _comfy_queue_message(client)
    if queue.startswith("ComfyUI:"):
        return f"{queue} · {clock}"
    return f"ComfyUI sampling {clock}"


def _comfy_queue_message(client: httpx.Client) -> str:
    try:
        response = client.get(f"{comfy_base()}/queue", timeout=3.0)
        data = response.json()
        running = len(data.get("queue_running") or [])
        pending = len(data.get("queue_pending") or [])
        if running or pending:
            return f"ComfyUI: {running} running, {pending} queued"
    except Exception:
        pass
    return "Sampling on ComfyUI"


def _first_media(outputs: dict[str, Any]) -> dict[str, Any] | None:
    for node in outputs.values():
        if not isinstance(node, dict):
            continue
        for items in node.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get("filename"):
                    return item
    return None

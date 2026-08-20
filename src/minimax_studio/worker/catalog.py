from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pack:
    id: str
    title: str
    summary: str
    repo_id: str
    local_dir: str
    approx_gb: float
    license_name: str
    family: str  # music | h3
    kind: str  # cuda | mlx | comfy
    ignore_patterns: tuple[str, ...] = ()
    allow_patterns: tuple[str, ...] | None = None
    marker_files: tuple[str, ...] = ()
    territory_notice: str | None = None


H3_TERRITORY = (
    "The MiniMax H3 Community License does not authorize using the open weights "
    "(or their outputs) in the US, EU, UK, or South Korea unless MiniMax grants "
    "a separate license. The MiniMax hosted API remains globally available."
)

PACKS: dict[str, Pack] = {
    "music3-cuda": Pack(
        id="music3-cuda",
        title="MiniMax-Music3 (CUDA / diffusers)",
        summary="Official weights for local song generation on NVIDIA.",
        repo_id="MiniMaxAI/MiniMax-Music3",
        local_dir="music3-cuda",
        approx_gb=63.0,
        license_name="MiniMax-Music3 Community License",
        family="music",
        kind="cuda",
        ignore_patterns=("figures/*", "assets/*", "scripts/*", ".gitattributes"),
        marker_files=("modular_model_index.json", "config.json"),
    ),
    "music3-mlx": Pack(
        id="music3-mlx",
        title="MiniMax-Music3 (MLX MXFP8)",
        summary="Apple Silicon pack for mlx-audio. Better lyric fidelity than MXFP4.",
        repo_id="mlx-community/MiniMax-Music3-mxfp8",
        local_dir="music3-mlx",
        approx_gb=16.0,
        license_name="MiniMax-Music3 Community License",
        family="music",
        kind="mlx",
        marker_files=("config.json", "model.safetensors.index.json"),
    ),
    "h3-diffusers-fl2va": Pack(
        id="h3-diffusers-fl2va",
        title="MiniMax H3 FL2VA (official diffusers)",
        summary="Text / first / last frame via ModularPipeline. Large. In-process generate (no ComfyUI).",
        repo_id="MiniMaxAI/MiniMax-H3",
        local_dir="h3-diffusers",
        approx_gb=130.0,
        license_name="MiniMax H3 Community License",
        family="h3",
        kind="diffusers",
        ignore_patterns=(
            "FL2VA/*",
            "Ref2VA/*",
            "transformer_ref/*",
            "assets/*",
            "scripts/*",
            "docs/*",
            "figures/*",
        ),
        marker_files=(
            "modular_model_index.json",
            "transformer/config.json",
            "text_encoder/config.json",
        ),
        territory_notice=H3_TERRITORY,
    ),
    "h3-diffusers-ref2va": Pack(
        id="h3-diffusers-ref2va",
        title="MiniMax H3 Ref2VA transformer (official)",
        summary="Adds transformer_ref/ to the official diffusers folder for omni-reference.",
        repo_id="MiniMaxAI/MiniMax-H3",
        local_dir="h3-diffusers",
        approx_gb=62.0,
        license_name="MiniMax H3 Community License",
        family="h3",
        kind="diffusers",
        allow_patterns=("transformer_ref/*",),
        marker_files=("transformer_ref/config.json",),
        territory_notice=H3_TERRITORY,
    ),
    "h3-fl2va": Pack(
        id="h3-fl2va",
        title="MiniMax H3 FL2VA (pruned INT8)",
        summary="Consumer CUDA pack (~42 GB). Detected in Studio or a ComfyUI models folder. Generate submits to ComfyUI when that server is running.",
        repo_id="Comfy-Org/MiniMax-H3",
        local_dir="h3-comfy",
        approx_gb=42.0,
        license_name="MiniMax H3 Community License",
        family="h3",
        kind="comfy",
        allow_patterns=(
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "vae/minimax_h3_video_vae_fp16.safetensors",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
        ),
        marker_files=(
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "vae/minimax_h3_video_vae_fp16.safetensors",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
        ),
        territory_notice=H3_TERRITORY,
    ),
    "h3-ref2va": Pack(
        id="h3-ref2va",
        title="MiniMax H3 Ref2VA (pruned INT8)",
        summary="Omni-reference checkpoint. Needs the FL2VA pack’s encoder and VAEs.",
        repo_id="Comfy-Org/MiniMax-H3",
        local_dir="h3-comfy",
        approx_gb=21.0,
        license_name="MiniMax H3 Community License",
        family="h3",
        kind="comfy",
        allow_patterns=(
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        ),
        marker_files=(
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        ),
        territory_notice=H3_TERRITORY,
    ),
    "h3-turbo": Pack(
        id="h3-turbo",
        title="MiniMax H3 Turbo LoRA (4-step)",
        summary="Fast sampling LoRA for Quality vs Fast on Generate Video.",
        repo_id="Comfy-Org/MiniMax-H3",
        local_dir="h3-comfy",
        approx_gb=0.8,
        license_name="MiniMax H3 Community License",
        family="h3",
        kind="comfy",
        allow_patterns=(
            "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        ),
        marker_files=(
            "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        ),
        territory_notice=H3_TERRITORY,
    ),
}


def pack_or_raise(pack_id: str) -> Pack:
    try:
        return PACKS[pack_id]
    except KeyError as exc:
        raise KeyError(f"unknown pack: {pack_id}") from exc

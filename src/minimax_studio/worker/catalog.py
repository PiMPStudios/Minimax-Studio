from __future__ import annotations

from dataclasses import dataclass

from minimax_studio.licenses import H3_TERRITORY


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
    # Packs whose files this pack needs at run time (same local_dir).
    # Deleting a required pack's files would break this one.
    requires: tuple[str, ...] = ()
    # Hugging Face commit SHA. None follows ``main`` (weight packs). A curated
    # LoRA row must pin this — the filename is not an identity.
    revision: str | None = None
    # Marker-file size band after download. Catches a truncated or 4× swap
    # that kept the same name. None skips the check (weight packs).
    min_bytes: int | None = None
    max_bytes: int | None = None
    # A revision says which commit; only a digest says which bytes. HF
    # publishes this: GET /api/models/<repo>/tree/<revision>?blobs=true →
    # lfs.oid. None skips the check (weight packs).
    sha256: str | None = None


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
    "music3-comfy": Pack(
        id="music3-comfy",
        title="MiniMax-Music3 (Comfy INT8)",
        summary="Consumer CUDA song pack. Detected in Studio or a ComfyUI models folder. Generate submits to ComfyUI when that server is running.",
        repo_id="Comfy-Org/MiniMax-Music-3",
        local_dir="music3-comfy",
        approx_gb=12.0,
        license_name="MiniMax-Music3 Community License",
        family="music",
        kind="comfy",
        allow_patterns=(
            "diffusion_models/minimax_music3_dit_int8_convrot.safetensors",
            "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
            "vae/minimax_music3_dav.safetensors",
        ),
        marker_files=(
            "diffusion_models/minimax_music3_dit_int8_convrot.safetensors",
            "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
            "vae/minimax_music3_dav.safetensors",
        ),
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
    "music3-train-encoder": Pack(
        id="music3-train-encoder",
        title="Music 3 Training Encoder (DAV VAE)",
        summary="SimpleTuner's converted DAV audio autoencoder. Only needed to train Music 3 LoRAs, not to generate.",
        repo_id="SimpleTuner/MiniMax-Music-3-Encoder",
        local_dir="music3-train-encoder",
        approx_gb=0.3,
        license_name="MiniMax-Music3 Community License (SimpleTuner conversion)",
        family="music",
        kind="cuda",
        allow_patterns=("audio_vae/*",),
        ignore_patterns=(".gitattributes",),
        marker_files=(
            "audio_vae/config.json",
            "audio_vae/diffusion_pytorch_model.safetensors",
        ),
    ),
    "h3-train": Pack(
        id="h3-train",
        title="H3 Training files (audio VAE + Qwen3-VL)",
        summary=(
            "Official audio VAE and Qwen3-VL-32B text encoder for SimpleTuner "
            "H3 LoRAs (~63 GB). Does not download the 130 GB transformer — "
            "ConvRot INT8 training uses the Comfy DiT pack for that. Same "
            "folder as official diffusers, so a full FL2VA download already "
            "counts as this pack."
        ),
        repo_id="MiniMaxAI/MiniMax-H3",
        local_dir="h3-diffusers",
        approx_gb=63.0,
        license_name="MiniMax H3 Community License",
        family="h3",
        kind="diffusers",
        allow_patterns=(
            "audio_vae/*",
            "text_encoder/*",
            "tokenizer/*",
            "processor/*",
            "scheduler/*",
            "audio_scheduler/*",
            "transformer/*.json",
            "vae/*.json",
            "model_index.json",
            "modular_model_index.json",
            "LICENSE",
        ),
        marker_files=(
            "audio_vae/config.json",
            "audio_vae/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00014.safetensors",
            "tokenizer/tokenizer_config.json",
            "modular_model_index.json",
        ),
        territory_notice=H3_TERRITORY,
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
            "transformer/diffusion_pytorch_model-00001-of-00014.safetensors",
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
        requires=("h3-diffusers-fl2va",),
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
        summary="Omni-reference INT8. Generate via ComfyUI when that server is running. Needs the FL2VA pack’s encoder and VAEs.",
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
        requires=("h3-fl2va",),
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

# PLAN-V3 S2: curated LoRAs, not a live scrape. Kind ``lora`` so Models does
# not list them. They land in models/loras/ next to trained/imported files.
ADAPTERS: dict[str, Pack] = {
    "h3-realism-people": Pack(
        id="h3-realism-people",
        title="H3 Realism People (fal)",
        summary=(
            "Faces, skin, people. Start the prompt with r34l1sm. "
            "Strength 1.0; 0.6–0.8 for a lighter touch."
        ),
        repo_id="fal/MiniMax-H3-Realism-People-LoRA",
        local_dir="loras",
        approx_gb=0.13,
        license_name="MiniMax H3 Community License",
        family="h3",
        kind="lora",
        allow_patterns=("h3-realism-people-t2v-i2v-r2v.safetensors",),
        marker_files=("h3-realism-people-t2v-i2v-r2v.safetensors",),
        territory_notice=H3_TERRITORY,
        # main HEAD 2026-08-12; LoRA bytes last changed b6e6b9f (rank 32 / 1500).
        revision="039cc8579d7aa357a882d7f4111b25da4f72dccc",
        min_bytes=1_000_000,
        max_bytes=250_000_000,
        # 131,229,656 B at that revision (HF lfs.oid + local sha256sum).
        sha256="acc529601d2da117fb81179e76c56e488a3beab1171659d305f04fa3655b787e",
    ),
    "h3-motion": Pack(
        id="h3-motion",
        title="H3 Motion Adapter (MATLOWAI)",
        summary=(
            "Rank-16 motion LoRA for FL2VA and Ref2VA. Reduces frame-to-frame "
            "snap on fast motion."
        ),
        repo_id="MATLOWAI/MiniMax-H3-Motion-Adapter",
        local_dir="loras",
        approx_gb=0.06,
        license_name="MiniMax H3 Community License",
        family="h3",
        kind="lora",
        allow_patterns=("minimax_h3_motion_adapter_pilot_r16.safetensors",),
        marker_files=("minimax_h3_motion_adapter_pilot_r16.safetensors",),
        territory_notice=H3_TERRITORY,
        # main HEAD 2026-08-25; LoRA bytes last changed 7d8c909 (pilot r16).
        revision="0bfe4d6263fe4cc6f36f7682c95d33e50a5f6362",
        min_bytes=1_000_000,
        max_bytes=150_000_000,
        # 63,103,768 B at that revision (HF lfs.oid + local sha256sum).
        sha256="2b31e67d0399eab21ae45fadddcefe82293b1c6cf677b87f8f19bcc745d02fe4",
    ),
}


def pack_or_raise(pack_id: str) -> Pack:
    pack = PACKS.get(pack_id) or ADAPTERS.get(pack_id)
    if pack is None:
        raise KeyError(f"unknown pack: {pack_id}")
    return pack

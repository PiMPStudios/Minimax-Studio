# Changelog

All notable changes to MiniMax Studio live here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/): **MAJOR.MINOR.PATCH**.

- **MAJOR** — breaking changes to the job API, config, or on-disk layout
- **MINOR** — new user-facing features that stay backward compatible
- **PATCH** — fixes and small internal changes

The version string is defined once in `src/minimax_studio/__init__.py` (`__version__`).
`pyproject.toml` reads it from there. The worker `/health` endpoint, window title, and Help page show the same value.

## [0.2.0] — 2026-08-19

First usable generate studio. Local git only; no GitHub release yet.

### Added

- PySide6 desktop shell: sidebar, docked inspector, first-launch output folder
- Separate FastAPI worker process (GUI never imports PyTorch)
- Hardware probe (CUDA / Apple Silicon / RAM)
- Hugging Face model packs: Music 3 CUDA + MLX, official H3 diffusers FL2VA/Ref2VA, Comfy-Org INT8 + Turbo LoRA
- H3 community-license territory warning before those downloads
- Generate Music: structured caption, lyric tags, local CUDA/MLX or stub
- Generate Video: T2VA / I2VA / L2VA / FL2VA / Ref2VA form, multi-file references
- MiniMax hosted APIs: H3 `/v2/video_generation` and Music `/v1/music_generation` (`music-3.0`)
- History: play, restore to Generate, delete
- Presets: save from Generate pages, apply/delete
- Settings: output/models dirs, HF token, MiniMax API, local LLM
- Local OpenAI-compatible LLM (default `127.0.0.1:8080`, `qwen3.8-27b-q4kxl`)
  - Enhance caption / Enhance prompt
  - Write lyrics
  - Medium thinking (`reasoning_effort=medium`, 512-token budget)
  - Key fallback: `~/.config/llama-api.key`
- LoRA import, inspector picker, generate-time `load_lora_weights`
- H3 Fast mode: Turbo LoRA + 8 steps
- Job cancel (cooperative)
- Single-GPU generate lock (one running job at a time)
- Help page with MiniMax H3 / Music 3 license notes
- MiniMax H3 **Context-IR** button (API) to expand a prompt with official multimodal IR
- History video preview widget when Qt Multimedia is present
- SemVer versioning: `__version__` in `minimax_studio/__init__.py` is the only source; `pyproject.toml` reads it

### Fixed

- Local H3 ignored inspector resolution/ratio (always 960×544)
- LoRA strength was not applied after load
- Settings empty fields could not clear tokens
- History timer could crash if the sidebar had no row
- Music decode assumed a torch tensor
- History index rewrite could drop a take that finished mid-delete

### Notes

- In-app local H3 generate uses the **official diffusers** pack, not the Comfy-Org INT8 files (those download for Comfy / a later loader).
- MiniMax’s hosted Music API is being restricted for new accounts; local Music 3 weights are the long-term path.

## [0.1.0] — 2026-08-19

### Added

- Repo, plan (`docs/PLAN.md`), empty-app scaffold, worker `/health` + `/probe`

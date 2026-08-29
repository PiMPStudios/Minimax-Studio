# Changelog

All notable changes to MiniMax Studio live here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/): **MAJOR.MINOR.PATCH**.

- **MAJOR** — breaking changes to the job API, config, or on-disk layout
- **MINOR** — new user-facing features that stay backward compatible
- **PATCH** — fixes and small internal changes

The version string is defined once in `src/minimax_studio/__init__.py` (`__version__`).
`pyproject.toml` reads it from there. The worker `/health` endpoint, window title, and Help page show the same value.

## [0.2.26] — 2026-08-21

### Fixed

- **Background workers no longer vanish before running.** Enhance caption /
  Enhance prompt, Write lyrics, Context-IR, the Settings connection checks,
  and the Inspector “Will use” line all started a `QThread` whose `QObject`
  worker was referenced only by signal connections — CPython freed it before
  `started` was delivered, so the request silently never happened (spinner
  forever). Workers now keep a strong reference and their results are
  marshalled back to the GUI thread.
- **Remove pack is honest.** It reports how many GB it actually freed, and
  when other installed packs share the folder it keeps the files they need
  (Ref2VA needs FL2VA’s encoder/VAE; the official Ref2VA transformer needs
  the FL2VA tree) instead of unlinking markers and leaving 130 GB behind.
  A follow-up dialog offers to wipe the whole shared folder if you really
  want those packs gone too. Removing a pack nothing else uses now deletes
  the folder entirely, as it always should have.
- **UI-thread pressure.** The 500 ms tick made five-plus HTTP round-trips
  (two queue lines, a status bar, two live-job polls); it now makes one
  `/jobs` call shared by all of them. The Models page refreshes every 2 s
  while open, not every 500 ms. Pack disk-size walks are cached for 5 s
  (file existence checks stay live). Inspector preflight (“Will use”) runs
  on a worker thread instead of the UI thread.

### Added

- **Disk guard before downloads.** A pack that would not fit is refused with
  “X GB free, needs about Y GB”, and Models offers Download anyway.
- **CI**: GitHub Actions on Linux/Windows/macOS × Python 3.11–3.13 running
  `ruff check` and `pytest`. `ruff` is in the dev extra with an explicit
  correctness-focused ruleset; unused imports and import order cleaned.
- `docs/PLAN.md` carries an **Amendments** section: the INT8-consumer path
  does drive a user-run ComfyUI (never bundled), 2K is API-relevant only,
  and the worker token gate is documented.

### Changed

- Removed the unused `WorkerClient.iter_job_events` SSE client helper; the
  documented `GET /jobs/{id}/events` endpoint stays for external tooling.

## [0.2.25] — 2026-08-21

### Fixed

- **Cancel is quiet**: cancelling mid-sample now lands the job in `cancelled` instead of `error`, so no “Generate failed — Cancelled” dialog with a Retry button. Backends raise a typed `jobs.CancelledError`.
- **Comfy file visibility**: if INT8/Ref2VA/Turbo files exist on disk but the running ComfyUI does not load the folder they live in, generate and preflight now fail fast and name the missing files, instead of failing inside Comfy after uploading. Files Comfy lists under a subfolder are resolved to their listed names.
- Presets keep everything the Inspector shows: Music **CFG**, **Attention**, and **both LoRA slots** now survive save → apply. The Video preset payload no longer sets `backend` twice.
- Preset save/delete no longer read-modify-write `presets.json` without the runtime lock.

### Added

- **Worker shared secret**: the GUI generates a per-launch token, passes it to the worker via `MINIMAX_STUDIO_WORKER_TOKEN`, and sends it as `X-Minimax-Studio-Token` on every request. Other local users/processes can no longer read API tokens from `GET /settings` or queue jobs. `python -m minimax_studio --worker-only` (no env var) stays open for development.

### Changed

- Asset inputs are validated by type: the H3 API only base64-encodes known image/video/audio extensions, Comfy uploads accept the same, and LoRA import accepts `.safetensors` only. Arbitrary paths are no longer read and encoded.
- Inspector “Will use: …” runs the same Comfy file check, so “comfy” is only offered when ComfyUI can actually load the graph’s files.

## [0.2.24] — 2026-08-20

### Added

- Inspector **LoRA 2** stacks a second adapter after LoRA 1 (and after Turbo when Fast)

## [0.2.23] — 2026-08-20

### Added

- Generate Music length chips: 30s / 60s / 2m / 3m / 5m
- History detail shows mode, duration, seed, steps, speed, ratio
- History meta now stores speed, CFG, ratio, quality, attention, LoRAs so Restore is complete

## [0.2.22] — 2026-08-20

### Added

- Generate Video length chips: **5s / 8s / 15s**
- Generate failure dialog **Retry**
- History **Copy prompt** (includes lyrics for music)
- File → **Open Comfy Log** and **Open ComfyUI in Browser**
- Generate pages show other queued jobs of the same kind

### Changed

- Inspector **Fast** is disabled on Generate Music (H3 Turbo only)

## [0.2.21] — 2026-08-20

### Added

- Inspector **Fast** is honest: needs the H3 Turbo LoRA on disk (preflight blocks otherwise) and snaps steps to 8 (4 for Ref2VA)
- Comfy generate status includes elapsed time and a progress bar that advances while sampling

## [0.2.20] — 2026-08-20

### Fixed

- Comfy H3 `SaveVideo` now sends `codec` as `"auto"` (DynamicCombo). Nested `{"codec": "auto"}` made Comfy drop the input after a full sample
- Comfy generate errors show the node and exception, not the raw message dump
- Switching from Music to Video no longer clamps a 30s song length to a 15s H3 take; duration resets to 8s when it is out of H3’s 5–15s range

## [0.2.19] — 2026-08-20

### Changed

- **Start ComfyUI** waits until `8188` answers (or the process dies / 90s timeout) and surfaces the log tail on failure
- Status/Welcome/Settings no longer claim Comfy is up the instant the process is spawned

## [0.2.18] — 2026-08-20

### Changed

- Generate preflight no longer pings MiniMax/LLM/Comfy on every Inspector refresh (those pings could freeze the window for seconds)
- Hardware probe is cached for 2 seconds
- Comfy reachability uses a 0.6s timeout and is cached for 2 seconds so a down ComfyUI does not stall the Inspector

## [0.2.17] — 2026-08-20

### Added

- View → **Setup checklist…** reopens the welcome GPU/packs/Comfy summary

## [0.2.16] — 2026-08-20

### Added

- Settings shows the detected ComfyUI install (folder, venv python, running or not)

## [0.2.15] — 2026-08-20

### Added

- Generate failure pops a dialog with the worker error (cancel stays quiet)

### Changed

- Progress bar hides when a take finishes or fails

## [0.2.14] — 2026-08-20

### Changed

- Generate stays enabled while a job runs, so another take can queue from the same page
- Presets filter by Video/Music and name

## [0.2.13] — 2026-08-20

### Added

- Settings **Store tokens in the OS keychain** (optional `keyring` extra). Hugging Face, MiniMax, and LLM keys leave `config.json` when a keychain backend is available
- History selects the newest take after a refresh (so a finished generate is ready to play)

## [0.2.12] — 2026-08-20

### Added

- Generate **queue**: a second job waits instead of failing with “already running” (max 8)
- Status bar shows how many jobs are queued behind the live one

### Changed

- Cancel on a queued job drops it immediately; the running job is unchanged

## [0.2.11] — 2026-08-20

### Added

- Status bar shows active model download progress
- Generate confirms when preflight has warnings (missing ffmpeg)
- Models **Show in folder** for a pack on disk
- History rows include backend and duration

## [0.2.10] — 2026-08-20

### Added

- **Start ComfyUI** from Welcome, Settings, and Go menu. Detects `main.py` under `~/ai/ComfyUI`, Settings → ComfyUI folder, or `COMFYUI_PATH`
- Settings: ComfyUI folder + extra args (passed to `main.py`)
- Go menu shortcuts: Ctrl+1…7 for pages, Ctrl+Enter to generate
- History filter (All / Video / Music) and prompt search

## [0.2.9] — 2026-08-20

### Added

- Worker `GET /jobs/{id}/events` Server-Sent Events stream for generate progress
- Status bar live job (kind, status, progress, message) with Cancel
- History **Show in folder**; File menu opens output and models folders
- H3 preflight warns when `ffmpeg` is not on PATH
- Welcome notes whether ffmpeg is available

### Changed

- Help Auto copy matches VRAM-aware routing (INT8/Comfy under 24 GB)

## [0.2.8] — 2026-08-19

### Added

- Inspector **Will use** line from generate preflight (updates every few seconds)
- Welcome **Download recommended** (consumer INT8 packs only)
- Cancel on Models downloads (cooperative; HF may finish the current file)
- Comfy generate status includes queue running/pending counts

### Changed

- Auto H3 prefers Comfy INT8 when the selected GPU has under 24 GB VRAM, even if official diffusers is installed
- Music duration is capped at 5 minutes

## [0.2.7] — 2026-08-19

### Added

- Generate Video: drag-and-drop on asset rows, Preview vs Native 768, 3:4 ratio, Structure shot-list insert
- Models: Remove deletes the Studio copy only (never a ComfyUI folder)
- Downloads pull MiniMax LICENSE/NOTICE into the pack folder when Hugging Face has one
- Inspector CFG for Music 3 (default 1.7)
- Apache-2.0 license for this repo’s code (model weights stay MiniMax’s)

### Changed

- Local MiniMax H3 on Apple Silicon is gated in v1 (API or CUDA machine)

## [0.2.6] — 2026-08-19

### Added

- Generate preflight (`GET /preflight`): warns before a job if ComfyUI/PyTorch/API is not ready
- Inspector and Settings CUDA GPU picker (in-process diffusers). Comfy still uses its launch device
- Auto backend skips official CUDA if PyTorch is not in the Studio venv, so INT8+Comfy is preferred

## [0.2.5] — 2026-08-19

### Fixed

- Hardware probe uses `nvidia-smi` when PyTorch is not installed in the Studio venv, so welcome/inspector still show NVIDIA GPUs
- Inspector notes that in-process diffusers needs PyTorch even if CUDA cards are present

## [0.2.4] — 2026-08-19

### Added

- One-shot welcome after launch: GPU, packs already on disk (including Comfy folders), ComfyUI up/down
- Inspector shows H3 duration snap (5–15 s, 17n+5 frames at 24 fps) on Generate Video
- Probe reports SageAttention, ready pack titles, and how many came from Comfy
- History Export… copies the take; selecting a row loads it in the player
- Generate Video opens History when a take finishes (same as Music)

### Changed

- CUDA recommended downloads: Music 3 INT8 + H3 INT8 + Turbo; official 130 GB H3 only if VRAM is huge
- Models page subtitle counts ready packs

## [0.2.3] — 2026-08-19

### Added

- Comfy-Org INT8 **Ref2VA**: images, videos, and audio references submit to a running ComfyUI
- Comfy-Org **Music 3 INT8** pack detect + generate via a running ComfyUI
- Inspector **Sage** attention (KJNodes `PathchSageAttentionKJ` on the Comfy graph)
- Models page also reads Comfy `extra_model_paths.yaml` when no Comfy models folder is set
- Reference size `match` / `max` on Generate Video
- Settings connection checks run off the UI thread

### Notes

- Name reference files in the prompt as `<Picture 1>` / `<Video 1>` / `<Audio 1>` in add order
- SageAttention is a Comfy-path toggle. Official diffusers still uses PyTorch attention
- If KJNodes is missing, generate retries without the Sage node

## [0.2.2] — 2026-08-19

### Added

- Settings shows inline ✓/✗ for MiniMax, the local LLM, and ComfyUI (no ping dialog)
- Settings: ComfyUI models folder + ComfyUI URL (`http://127.0.0.1:8188`)
- Models page detects Comfy-Org INT8 H3 already on disk (`minimax-h3/`, `h3-comfy/`, or a Comfy `models/` tree)
- Local H3 generate can submit T2VA / I2VA / L2VA / FL2VA to a running ComfyUI when INT8 files are present
- Inspector backend: Comfy

### Changed

- Auto backend: official diffusers in-process, then Comfy INT8, then MiniMax API
- CUDA recommended pack is now Comfy-Org INT8 alongside official diffusers

### Notes

- INT8 files use Comfy convrot kernels. Diffusers cannot load them; Studio will not try `from_single_file` on a 42 GB checkpoint just to fail. Start ComfyUI or download official FL2VA.
- Reference mode on INT8 now goes through ComfyUI when that pack and server are present (0.2.3).

## [0.2.1] — 2026-08-19

### Added

- Resume label on partial Hugging Face pack downloads (`resume_download=True`)
- Presets store/restore assets, LoRA, speed, and backend
- Settings save pings MiniMax and the local LLM (`GET /ping`)
- Local H3/Music sampling checks cancel between steps via `callback_on_step_end`

### Fixed

- Inspector LoRA selection now follows restored presets

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

- In-app local H3 generate uses **official diffusers** in-process when that pack is installed. Comfy-Org INT8 is used when those files are on disk and ComfyUI is running.
- MiniMax’s hosted Music API is being restricted for new accounts; local Music 3 weights are the long-term path.

## [0.1.0] — 2026-08-19

### Added

- Repo, plan (`docs/PLAN.md`), empty-app scaffold, worker `/health` + `/probe`

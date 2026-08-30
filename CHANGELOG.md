# Changelog

All notable changes to MiniMax Studio live here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/): **MAJOR.MINOR.PATCH**.

- **MAJOR** — breaking changes to the job API, config, or on-disk layout
- **MINOR** — new user-facing features that stay backward compatible
- **PATCH** — fixes and small internal changes

The version string is defined once in `src/minimax_studio/__init__.py` (`__version__`).
`pyproject.toml` reads it from there. The worker `/health` endpoint, window title, and Help page show the same value.

## [0.2.33] — 2026-08-29

### Added

- **Storage — the disk half of a long run (PLAN-V2 S5).** A run that goes well
  is tens of gigabytes and SimpleTuner has no reason to tidy up after itself, so
  the Train page gained a **Storage…** dialog that names the numbers *before*
  anything is deleted: checkpoints, caches, the whole run, and the gigabytes free
  on that volume. **Prune checkpoints** keeps the newest N **plus every
  checkpoint you installed as an adapter** — there is no eval score in
  SimpleTuner's stdout, so "best" means *the one you chose to keep*, and the
  dialog says that instead of inventing a metric. The number in that
  confirmation comes from a **dry run** over the same code path that then does
  the deleting, and a pruned step takes its **whole folder** — weights plus
  SimpleTuner's optimiser state — so the gigabytes promised are the gigabytes
  freed. **Clear caches** frees the VAE
  and text-embedding caches (derived data: the next run rebuilds them, slowly).
  **Delete run folder** reports what it frees, and never touches installed
  adapters — those are copies.
- **Resume from checkpoint.** A stopped, cancelled or 4 a.m.-failed run continues
  in its **own** folder: `resume_from_checkpoint` goes into the same config, the
  caches stay warm, the log keeps appending, `resume_count` ticks up. The
  **Resume from checkpoint** button takes the newest; the Storage dialog's
  **Resume from selected** takes whichever checkpoint you highlight. Only a
  `.safetensors` that lives inside that run is accepted — weights from another run
  would train the wrong base and credit the wrong provenance.
- **Export / Import a run.** Export copies state, config, checkpoints and the log
  with an `EXPORT.json` manifest (file count, bytes, whether caches came along)
  and **leaves the caches behind** — they are most of the bytes and the receiving
  machine recomputes them anyway. Import takes that folder back into the list
  with its history, and refuses to merge onto an id that is already here rather
  than mixing two runs' checkpoints.
- **Nothing destructive while a run is live.** Prune, clear, delete and resume
  are refused twice over: the buttons are disabled with the reason in the
  tooltip, and the worker refuses again with the pid inside the sentence. On
  Windows a running trainer holds those files open, so "cleanup" there is not
  freed disk — it is a half-deleted run.
- New worker API: `/train/storage`, `/train/runs/{id}/storage`,
  `…/prune`, `…/cache/clear`, `…/resume`, `…/export`, `POST /train/runs/import`,
  `DELETE /train/runs/{id}`. Storage reports are cached ~15 s and invalidated by
  every deletion — walking a VAE cache means thousands of files, which has no
  business in a 2-second poll.

### Fixed

- **Run-relative paths are POSIX-style in the API** — `checkpoints/step-800/lora.safetensors`,
  not `checkpoints\step-800\lora.safetensors`. The Storage dialog, the prune
  plan and `EXPORT.json` now read the same on every OS, which is the point of an
  export contract. (Windows CI caught this on the first run.)
- The modal-box harness in the tests moved to `tests/dialogs.py` and now stands
  in for `QFileDialog` as well, so the pages that ask for a folder are testable
  without a hung CI runner.

## [0.2.32] — 2026-08-29

### Added

- **Adapters: provenance, and the loop closes (PLAN-V2 S3).** New page
  (**Ctrl+Shift+A**) lists every `.safetensors` the picker can load and says
  where each came from — **trained here**, **imported**, or **found on disk**
  (files Studio never registered but loads anyway; saying so beats a lie by
  omission). A trained row carries what a filename can't: dataset name, clip
  count, a **manifest hash** of those clip names and sizes, run id and name,
  preset, rank, steps, base pack, and the pinned
  `simpletuner 4.8.0` that made it. Registry lives in `<models>/adapters.json`,
  is keyed by file name (the picker's own id), survives being hand-edited or
  corrupted — and is **description, never a gate**: delete it and every LoRA
  still loads.
- **Audition** — one click, one ordinary generate job: 30 s at **0.8** strength
  with the caption the dataset used *most*, badged `audition:<adapter>` on the
  job and in History, where **Restore to Generate** works on it like any take.
  That is the answer to “were those 3 hours worth it?” without setting up
  anything. It refuses with a reason when there is nothing to sing with — a
  hand-imported adapter or a deleted dataset has no caption — and a typed
  prompt is accepted as the honest substitute. H3 adapters wait for S4.
- **Forget** removes the provenance row and leaves the file on disk and in the
  picker; deleting the *file* keeps the row, flagged “file is gone”, because
  the story of an adapter outlives it. Filters: *only what Studio trained*,
  *only missing files*.
- History badges auditions in the list and names the adapter in the detail.
- Endpoints `GET /adapters`, `POST /adapters/{id}/audition`,
  `DELETE /adapters/{id}`; jobs carry `audition` at the top level of `GET
  /jobs/{id}`, `/jobs` and SSE instead of buried in `request`.
- Build shortcuts move to **Ctrl+Shift+D / T / A**; Models, Settings and Help
  keep Ctrl+5/6/7 exactly as before (there is no Ctrl+10 key, and renaming
  users' muscle memory is not a feature).

### Fixed

- **Buttons no longer shadow their own handlers.** `self._audition =
  QPushButton(...)` silently *replaced* the method `def _audition(self)`, so
  the next line handed a widget to `.clicked.connect` — a TypeError during page
  construction. It bit the Datasets, Train **and** Adapters pages (twice:
  once in review, once in CI). Buttons now carry a `_btn` suffix on the Build
  pages and `test_build_pages_name_their_buttons_consistently` enforces it.

### Tests

- 25 new (193 total, still GPU-free). The audition test runs the real loop
  against the stub backend: queue → job → History row → and checks the LoRA
  rode along at 0.8 with the caption the adapter actually saw most.

## [0.2.31] — 2026-08-29

### Added

- **Build pages (PLAN-V2 S2) — the last two releases finally have a screen.**
  Everything S0/S1 built was reachable by HTTP only.
  **Datasets** (Ctrl+5): create a dataset, **import a folder** (it *copies*,
  and the result names how many clips brought a caption), **Add from
  History** (a good generation comes with its caption and lyrics already
  written), Validate, Show in folder, Delete — whose confirmation says what
  survives (your originals and History). The report is one row per clip,
  broken ones on top, with the exact reason:
  `✗ missing caption one.txt; 4.0s is under the 3s floor`.
  **Train LoRA** (Ctrl+6): dataset picker that tells “2 problems” apart from
  “not checked”, VRAM presets read *from the worker* rather than a second copy
  of the table, steps/rank/validation-prompt (so SimpleTuner renders clips you
  can hear), then run rows with step/total, loss, checkpoints and an ETA from
  measured step time, a log tail, **Cancel** (process group; checkpoints
  stay), **Open folder**, and **Install adapter** → the LoRA picker.
  Both pages carry the Experimental badge; Music and CUDA only, as planned.
- **Training and generating now know about each other.** Pressing Start
  re-runs preflight and a full dataset validation and refuses both, with
  numbers, *before* anything is written. While a run is live, generate
  preflight warns (“1 training run is live (Overnight) and wants the whole
  GPU — a generation now can OOM one or stall the other”) instead of letting
  you discover it as a CUDA failure, and the status bar names the live run
  from any page — a detached trainer has no job in the queue to be visible
  through. Warn, not block: cancelling someone’s three-hour run is not a
  side effect of pressing Generate.
- Dataset rows and train runs now carry their on-disk `path`: the Train page
  hands it straight back as `dataset_dir`, and “Open folder” needs it — the
  layout is ours, the path never was.

### Tests

- 25 new (167 total, still no GPU, no SimpleTuner): the Build pages are tested
  mostly as refusals — a modal in a test is a hung CI runner, so every box is
  answered by a stand-in. Plus the live-run warning against a stub trainer
  that actually sleeps.

## [0.2.30] — 2026-08-29

### Changed

- **One Python: 3.12, pinned everywhere (install-contract change).** Nothing
  ever pinned it — `requires-python` said `>=3.11` with no ceiling, `run.sh`
  called bare `python3`, and CI installed `.[dev]` on 3.11/3.12/3.13 but never
  `.[train]`. This box's `python3` is 3.14, so the venv 0.2.28 was written in
  could not install the `[train]` extra it released: `simpletuner==4.8.0`
  declares `Requires-Python >=3.12,<3.14`, and no test looked. Now
  [`.python-version`](../.python-version) is the single source of truth and
  `requires-python`, the CI matrix, `scripts/run.sh` / `run.bat` and a startup
  check in `app.py` all read it. **If your `.venv` is not 3.12, run
  `scripts/run.sh`: it moves the old one to `.venv.pre-<ver>` and rebuilds**
  (a wrong-version venv looks ready and cannot train, which is worse than
  missing). Verified resolution on 3.12: SimpleTuner 4.8.0 + torch 2.13.0 +
  torchvision 0.28.0, no conflicts with our generate pins — S0 step 1's
  "resolvable lockstep" is now actually met, and CI asserts it on every push
  with `pip install --dry-run ".[train]"`.

### Added

- **Startup refuses an off-pin interpreter with the fix in the message**
  ("this interpreter is 3.14 … delete .venv and re-run scripts/run.sh"),
  instead of failing later as an ImportError or a silently untrainable install.
- `tests/test_python_pin.py` keeps the five places that name a Python
  version agreeing: `.python-version` ↔ the startup guard ↔ `requires-python`
  ↔ the CI matrix ↔ both launchers, plus a red test if you run the suite on
  any other interpreter.
- README documents the pin, the reasoning, and the `[train]` extra.

## [0.2.29] — 2026-08-29

### Added

- **Datasets foundation (PLAN-V2 S1 — no UI yet).** `worker/datasets.py` +
  `/datasets*` endpoints: create datasets as plain folders in SimpleTuner's
  native layout (`track.wav` + `track.txt` caption + optional
  `track.lyrics`) plus a thin `dataset.json` manifest; import from any
  folder (**copies, never references** — cleaning the source can't gut the
  dataset) and **from History** (the good generations carry their caption +
  lyrics over for free). The validator probes WAV duration with the stdlib
  `wave` module (real numbers, no ffmpeg, CI-honest), flags missing
  captions, orphaned captions, un-readable audio, and clips outside the
  3–300 s window — and `POST /train/runs` now refuses to start training on
  a managed dataset that doesn't validate, naming the first problem.

## [0.2.28] — 2026-08-29

### Added

- **v2-S0 training scaffolding (no UI yet).** `pip install
  "minimax-studio[train]"` pins SimpleTuner 4.8.0; the Music 3 Training
  Encoder (DAV VAE, ~0.3 GB) is a new Models-page pack; `/train/preflight`
  gates on packs, real free VRAM, active generations, and cache disk with
  named numbers; `/train/runs` launches **detached** SimpleTuner runs that
  survive the app closing — reconnect by pid, cancel the whole process
  group, parse `train.log` for step/loss, and `POST /train/runs/{id}/install`
  drops the trained `.safetensors` straight into the LoRA picker. The GUI
  never imports torch: SimpleTuner is a pinned subprocess, its contract is
  the two JSON files our config writer emits. See docs/PLAN-V2.md.
- The worker probe now reports **free** VRAM (`free_vram_gb`) via
  nvidia-smi — deliberately not `torch.cuda.mem_get_info`, which would
  silently tax ~300 MB of CUDA context per GPU on every probe.

## [0.2.27] — 2026-08-29

### Added

- **2K is honest about needing the API.** The Video page greys out the 2K
  resolution option while an explicitly local backend (Local/Comfy) is
  selected, preflight blocks 2K jobs that resolve to a local backend with a
  recipe to fix it, and the local generate path now refuses outright instead
  of silently rendering 768P under a “2K” label.
- **Inspector “Now” row.** The dock’s Hardware line is your machine spec —
  which stayed true even while an API job was running. The new Now row names
  the active job and the backend actually running it, flags when your GPU is
  idle, and shows where “auto” resolved to.
- **`[Post-Chorus]` and `[Solo]`** structure tags on the Music page (and in
  the Write-lyrics prompt), so song forms can use the sections people
  actually write.

### Changed

- **Music API no longer pretends.** When Music resolves to the MiniMax API,
  the route line says the endpoint takes prompt + lyrics only, and the
  Duration, Seed, Steps and CFG controls say in their tooltips that the API
  ignores them — they only shape local generation.

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

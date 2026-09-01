# MiniMax Studio — Build Plan

Working name: **MiniMax Studio**. Repo: `Minimax-Studio`
(github.com/PiMPStudios/Minimax-Studio, public).

## Status amendments (as built, v0.2.26)

Reality notes against this plan, newest first:

- **INT8 + ComfyUI.** This plan says we will not require Comfy. We still do
  not bundle, embed, or depend on it — but the consumer CUDA default path
  (Comfy-Org INT8 packs) only runs inside a user-owned ComfyUI, so Studio
  detects, launches, and drives one as a separate process over its HTTP API.
  The official diffusers packs remain the no-Comfy in-process path. Since
  0.2.25 Studio verifies via `/object_info` that the running ComfyUI can
  actually see every file the graph needs before submitting.
- **2K is API-only.** The Resolution combo's `2K` entry maps to the API
  backend. Local generate stays on the 768 canvas (H3-Regenerate-2K is not
  in the open drop).
- **Worker token.** GUI launches pass a per-launch shared secret
  (`MINIMAX_STUDIO_WORKER_TOKEN` → `X-Minimax-Studio-Token` header). `--worker-only` runs open for development.
- **Music API backend ignores duration/seed/steps** (MiniMax’s
  `/v1/music_generation` accepts none of them). UI should surface that; do
  not fake it.
- Presets store the full inspector state (CFG, both LoRA slots, attention).

This is the PiMP Audio Studio idea — first-launch downloader, sidebar studio, history, presets, inspector — rebuilt as a **Windows / Linux / macOS** desktop app for MiniMax H3 and MiniMax Music 3. Not a Swift/MLX Mac app. Not ComfyUI.

## Locked decisions

| Decision | Choice |
|---|---|
| Platforms | Windows, Linux, **and Mac, all trying to run local** |
| v1 scope | Generate studio first; datasets + training next |
| Cloud | Local weights first; optional MiniMax hosted API |
| Distribution | App never bundles weights; downloader after install |
| UX | Point-and-click forms. No wires, no node graph |
| Desktop UI | **PySide6 (Qt)** — native split/dock studio, not a webview |
| Reference product | PiMP Audio Studio layout and product flow, not its stack |

Honest Mac note: local H3 on Apple Silicon is real in the community (GGUF in Comfy, MLX ports, Metal `h3.c`) but it is **slow and RAM-hungry**. CUDA NVIDIA is the first-class local path. Mac local ships as a real backend with hard RAM gates, not a fake “it runs on an M1 8 GB.” The optional API is the escape hatch when local will not fit.

## What PiMP got right (copy)

From `../PiMPAudioStudio`:

- First launch: essential models + output folder, then the studio
- Sidebar: Create / Build / Library
- Generate canvas + trailing inspector (duration, seed, guidance, active models)
- History with playback, restore-to-generate, export
- Presets of generation settings
- Settings → Models as a download/activate/remove manager
- Adapters as a picker with strength, not a second graph
- Help/docs inside the app
- No weights in the git repo or the installer

## What PiMP we do not copy

- Swift / SwiftUI / MLX / Xcode / App Store sandbox
- ACE-Step and Stable Audio pipelines
- In-app DSP editor and latent inpaint in v1
- MCP in v1
- RunPod remote worker in v1

Those can come back later. v1 is generate + download + history.

---

## Capability catalog — MiniMax H3

Official open weights: **H3-Base only**, two task families.

### Modes (this is the Generate Video form)

| Mode | Checkpoint | Inputs | Output |
|---|---|---|---|
| Text → video+audio (T2VA) | FL2VA | Prompt | 4–15 s, 24 fps, 32 kHz stereo |
| First-frame → video+audio (I2VA) | FL2VA | Prompt + start image | same |
| Last-frame → video+audio (L2VA) | FL2VA | Prompt + end image | same |
| First+last frame (FL2VA) | FL2VA | Prompt + start + end | same |
| Omni-reference (Ref2VA) | **Ref2VA** (different weights) | Prompt + up to 9 images, 3 videos, 3 audio, 12 files max | same |

Aspect ratios: 21:9, 16:9, 4:3, 1:1, 3:4, 9:16. Native canvas is ~768 px on the short edge. Official 2K is **H3-Regenerate-2K**, not in the open drop.

### Not in the open-weight drop (API-only)

- **H3-Context-IR** — the prompt/context preprocessor MiniMax says is critical for quality
- **H3-Regenerate-2K** — in-context 2K upsample

v1 Generate should still have a “Enhance prompt” toggle. Locally that is our own structured prompt helper (MiniMax’s published prompt guides / skills). On API that can call Context-IR.

### Official local runtimes (CUDA)

- [SGLang](https://docs.sglang.io/) cookbook
- vLLM recipes
- Hugging Face **diffusers** `ModularPipeline`
- ComfyUI native nodes (we will **not** require Comfy as a dependency)

Recommended v1 local CUDA path: **diffusers ModularPipeline** for a single Python worker we control. SGLang later if we need a persistent server for speed.

### Add-ons we should surface as checkboxes / dropdowns, not nodes

| Add-on | What it is | v1? |
|---|---|---|
| **Turbo LoRA** (lightx2v / larryvrh / Comfy-Org 4-step and 8-step) | 20 steps → 4–8, ~5× faster, keep stereo audio | Yes — Quality vs Fast |
| **Pruned INT8 + NVFP4 text encoder** | Comfy-Org pack; ~42 GB working set vs 120+ GB | Yes — default CUDA download |
| **SageAttention / Sol Attention** | Faster attention on NVIDIA | Yes as an Advanced speed toggle |
| Community identity/style LoRAs (e.g. fal realism) | Apply at generate time | Yes — LoRA picker |
| GGUF quants | Lower VRAM, quality tradeoff | v1.1 if CUDA VRAM profiles need it |
| Spectrum / latent upscale / motion-context packs | Comfy ecosystem extras | Not v1 |
| MiniMax partner/API nodes | Hosted Hailuo/H3 | Yes as the API backend |

### H3 LoRA training (not v1)

This **is** real, unlike Music 3:

- **Fizgig** (Win/Linux, 16 GB+): stills, 24 fps clips, voice-only audio, Turbo preview, Comfy-ready `.safetensors`
- **fal** hosted trainers: t2v / i2v / flf2v / ref2va
- Audio heads are often skipped in early trainers (look changes, sound stays base) — Fizgig 4.x claims sound/voice training now

v2 Training should wrap a Python trainer (Fizgig-style or SimpleTuner if H3 lands there), not send people to Comfy.

> **Update (2026-08-29):** SimpleTuner already lands H3 — and Music 3 — as
> first-class training targets. See [PLAN-V2.md](PLAN-V2.md).

---

## Capability catalog — MiniMax Music 3

Official open weights: full song generator, lyrics + music description, up to **5 minutes**, 32 kHz 16-bit stereo WAV.

### What it actually does (open weights)

| Input | Role |
|---|---|
| **Caption / description** | Genre, BPM, key, vocals, arrangement (structured caption recommended) |
| **Lyrics** | Sung words; section tags on their own lines: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`, `[Bridge]`, `[Instrumental]`, `[Solo]`, `[Outro]` |
| Duration / max frames | Upper bound; model may end early |
| Seed | Reproducible when the backend supports it |

### What it does **not** do (yet)

Official Comfy workflow: **no reference audio, no cover, no continuation, no stem split, no inpaint.** Do not fake ACE-Step cover mode in v1.

### Official local runtimes

- SGLang-Omni (`/v1/audio/speech`, lyrics in `input`, description in `instructions`)
- diffusers `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3")` — fits ~24 GB; group offload down toward 8 GB
- ComfyUI native (fp16 DiT or int8 convrot)

v1 CUDA path: **diffusers**, same worker as H3.

### Mac local Music 3 (this is the realistic Mac win)

- [`mlx-community/MiniMax-Music3-mxfp8`](https://huggingface.co/mlx-community/MiniMax-Music3-mxfp8) recommended; MXFP4 is smaller and worse at lyrics
- [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) `mlx_audio.music.generate`
- Alternate: [mikolaj92/minimax-music3-mlx](https://github.com/mikolaj92/minimax-music3-mlx)

v1 Mac Music = **mlx-audio**. Same Generate form as CUDA.

### Music LoRA / training — honest status

**Inference LoRAs:** Hugging Face currently lists **two** adapters on the official tree, both experimental SimpleTuner reggae tests (`bghira/minimax-music-suno-reggae-rank128-v*`). There is no ACE-Step-like LoRA bazaar yet.

**Training:** SimpleTuner documents MiniMax Music 3 LoRA / LyCORIS / full-rank, ComfyUI `lora_format`, 24 GB minimum, 48 GB recommended. The missing official **RVQ encoder** is the fight: community is reverse-engineering one (`Mothersuperior/open-rvq-encoder-minimax-music3-169m-pooled-v4`). bghira says VAE-latent flow training works; ostris says without the encoder you are training on unaligned tokens.

**v1:** LoRA *apply* if a `.safetensors` is present. Do not ship a Music trainer until v2, and even then label it experimental. Point at SimpleTuner under the hood when we do.

---

## Product surface (v1)

Copy PiMP’s shell, two studios instead of Music/SFX.

```
Create
  Generate Video     ← H3 modes as a single form
  Generate Music     ← caption + lyrics
Library
  History
  Presets
Setup
  Models             ← downloader / disk / activate
  Settings           ← output dir, HF token, API key, GPU
```

Inspector (always): duration, seed, guidance/steps, active backend, active models, LoRA stack.

Qt mapping from PiMP’s SwiftUI shell:

| PiMP | Qt |
|---|---|
| `NavigationSplitView` sidebar | `QListWidget` / `QTreeWidget` in a left pane |
| Canvas | `QStackedWidget` pages |
| `.inspector` | `QDockWidget` on the right, checkable |
| First launch | Modal / stacked first-run pages before the main window |
| History playback | `QMediaPlayer` + `QVideoWidget` / audio output |
| Settings file dialogs | `QFileDialog` |

First launch:

1. Pick output directory
2. Detect GPU (CUDA / Apple / none)
3. Offer download packs, not “download 330 GB”
4. Show MiniMax H3 / MiniMax-Music3 names in the chrome (community-license UI credit)

### Generate Video — point and click

One page. A **mode** radio at the top reveals the extra drop-zones.

- Prompt (large). Optional “Structure prompt” button (shot list / camera / audio from MiniMax’s guides)
- Mode: Text | First frame | Last frame | First+last | References
- Drop-zones appear for the mode
- Duration slider (snaps to H3’s 17k+5 / 24 fps grid)
- Aspect + quality (Preview / Native 768 / API 2K if API backend)
- Speed: Quality (base steps) | Fast (Turbo LoRA 4–8)
- LoRA list: Turbo (built-in pack) + user LoRAs with strength
- Generate. Progress. Play video+audio in-app. Save to History

No node editor. Internally we build one job object and hand it to the worker.

### Generate Music — point and click

- Caption: either one box, or three labeled fields (Global / Vocals / Arrangement) that concatenate
- Lyrics: textarea with a section-tag toolbar (`[Verse]` etc.)
- Optional lyric helper (user’s own LLM keys later; not v1-blocking)
- Duration, seed, steps
- LoRA picker (empty-state: “community adapters are rare; import a file”)
- Generate → WAV player → History

### Models

Download packs:

| Pack | Why |
|---|---|
| Music 3 (CUDA) | Official diffusers layout |
| Music 3 (MLX MXFP8) | Mac |
| H3 FL2VA official diffusers | In-app local generate (large) |
| H3 Ref2VA official transformer_ref | Omni-reference |
| H3 FL2VA pruned INT8 (Comfy-Org) | Shared with Comfy; in-app loader later |
| H3 Turbo LoRA | Fast mode |

Never auto-pull both H3 families plus bf16.

Reuse Comfy folder layout **optionally** (`models/diffusion_models`, `text_encoders`, `vae`, `loras`) so people who already downloaded Comfy-Org packs can point Settings at that root.

---

## Architecture

Cross-platform desktop = **PySide6 (Qt) shell + Python worker process**.

```
┌─────────────────────────────────────────┐
│  PySide6 / Qt (Win / Linux / macOS)     │
│  QMainWindow + sidebar + stacked pages  │
│  + docked inspector (PiMP-style)        │
└──────────────┬──────────────────────────┘
               │ localhost HTTP + SSE
┌──────────────▼──────────────────────────┐
│  Python worker (FastAPI, child process) │
│  job queue, downloads, history index    │
│  backends:                              │
│    cuda-h3      diffusers / later SGLang│
│    cuda-music   diffusers               │
│    mlx-music    mlx-audio               │
│    mlx-h3       community MLX / Metal   │
│    api-h3       MiniMax video API       │
│    api-music    MiniMax music API       │
└─────────────────────────────────────────┘
```

The GUI process never imports PyTorch. Generate, download, and probe run in the worker so the window stays responsive. Qt talks to the worker over localhost (health, jobs, SSE progress).

Why Qt and not Swift: Mac is one of three OS targets; CUDA is the real H3 engine. Why not Tauri/React: that is a third toolchain (Rust + Node + a web UI) around a Python job. The chrome we need — sidebar, docked inspector, native file dialogs, dark studio theme — is what Qt is for (Resolve and a lot of VFX tools). Why not Gradio: demo, not a studio. Why not embed Comfy: that *is* the wire UI we are replacing.

One language for app and worker. Worker is still a **separate process** so a 30 GB model load cannot freeze the shell.

### Backend policy

| Machine | Music | H3 |
|---|---|---|
| NVIDIA 8–12 GB | Music 3 offload | Maybe GGUF/preview later; else API |
| NVIDIA 16–24 GB | Music 3 comfortable | FL2VA pruned + Turbo; Ref2VA tight |
| NVIDIA 24 GB+ | Full | FL2VA + Ref2VA, Quality or Fast |
| Apple Silicon 32 GB | MLX Music (MXFP4/8) | Local H3 likely no; API |
| Apple Silicon 64 GB+ | MLX Music MXFP8/BF16 | Experimental local H3 (MLX/GGUF/Metal) or API |
| No GPU | API only | API only |

v1 Mac H3 local is **gated**: detect unified memory, refuse with a clear message, offer API. Do not block Win/Linux CUDA on Mac H3 quality.

### Job schema (sketch)

One JSON job the UI always submits:

```json
{
  "kind": "h3" | "music",
  "backend": "auto" | "local" | "api",
  "mode": "t2va" | "i2va" | "l2va" | "fl2va" | "ref2va" | "ttm",
  "prompt": "...",
  "lyrics": "...",
  "assets": [{ "role": "first_frame", "path": "..." }],
  "duration_s": 8,
  "seed": 7,
  "loras": [{ "id": "turbo-4step", "strength": 1.0 }],
  "speed": "quality" | "fast"
}
```

`auto` = local if the pack is installed and VRAM/RAM fits, else API if a key exists, else a “download this pack” error.

---

## v1 / v2 / v3

### v1 — Generate studio (this plan)

1. Repo skeleton, local git, app name, license for *our* code
2. Python worker: health, GPU probe, download manager (HF), job queue, SSE progress
3. PySide6 shell: first launch, sidebar, docked inspector, Models page
4. Music generate (CUDA diffusers)
5. Music generate (MLX) on Mac
6. H3 FL2VA: text / first / last / both (CUDA)
7. H3 Ref2VA + Turbo LoRA + LoRA picker
8. History + presets + in-app players
9. Optional MiniMax API backends for H3 (and Music if the public API is usable)
10. Attribution chrome, license copy shipped next to downloads, NOTICE for H3 redistributes

Out of v1: training, datasets, editor, MCP, RunPod, prompt LLM, 2K local.

### v2 — Training (PiMP “Build”)

**Closed.** Plan: [PLAN-V2.md](PLAN-V2.md) — slices S0–S5 shipped (metal
0.2.37–0.2.40, caption validator 0.2.42). Summary bullets kept below for
the history book.

- Datasets from local files (music: wav + `.txt` caption + `.lyrics`; video: Fizgig/Gizmo-style clip spec)
- Music LoRA via SimpleTuner wrapper, experimental badge
- H3 LoRA via Fizgig-class trainer or a thin wrapper, stills first, clips later
- Adapter registry, strength, audition into History (the PiMP loop)

### v3 — Studio extras (the take after it lands)

**Closed.** Plan: [PLAN-V3.md](PLAN-V3.md). History trim and curated adapter
catalog shipped (0.2.44–0.2.45). Repeat-generate warm worker skipped
(in-session pipe cache is enough).

- Light audio/video trim in History (new take; original kept)
- Curated adapter catalog (download + import; no sharing)
- Faster repeat generate — skipped; SGLang after v3

---

## Implementation order (v1)

Do these as local commits, not GitHub PRs yet.

1. **Foundation** — git, README, this plan, Python package `minimax_studio`, PySide6 shell that can ping the worker `/health`
2. **Model manager** — probe disk/GPU, download one small test file, then Music 3 pack with progress
3. **Music CUDA generate** — caption + lyrics → WAV → History. Proves the whole loop
4. **Shell polish** — sidebar, inspector, first launch, output dir (the PiMP feel)
5. **Music MLX** — same form, Mac backend
6. **H3 FL2VA local CUDA** — modes as form fields, mp4+audio in History
7. **H3 Fast mode** — Turbo LoRA pack + steps preset
8. **H3 Ref2VA** — extra drop-zones, second checkpoint
9. **API backends** — same forms, MiniMax keys in OS secret store
10. **LoRA picker** — import `.safetensors`, stack, strength
11. **Presets + license/help**

Music CUDA generate is the first “it works” milestone. H3 is larger and meaner; do not start there.

---

## Stack versions (starting point)

- Python **3.12, pinned** (`.python-version` is the source of truth; CI,
  `requires-python`, `scripts/run.*` and a startup check all read it). Not
  "3.11 or 3.12": the v2 trainer's `simpletuner==4.8.0` ships nothing outside
  `>=3.12,<3.14`, so one version is tested and shipped rather than three
  promised
- uv for the worker venv
- PyTorch CUDA wheels on Win/Linux; MLX on Mac
- diffusers (Music 3 + H3 ModularPipeline)
- huggingface_hub for downloads
- FastAPI + SSE
- PySide6 (Qt 6) for the desktop shell — Fusion style, dark studio palette
- ffmpeg in PATH (or a sidecar binary) for mux/probe

Windows: `run.bat` creates `.venv`, installs CUDA torch from the official index when needed, starts the worker, then the Qt app. Linux: `run.sh`. Mac: `run.sh` with MLX extras. The launcher owns process lifetime: quitting the window stops the worker.

---

## License / product hygiene

- Our code: pick a license at first code commit (Apache-2.0 matches a lot of this ecosystem; confirm later)
- Ship MiniMax LICENSE files next to downloaded packs
- UI strings: **MiniMax H3** and **MiniMax-Music3** visible on Generate
- Do not geo-block the downloader (user’s call). Do show the H3 territory text on first H3 download
- Do not train other models on H3 outputs (H3 license V.3) — keep that out of any future “distill” feature

---

## Risks

| Risk | Mitigation |
|---|---|
| H3 is huge and moving (pruned, turbo, quants) | Pin Comfy-Org filenames as the CUDA default pack; swap later |
| Mac H3 local is a science project | Gate on RAM; API fallback; do not block CUDA v1 |
| Music LoRA ecosystem is empty | Apply-if-present; don’t market a LoRA store |
| Music training quality is disputed | v2, experimental |
| diffusers H3/Music APIs still landing | Worker interface stays stable; backend adapters can change |
| Optional API vs local output mismatch | Same job schema; label History with backend |

---

## Next step after this plan

Scaffold the repo (worker + PySide6 shell + Models stub) and land **Music 3 CUDA generate** as the first vertical slice. Training stays on the v2 list.

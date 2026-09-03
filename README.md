# MiniMax Studio

A point-and-click desktop studio for **MiniMax H3** (video + stereo audio) and **MiniMax Music 3** (full songs). Windows, Linux, and macOS. No ComfyUI node graph.

**PySide6 (Qt)** shell, Python worker process for downloads and inference. Weights are **not** shipped in the app. A first-launch downloader pulls what you need.

Status: **0.2.63** generate studio + Build pages (datasets, LoRA training —
experimental; Music and H3 24 GB smokes have run on a real NVIDIA GPU).
Changelog: [`CHANGELOG.md`](CHANGELOG.md). Plan: [`docs/PLAN.md`](docs/PLAN.md)
([v2](docs/PLAN-V2.md), [v3](docs/PLAN-V3.md)).

## Locked

- Local inference first, optional MiniMax hosted API
- Generate studio plus experimental LoRA training (CUDA). Mac stays generate-only
- Same Qt UI on Windows, Linux, and Mac — backends differ by GPU
- GUI never imports PyTorch; the worker does

## Run (dev)

```bash
scripts/run.sh          # Windows: scripts\run.bat
```

That finds Python 3.12, builds `.venv` with it, installs the app, and starts.
If `.venv` was built with any other Python it is moved aside
(`.venv.pre-3.14/`) and rebuilt — never quietly reused.

**Python 3.12 only, everywhere.** `.python-version` is the source of truth;
`requires-python`, the CI matrix, both launchers, and a startup check in
`app.py` all read the same pin. The reason is the v2 trainer:
`simpletuner==4.8.0` ships no wheels outside `>=3.12,<3.14`, and a newer
interpreter gives you a `.venv` that installs happily and then cannot train.

By hand, if you prefer:

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m minimax_studio
```

The app starts the worker, opens the studio window, and stops the worker on quit.

The GUI hands the worker a fresh shared secret at launch
(`MINIMAX_STUDIO_WORKER_TOKEN`, sent as `X-Minimax-Studio-Token`) so other
local processes cannot call it. A worker you start yourself with
`python -m minimax_studio --worker-only` has no token and stays open for
development.

Comfy-Org INT8 packs generate through a **running ComfyUI** (Studio does not embed it). Use **Start ComfyUI** on Welcome, Settings, or Go, or launch Comfy yourself. Extra args in Settings are passed to `main.py`, for example `--listen 0.0.0.0 --default-device 1`.

First launch asks for an output folder. Models and history live there.

Local Music 3 generate also needs PyTorch + diffusers (CUDA) or mlx-audio (Mac).
`scripts/run.sh` installs the `[mlx]` extra on Apple Silicon.

```bash
pip install torch diffusers accelerate   # NVIDIA
pip install -e ".[mlx]"                  # Apple Silicon (mlx-audio >= 0.5.0)
```

Optional OS keychain for API tokens:

```bash
pip install 'minimax-studio[secrets]'
```

Then enable **Store tokens in the OS keychain** in Settings.

LoRA training is **Experimental** and CUDA-only (Mac stays generate-only).
`scripts/run.sh` does not install the trainer. Add the pinned SimpleTuner
4.8.0 extra in the same 3.12 venv, then **restart Studio** so the worker can
see `simpletuner` on PATH:

```bash
pip install -e ".[train]"      # resolves on 3.12 only; CI asserts this
```

Then download the **Music 3 Training Encoder** pack on Models. Video LoRAs
need **H3 Training files (audio VAE + Qwen3-VL)** (~63 GB — not the 130 GB
official transformer) plus the Comfy **H3 FL2VA INT8** pack for the ConvRot
DiT. The 80 GB tier still wants the full official FL2VA tree. **Datasets** (Ctrl+Shift+D) takes a
folder of `track.wav` + `track.txt` (+ optional `track.lyrics`) or pulls good
generations out of History; an H3 dataset takes `shot.png` stills and short
`shot.mp4` clips instead — capped at 8 seconds, with `av` (audio+video) mode as
an opt-in checkbox because it costs VRAM and disk. **Validate** refuses extra
lines in a caption `.txt` — one caption per file; extra lines miss the
text-embed cache. **Train LoRA**
(Ctrl+Shift+T) checks VRAM, packs, active jobs and cache disk by name, then
launches the trainer as **its own process** — closing Studio does not stop a
run, and the page reattaches to it. The VRAM presets change with the dataset you
pick: Music 3 has 24/48 GB tiers, H3 has 24 GB (RamTorch CPU-offload), 32, 48
and 80 GB. **Storage…** on that page names the
bytes before anything goes: prune old checkpoints (whatever you installed as an
adapter is always kept), clear the VAE/text caches, **resume** a stopped run
from any checkpoint it wrote, or **export** the run to another disk — caches
excluded, since the next run rebuilds them. **Adapters** (Ctrl+Shift+A) records where
every LoRA came from and **auditions** it at 0.8 strength with the caption it
trained on, badged in History: Music a 30 s song, H3 a short still-pair (or
text-to-video if the dataset has no stills). That path is ComfyUI (the CUDA
pipeline cannot load LoRAs). 24 GB VRAM floor. See
[`docs/PLAN-V2.md`](docs/PLAN-V2.md).

## License (models)

- H3 weights: MiniMax H3 Community License (territory limits; US/EU/UK/Korea need a separate grant or the API)
- Music 3 weights: MiniMax-Music3 Community License (no geo carve-out; UI must show “MiniMax-Music3” on commercial products)

This repo is Apache-2.0. Model weights stay MiniMax’s.

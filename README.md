# MiniMax Studio

A point-and-click desktop studio for **MiniMax H3** (video + stereo audio) and **MiniMax Music 3** (full songs). Windows, Linux, and macOS. No ComfyUI node graph.

**PySide6 (Qt)** shell, Python worker process for downloads and inference. Weights are **not** shipped in the app. A first-launch downloader pulls what you need.

Status: **0.2.5** generate studio. Changelog: [`CHANGELOG.md`](CHANGELOG.md). Plan: [`docs/PLAN.md`](docs/PLAN.md).

## Locked for v1

- Local inference first, optional MiniMax hosted API
- Generate studio now; datasets + LoRA training next
- Same Qt UI on Windows, Linux, and Mac — backends differ by GPU
- GUI never imports PyTorch; the worker does

## Run (dev)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
python -m minimax_studio
```

The app starts the worker, opens the studio window, and stops the worker on quit.

First launch asks for an output folder. Models and history live there.

Local Music 3 generate also needs PyTorch + diffusers (CUDA) or mlx-audio (Mac):

```bash
pip install torch diffusers accelerate
```

## License (models)

- H3 weights: MiniMax H3 Community License (territory limits; US/EU/UK/Korea need a separate grant or the API)
- Music 3 weights: MiniMax-Music3 Community License (no geo carve-out; UI must show “MiniMax-Music3” on commercial products)

This repo will be our own code license (TBD). Model weights stay MiniMax’s.

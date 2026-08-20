# MiniMax Studio

A point-and-click desktop studio for **MiniMax H3** (video + stereo audio) and **MiniMax Music 3** (full songs). Windows, Linux, and macOS. No ComfyUI node graph.

Weights are **not** shipped in the app. A first-launch downloader pulls what you need.

Status: planning. See [`docs/PLAN.md`](docs/PLAN.md).

## Locked for v1

- Local inference first, optional MiniMax hosted API
- Generate studio now; datasets + LoRA training next
- Same UI on Windows, Linux, and Mac — backends differ by GPU

## License (models)

- H3 weights: MiniMax H3 Community License (territory limits; US/EU/UK/Korea need a separate grant or the API)
- Music 3 weights: MiniMax-Music3 Community License (no geo carve-out; UI must show “MiniMax-Music3” on commercial products)

This repo will be our own code license (TBD). Model weights stay MiniMax’s.

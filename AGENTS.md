# AGENTS.md — MiniMax Studio

Point-and-click desktop studio for **MiniMax H3** (video + stereo audio) and
**MiniMax Music 3** (full songs). PySide6 (Qt) GUI + a separate Python worker
process for downloads and inference. Windows / Linux / macOS. No node graph.

Read [`docs/PLAN.md`](docs/PLAN.md) (v1, as-built), [`docs/PLAN-V2.md`](docs/PLAN-V2.md)
(Build/LoRA), [`docs/PLAN-V3.md`](docs/PLAN-V3.md) (library extras) before making
architectural changes. The plans carry "locked decisions" tables — respect them.
[`docs/CODE-REVIEW-ISSUES.md`](docs/CODE-REVIEW-ISSUES.md) is the running review
log (numbered issues, fixed/shipped status) — read it before touching the
dataset / train / adapter code, and add findings there rather than in chat.

## Commands

```bash
scripts/run.sh                      # build .venv on the pinned Python + launch (Windows: scripts\run.bat)
.venv/bin/ruff check src tests      # lint (CI runs exactly this)
.venv/bin/python -m pytest -v --tb=short -rf      # full suite (Qt runs offscreen, no GPU)
.venv/bin/python -m pytest tests/test_routes.py -q               # one file
.venv/bin/python -m pytest tests/test_datasets.py::test_x        # one test
.venv/bin/python -m minimax_studio --worker-only # worker alone, no GUI, open (no token) — dev only
pip install -e ".[train]"           # SimpleTuner extra; 3.12 only, restart Studio after
```

There is no formatter step and no type-check step in CI — only `ruff check` and
`pytest`. Don't invent a `black`/`mypy` gate.

## Non-negotiable

- **Python 3.12, exactly.** `.python-version` is the source of truth; it is
  mirrored in `pyproject requires-python`, the CI matrix, both launchers,
  `app.SUPPORTED_PYTHON`, and `tests/test_python_pin.py`. `simpletuner==4.8.0`
  ships no wheels outside `>=3.12,<3.14`. Never "just try" a newer interpreter,
  and never add a second version to the CI matrix.
- **The GUI never imports PyTorch / diffusers / mlx / simpletuner.** Those live
  in `worker/` and are imported lazily *inside job threads* (see
  `worker/backends/*.py`). Anything heavy belongs behind such a lazy import, or
  in a subprocess (`train_runs.py`, `trim.py`, `comfy_launch.py`).
- **Never commit weights or user media.** `models/ outputs/ datasets/ cache/
  *.safetensors *.wav *.mp4 …` are gitignored for a reason.
- **The worker is localhost-but-open unless tokened.** The GUI passes a per-launch
  secret (`MINIMAX_STUDIO_WORKER_TOKEN` → `X-Minimax-Studio-Token`), enforced by
  the middleware in `worker/server.py`. Don't add routes that bypass it, and
  don't log the token.
- **ComfyUI is not embedded.** Studio detects, launches, and drives a
  user-owned ComfyUI over HTTP. Keep it that way.

## Architecture

```
app.py  ── spawns ──▶  worker: uvicorn on 127.0.0.1 (worker/server.py)
  ▲                              │
  │ Qt GUI (ui/main_window.py,   │  the worker owns ALL job/model/disk state;
  │ ui/pages/*) via              │  the GUI asks: POST to act, GET-poll or SSE
  └── worker_client.WorkerClient ◀┘  to read (httpx + X-Minimax-Studio-Token)

ui/state.py StudioState = Qt signals sharing *inspector* state (backend, seed,
CFG, LoRA slots…) between Music / Video / History / Presets pages. UI-only.
```

- `worker/runtime.py` — module-level `runtime` singleton: config, `jobs`,
  `downloads`, warm pipes (`music_pipe`, `h3_pipe`), Comfy proc. Global state by
  design; go through it, don't create parallel globals.
- `worker/server.py` — all HTTP routes. Keep the JSON shape returned by
  `/health`, `/probe`, `/jobs*`, `/settings`, `/packs` stable: the GUI reads
  those keys directly, and a rename is a breaking MAJOR.
- `worker/jobs.py` — job lifecycle in a thread. Statuses:
  `queued | running | cancelling → done | error | cancelled`. `MAX_QUEUE = 8`,
  progress + messages stream over SSE at `/jobs/{id}/events`. Cancels raise
  `CancelledError` inside the job so the row lands in `cancelled`, not `error`.
- `worker/backends/` — `stub | api | comfy | cuda | mlx` per family.
  `resolve_*_backend()` decides from installed packs, hardware probe, and
  Comfy reachability; `MINIMAX_STUDIO_STUB=1` forces `stub`.
- `worker/catalog.py` — `Pack` dataclass registry (repo, license, markers,
  territory notice, `requires`). Model downloads, disk guards, and the LoRA
  adapter catalog hang off this shape — extend it rather than hardcoding repos.
- On-disk truth lives under the user's output dir: models, History index,
  presets, datasets, train runs. `worker/history.py` rebuilds the index from
  directories if it is missing — keep History entries immutable (trim writes a
  **new** row with `trimmed_from`, it never mutates the original take).

## Tests

- Use the `studio_home` fixture (`tests/conftest.py`): points
  `MINIMAX_STUDIO_CONFIG` at a tmp dir and resets `runtime` — jobs, downloads,
  pipes, probe cache. Any test touching the worker needs it.
- **CI runs stub backends only. No model, GPU, ComfyUI, or network runs.** Tests
  pass `{"backend": "stub"}` or set `MINIMAX_STUDIO_STUB`. Don't write a test
  that needs real weights.
- Qt tests run `QT_QPA_PLATFORM=offscreen` (conftest sets it). For modals use
  `tests/dialogs.py` (`Dialogs`, `MessageBoxShim`) — it swaps the name inside the
  page module, because Shiboken classes reject attributes. A real modal in a
  test is a hung CI runner.
- `tests/dialogs.wait_background()` joins the QThreads that
  `ui/enhance.start_background()` spawns. `_wait_for_job()` in test_routes polls
  on wall-clock, not a fixed poll count (fixed counts lose on loaded Windows
  CI). Prefer wall-clock deadlines everywhere.
- GUI tests drive pages with `FakeWorker` (`tests/test_main_window.py`) — a
  duck-typed `WorkerClient` double. Real subprocesses appear only where the
  subprocess *is* the subject (`test_train_runs.py`, `test_comfy_launch.py`).

## Conventions

- Ruff: line length 100, `target-version = py312`, select `E4 E7 E9 F I UP035`.
  Style rules (BLE/S/TRY/SIM) are deliberately off — broad `except` at the
  worker boundary is intentional. Don't enable them; don't churn-fix for style.
- Comments explain **why**, with the incident attached ("this is why 0.2.28
  could not install `[train]`"). Match the surrounding density — this codebase
  is heavily commented on purpose.
- Errors are named sentences a user can act on ("ffmpeg is missing — install
  it", plus the fix path). No bare codes, no silent fallbacks, no fake UI
  affordances (e.g. Music API ignores duration/seed/steps — don't render them).
- Binary/tool seams are env-var overrides resolved by a function
  (`MINIMAX_STUDIO_FFMPEG_BIN`, `…_FFPROBE_BIN`, `…_SIMPLETUNER_BIN`) so tests
  can inject a stub. New external tool → same pattern.
- Version string exists once: `src/minimax_studio/__init__.py :: __version__`
  (pyproject reads it). `/health`, window title, and Help page mirror it.

## Release flow

Per release: bump `__version__`, add a `## [x.y.z] — YYYY-MM-DD` block to
`CHANGELOG.md` (Keep a Changelog + SemVer; MAJOR = breaking job API / config /
on-disk layout), and commit as `Release x.y.z: <one-line theme>`.
Other commit prefixes in history: `Docs:`, `CI:`, `Housekeeping:`, `S<n(a|b)> (k/n):`
for plan-slice steps, `wip:` for red tests. Squash-ish, imperative, short subject.

## Gotchas

- Job work is threaded and History/Train state is process-global — tests that
  forget `runtime.reload_config()` / cache resets flake in ways unrelated to
  their own changes.
- Train runs are **detached on purpose**: closing Studio must not stop a run,
  and `train_page` reattaches. Don't "clean up" that lifecycle.
- H3 territory notice (`H3_TERRITORY`) and Music 3 credit (`MUSIC_CREDIT`)
  live in `licenses.py` — copy them, don't paraphrase. Help quotes both.
- `src/minimax_studio/secrets.py` is our keyring helper; `from secrets import token_hex`
  is stdlib. `app.py` imports it as `# stdlib secrets` for that reason.
- `python -m minimax_studio --worker-only` has **no token**. Never ship it as a
  user-facing mode; it exists so you can curl the API while developing.

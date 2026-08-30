# MiniMax Studio — v2 Plan: Build (datasets + LoRA training)

Expands the v2 bullets in [PLAN.md](PLAN.md). Engine research below was done
2026-08-29 against SimpleTuner `main` and the Hugging Face model cards — the
v1 plan's two biggest v2 unknowns are already answered, and one of them
changed the shape of this plan.

---

## Research findings (done — these replace the v1 hopes)

**SimpleTuner treats both of our models as first-class citizens.**

| Fact | MiniMax Music 3 | MiniMax H3 |
|---|---|---|
| SimpleTuner model family | `minimaxmusic` (`model_flavour: music3`) | `minimaxh3` (FL2VA-style first/last conditioning) |
| Trainable today | LoRA, LyCORIS, full-rank transformer | LoRA (PEFT), incl. **ConvRot INT8 flavours** — the same quantised files our Comfy-Org packs ship |
| VRAM floor | **24 GB** conservative LoRA (`int8-quanto` + gradient checkpointing); 48 GB recommended | **24 GB** with the RamTorch preset (`minimaxh3-fl2va-convrot-int8-24g.peft-lora+ramtorch`); presets up to 80 GB |
| Dataset format | Local audio backend: **wav + textfile captions** (`caption_strategy: "textfile"`, duration bucketing) — v1 PLAN's guessed format is SimpleTuner's native one | Local video backend; `minimax_h3_target_mode: video / av / audio`; audio-only via `dataset_type: "audio"` |
| LoRA output | `lora_format: "comfyui"` export — **loads directly in our LoRA picker and Comfy path** | PEFT adapters; community `.safetensors` in the HF ecosystem already |
| Validation | Built-in: `validation_prompt`, `validation_lyrics`, `validation_audio_duration` render audio during the run | Validation samples incl. joint A/V; Drift-Distillation/CREPA regularization keeps the base model's behaviour |
| Extra weights needed | DAV encoder for VAE caching: `dav.pth` (in the upstream repo) or `SimpleTuner/MiniMax-Music-3-Encoder` — a **new small pack in our catalog** | None beyond the packs we already download (verify ConvRot file match in S0) |
| Ready-made examples | `simpletuner train example=minimaxmusic-music3.peft-lora` | `simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora` (+ VRAM tiers) |

What the v1 plan hedged on, resolved:

- *"Music LoRA via SimpleTuner wrapper, experimental badge"* → concrete:
  `minimaxmusic` exists; the wrapper is config generation + process
  management, nothing exotic.
- *"SimpleTuner if H3 lands there"* → **H3 is already in there**, with the
  INT8 story matching our Comfy packs. "Fizgig-style or a thin wrapper" from
  PLAN.md becomes: **wrap SimpleTuner; Fizgig-class is what we are describing.**
- *"Do not ship a Music trainer until v2, and even then label it
  experimental"* → kept. Quality disputes are a product-truth problem, not a
  SimpleTuner-support problem. Badge stays.

**License landmines found (gate on these):**

- H3 Community License (as surfaced in SimpleTuner's own matrix): territory
  conditions — **authorization required in US/EU/UK/KR** — and PLAN.md's
  existing note: *"do not train other models on H3 outputs (H3 license V.3)."*
  Training our own H3 LoRAs locally is the same risk class as downloading
  H3 weights (we already show the territory text) — but any future
  adapter-**sharing** feature needs a legal pass first. Not in v2 scope.
- Music 3 carries its own license (CC-family badge on the model card) —
  ship the license file next to the encoder pack like every other pack.
- SimpleTuner is MIT with copyleft-adjacent *targets* (e.g. Hunyuan rows) —
  we only run it; we never vendor its code (locked below).

---

## Locked v2 decisions

| Decision | Choice |
|---|---|
| Engine | **Wrap SimpleTuner** (`pip install simpletuner`, version-pinned). No vendoring, no fork, no custom training loop in this repo |
| Datasets | Store in **SimpleTuner-native layout** under `~/MiniMaxStudio/datasets/<name>/` + a thin `dataset.json` manifest of ours. No proprietary format to migrate later |
| Run model | Training is a **detached subprocess**, not a worker job — it must survive app restarts and run for hours; Studio reconnects, never hosts |
| GPU etiquette | Training refuses to start while a generation is in flight or the VRAM floor of the chosen preset isn't free |
| Adapters out | PEFT `.safetensors`, `lora_format: "comfyui"` → straight into the existing LoRA picker; registry records provenance |
| Hardware | **CUDA (Windows/Linux) only for training.** Mac stays generate-only (MLX + API) — the Quickstarts say so themselves |
| Badges | Music training = **Experimental**. H3 training = "community trainer," honest VRAM/disk floors in the UI |
| Cancellation | SIGTERM the process group; a run is resumable from the last checkpoint, always |

---

## Slices, in order (vertical-slice discipline, same as v1)

### S0 — Engine spike (the gate for everything)

> **Status (0.2.28): off-GPU half landed.** `pip install "minimax-studio[train]"`
> (pins SimpleTuner 4.8.0), the DAV encoder pack on the Models page, the
> two-config writer (`worker/train_config.py` — `minimaxmusic`/`music3`, rank +
> precision presets, `lora_format: comfyui`, all writes confined to the run
> dir), train preflight with named numbers (packs, free VRAM via nvidia-smi,
> GPU-sharing block, cache-disk guard), the detached runner
> (`worker/train_runs.py` — survives worker restarts, group-cancel, log→progress,
> `install` → LoRA picker), and `/train/*` endpoints — 18 tests against a stub
> trainer, CI-green on all three OSes. **Remaining: steps 3–5 on real metal**
> (recalibrate the log regexes and the `train env=` arg shape against real
> SimpleTuner output the first time it runs — they were written from its docs,
> not its stdout).
>
> **Step 1 is now genuinely closed (0.2.30) — and it had not been before.**
> `simpletuner==4.8.0` declares `Requires-Python >=3.12,<3.14`; the dev venv was
> 3.14 (the machine's `python3` default), so the extra resolved **nowhere** and
> CI — which tested 3.11/3.12/3.13 but only ever installed `.[dev]` — could not
> see it. Fix: one pinned Python (3.12) across `.python-version`,
> `requires-python`, the CI matrix, both launchers, and a startup check; CI now
> also runs `pip install --dry-run ".[train]"` so the torch/SimpleTuner lockstep
> is asserted on every push. Resolution verified on 3.12: `simpletuner 4.8.0 +
> torch 2.13.0 + torchvision 0.28.0`, no conflicts. **The metal run in steps
> 3–5 must therefore happen in a 3.12 venv** — `scripts/run.sh` rebuilds `.venv`
> on 3.12 (and moves a wrong-version one aside) if you are not already there.

Nothing ships until this passes on real metal:

1. Pin a SimpleTuner version; add `[train]` extra; resolve torch/torchvision
   pin conflicts with our generate pins (**exit criterion: one resolvable
   lockstep** — ✅ done in 0.2.30, on Python 3.12 only, asserted in CI).
2. Add the Music 3 encoder pack (`dav.pth` / `MiniMax-Music-3-Encoder`) to
   the catalog with license file + disk-space guard like every pack.
3. Smallest possible Music LoRA run: ~5 clips × 15 s, ~200 steps, 24 GB
   preset, config generated *by our code*, launched *by our code*.
4. The produced `.safetensors` appears in the Studio LoRA picker and an
   audition render goes through History.
5. Same smoke for H3: ConvRot INT8 files from our existing Comfy pack
   accepted by the 24G RamTorch preset (if not, record the delta — a separate
   "H3 training files" pack may be needed).

**Exit = demo video of 3→4 + a written config contract.** If 4 fails
(adapter format drift), the whole plan re-thinks; if 1 fails, the extra
becomes a managed separate venv (decision deferred to S0, both paths priced).

### S1 — Datasets foundation (fully off-GPU; CI-coverable)

> **Status (0.2.29): landed (backend).** `worker/datasets.py` + `/datasets*`
> endpoints: manifest + native folder layout, import-from-folder (copies),
> import-from-History (caption/lyrics carried), stdlib-`wave` validator with
> named per-clip problems, and `train_runs` refusing to launch on a managed
> dataset that doesn't validate clean. **Dataset UI landed in 0.2.31** — the
> `Datasets` page (list/create/import/from-History/validate/report), built on
> the API surface that has not needed a change since (one addition: rows carry
> `path`, because the Train page must hand `dataset_dir` back to the worker).

- `datasets.py`: `dataset.json` manifest (kind music/video, created_at,
  clip list, validation report), import from folder **and from History**
  (select generations → "add to dataset" copies wavs/clips + writes caption
  `.txt` from prompt/lyrics already stored on the history entry)
- Validators (music: wav decodes via ffmpeg probe, caption file present,
  duration in [3, 300] s; video: resolution/fps probe, caption present) —
  the Dataset page lists per-clip ✓/✗ with the reason, blocks training start
  until clean
- Endpoints: `GET/POST /datasets`, `/datasets/{id}/validate`,
  `/datasets/{id}/entries` (History picker reuses the History page UI)
- Tests: synthetic 1-second wavs written by stdlib + real ffmpeg probe —
  no GPU, no models, CI-safe

### S2 — Music LoRA slice (v2's "Music CUDA generate" moment)

> **Status (0.2.31): UI landed; the metal is still the gate.** `Datasets` and
> `Train LoRA` pages exist: dataset picker that distinguishes "2 problems"
> from "never checked", VRAM presets read from `/train/preflight` (no second
> copy of the table to drift), steps/rank/validation prompt, run rows with
> step/loss/checkpoints/step-time ETA, log tail, Cancel, Open folder,
> **Install adapter** → LoRA picker, Experimental badge on both. Start
> re-checks preflight *and* validates the dataset before writing anything, and
> generate preflight now warns when a detached run holds the GPU (plus a
> status-bar line, since training has no job in the queue). 25 off-GPU tests.
> **What S2 still cannot prove:** that `STEP_RE`/`LOSS_RE` and
> `simpletuner train env=<id>` match real output — that is S0 steps 3–5 on a
> 24 GB card; the run rows here run against a stub and a fake worker.

- `TrainRunner`: writes `config.json` + `multidatabackend.json` into
  `runs/<id>/`, launches `simpletuner train` detached (own process group,
  `pid`, `train.log`, `state.json` in the run dir), reconnect-on-launch
  scan of `runs/` for live runs
- Log parser → progress (step/total, loss, checkpoint path) surfaced in the
  Build page run list + status bar; ETA from step time
- Cancel → SIGTERM group, resumable; app restart → run keeps going, Studio
  reattaches (this is the pattern diff between our job queue and training)
- Build page: dataset picker → VRAM-bucket preset (from `probe()` free VRAM)
  → steps/rank/output name → Start; run row: live log tail, Cancel, Open
  folder, and on completion **"Install adapter"** → registry → picker
- Training preflight (`/train/preflight`): train extra installed, encoder
  pack present + disk guard for the VAE/text-embeds **cache dir** (that thing
  grows to tens of GB — reuse the 0.2.26 free-space guard, `force` bypass),
  free VRAM ≥ preset floor, no active jobs
- Experimental badge on every Music-train surface, Help page text included

### S3 — Adapter registry + the PiMP loop

> **Status (0.2.32): landed, off-GPU.** `worker/adapters.py` +
> `<models root>/adapters.json` (versioned, keyed by file name, survives being
> hand-edited or corrupted, and is **description not a gate** — delete it and
> every LoRA still loads). `Adapters` page shows trained / imported / found on
> disk, with dataset name + clip count + manifest hash (sorted `name|size`,
> deliberately no mtimes: a copy is not new training data), run id, preset,
> rank, steps, base pack, pinned SimpleTuner. **Audition** queues an ordinary
> 30 s job at 0.8 strength with the dataset's most-used caption, badged
> `audition:<adapter>` on the job and in History, where Restore-to-Generate
> works unchanged; it refuses with a reason when there is no caption (imported
> adapter, deleted dataset) and takes a typed prompt instead. `install_adapter`
> writes the trained row, `import_lora` writes `source: imported`, and
> unregistered files still appear — as `found on disk`, which is the honest
> part. **Still to prove on metal:** an audition of a *real* trained adapter —
> same `load_lora_weights` path the Music picker has used since 0.2.2x, but the
> first one carrying a Studio-trained checkpoint.

- `adapters.json` registry: id, name, kind (music/h3), base pack, trainer
  (SimpleTuner vX), dataset + its manifest hash, created_at, source
  (`trained` | `imported`) — supersedes the filename-only LoRA list, and
  `import_lora` writes `source: imported` rows so the picker shows one
  coherent list
- **Audition**: one click → queues a short generation (music: 30 s with
  dataset's typical caption; H3: still pair) at strength 0.8 with the
  adapter, lands in History tagged `audition:<adapter>`; restore-to-generate
  works as usual

### S4 — H3 LoRA (stills first, clips after)

> **Status (0.2.34): the off-GPU half landed.** An H3 dataset validates (stills
> ≥ 256 px on the short edge, clips ≤ 8 s with the *reason* in the message —
> dialogue footage waits for proof it doesn't wreck the audio heads), the four
> SimpleTuner tiers exist (`h3-24g` with RamTorch, `h3-32g`, `h3-48g`, `h3-80g`)
> each with its own VRAM and cache floor, and an H3 run writes
> `model_family: minimaxh3` + `minimax_h3_target_mode` with `ramtorch` only
> where it pays. `av` is a checkbox, validated: every clip needs audio, stills
> are incompatible with it, and the refusal names the clips. The preset list
> filters by the dataset's kind and the mismatch is refused twice (in the config
> writer and in `start_run`, outside the force switch) — training the wrong model
> and crediting the wrong provenance is exactly what this app exists to prevent.
> `dataset_spec` (kind, stills/clips, mode) is stored in `state.json`, so a
> resume writes the same kind of config even if the dataset folder moved.
> Drift distillation **is written** (`distillation_method: h3_drift`) — that is
> how SimpleTuner's own H3 LoRA examples turn the audio-head safety net on;
> omitting the key leaves it off.
>
> **Still to prove:** everything that needs the card. `minimax_h3_target_mode`,
> `ramtorch` and the resolution buckets are listed in
> `train_config.H3_UNVERIFIED_KEYS` and announced by preflight on every H3
> preset — that warning retires when SimpleTuner's own output confirms them
> (S0 steps 3–5). Unchanged from before: H3 adapter **audition** (the still
> pair) is not built; an H3 LoRA loads on Generate Video today and has no
> one-click preview.

- Video dataset validator in S1's frame; `minimax_h3_target_mode: "video"`
  default, `av` only when the dataset has clean audio (checkbox, not default
  — audio VAE work is extra VRAM/disk)
- Presets map free-VRAM → SimpleTuner tier (24G RamTorch / 32G / 48G / 80G);
  below floor = blocked with the number named, per product style
- Stills/short clips only; "train on clips with dialogue" waits for proof it
  doesn't wreck the audio heads (SimpleTuner's `h3_drift` block is the safety
  net — written, matching the official examples)

### S5 — Long-run hardening

> **Status (0.2.33): landed, off-GPU.** Caches and checkpoints both live in the
> run dir (`runs/<id>/cache`, `runs/<id>/checkpoints`) — that is where
> SimpleTuner writes them — so the sweeper is a per-run **Storage…** dialog on
> the Train page rather than the `/datasets/{id}/cache-size` line this plan
> first sketched; `/train/storage` carries the totals across runs (cached ~15 s,
> because walking a VAE cache means thousands of files and no page should pay
> for that in a 2 s poll). **Prune** keeps the newest N *plus every checkpoint
> that was installed as an adapter*: there is no eval score in SimpleTuner's
> stdout, so "best" means *the one you chose to keep*, and the dialog says that
> out loud instead of inventing a metric. A pruned step goes **whole** — the
> `.safetensors` plus the accelerator/optimiser state SimpleTuner left beside it
> — and the gigabytes in the confirmation come from a `dry_run` of that same
> code path, so the promise is the number. **Clear caches** frees the derived VAE
> and text-embedding caches. No destructive button exists at all while a run is
> live, and the worker refuses a second time with the pid inside the sentence —
> on Windows a live trainer holds those files open, so "cleanup" there is not
> freed disk, it is a half-deleted run. **Resume** rewrites that run's own config
> with `resume_from_checkpoint` and relaunches in place (new pid,
> `resume_count++`, same caches, log keeps appending), and accepts only a
> `.safetensors` that lives inside that run dir. **Export** copies state +
> config + checkpoints + log with an `EXPORT.json` manifest and leaves the
> caches behind; **Import** takes that folder back and refuses to merge onto an
> id that is already here.
>
> **What S5 still cannot prove:** that `resume_from_checkpoint` is the key
> SimpleTuner 4.8.0 actually reads, and what a rebuilt cache really costs in
> wall-clock — both belong to S0 steps 3–5 on the 24 GB card.

- Checkpoint retention policy (keep last N + best, disk-guarded), resume
  picker ("resume run from checkpoint X"), run-dir import/export, cache-dir
  sweeper (`/datasets/{id}/cache-size`, one-click clear)

**Explicitly after v2:** LyCORIS/full-rank UI, RVQ-encoder training
(`scripts/train_minimax_music_rvq_encoder.py` exists — power-user, not v2),
drift-distillation tuning UI, dataset pack sharing, SimpleTuner web-API
deep-linking.

---

## Architecture notes (the parts that aren't obvious)

- **GUI stays torch-free.** TrainRunner lives in the worker; SimpleTuner is
  a subprocess, so our no-torch-in-GUI invariant survives. torch pin
  conflicts live and die inside the `[train]` extra decision in S0.
- **Detached, not inherited.** The worker dies with the GUI; training must
  not. Launch with `start_new_session=True`, stdio redirected into the run
  dir, pidfile + heartbeat (log mtime is the heartbeat; stale > 5 min →
  "lost" badge, offer Open folder).
- **Our contract is the two JSON files.** We generate `config.json` +
  `multidatabackend.json` (golden-tested as text), call `simpletuner train`,
  parse `train.log`, and read the checkpoint dir. That's the whole coupling
  surface — pin the SimpleTuner version and assert its `--version` in
  preflight so drift fails loudly at the boundary.
- **Datasets are read-only to the trainer.** Our pages own writes; SimpleTuner
  writes only into `runs/<id>/` and its caches.
- **Preflight-first everywhere**, same honesty rules as generate: named
  numbers ("24 GB free VRAM needed, 18 GB free — close ComfyUI or pick 768P
  steps preset"), never a mystery OOM.

## Test strategy (CI stays off-GPU, all of it)

- Golden tests: generated `config.json`/`multidatabackend.json` per preset
- Dataset validator tests: stdlib-synthesized wavs, fake ffmpeg stubs,
  missing-caption cases
- Run-dir parser: committed fixture `train.log` fragments (recorded from
  S0), checkpoint discovery, lost-heartbeat detection
- TrainRunner: launch/cancel/reattach against a stub executable (`python -c`
  loop) — no torch, no GPU
- On-GPU validation = the manual checklists at the bottom of each slice,
  run by you, results pasted into the release notes

## Risks & abort criteria

| Risk | Mitigation / abort |
|---|---|
| SimpleTuner `minimaxmusic` LoRAs don't survive the "sounds like the base model" smell test | S0 step 4 is the kill switch; badge or drop, per v1 risk table |
| torch/torchvision/`simpletuner` pin conflicts with our generate pins | S0 exit criterion; fallback = managed separate venv (priced, one decision, then move on) |
| SimpleTuner breaking changes at HEAD | Pin versions; preflight asserts `--version`; upgrade cadence = opt-in checkbox in Settings |
| VAE/text-embeds cache disk blowout | S2 disk guard + S5 sweeper (Storage dialog on the run, whose caches they are) |
| H3 license (territories, derivatives) | No adapter **sharing** in v2 at all; territory text shown on first train preflight, same as first download |
| Training + Comfy fighting over one GPU | Hard mutual exclusion in `/train/preflight`, named in the error |

## Not in v2

From PLAN.md v3, unchanged: community adapter browser, in-app trim/editor,
MCP, RunPod. Plus: Mac training, tensorboard embed (launch external TB
instead), dataset pack sharing, voice-cloning datasets (licensing first).

## Open questions (product, not engineering)

1. Extra-vs-managed-venv for the trainer install → decided at S0 exit.
2. Should completed runs auto-audition (one free render with the new
   adapter), or is any GPU burn after "training finished" unwelcome?
3. Datasets from History: copy clips into the dataset folder (safe, 2× disk)
   or reference them in place (cheap, breaks when people clean History)?

## Next step

**S0 steps 3–5 on a 24 GB card** is the only thing still blocking real
training: `simpletuner train env=<id>`, `STEP_RE`/`LOSS_RE`,
`resume_from_checkpoint`, and the H3 keys in `H3_UNVERIFIED_KEYS` have all been
written against SimpleTuner's docs, not its stdout. Everything around them has a
screen and a test now (S1 datasets incl. H3, S2 Build pages, S3 adapters +
audition, S4a H3 training surfaces, S5 long-run hardening), so that evening is
calibration, not construction — run `scripts/run.sh` (Python 3.12), download the
Music 3 Training Encoder pack (and the H3 diffusers weights, for the video run),
train ~200 steps on 5 clips, install, audition, then let it run past a second
checkpoint and prune it.
Remaining after that: the H3 **audition** (still pair) and whatever the metal
session says the H3 config keys need.

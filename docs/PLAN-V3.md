# MiniMax Studio — v3 Plan: Studio extras (the take after it lands)

Expands the v3 bullets in [PLAN.md](PLAN.md). v1 was generate; v2 was
Build (datasets + LoRA). v3 is the **library loop**: what you do with a take
once it is in History, and how you find adapters without leaving the app.

Written 2026-09-01 against the shipped tree (`0.2.42`). Several PLAN.md v3
bullets already exist — they are inventory, not work.

---

## Already shipped (do not rebuild)

| PLAN.md v3 bullet | Where it landed |
|---|---|
| Prompt enhancement / Context-IR | Generate Video **Context-IR (API)**; Enhance caption/prompt via local OpenAI-compatible LLM; Structure inserts a shot-list scaffold |
| Optional Comfy model-root import | Settings + auto-detect (`~/ai/ComfyUI/models`, `~/models`, `extra_model_paths.yaml`); **Start ComfyUI** from Welcome / Settings / Go |
| LoRA picker + import | Adapters page: trained / imported / found on disk; **Import…** a `.safetensors`; Audition is one click (Music 30 s, H3 still-pair / t2va) |
| In-process pipe cache | CUDA H3 and Music keep `runtime.h3_pipe` / `runtime.music_pipe` across jobs in the same worker process |
| History basics | Play, Restore to Generate, Export (copy the file), Show in folder, Copy prompt, Delete, filter by kind + prompt |

v1.1 GGUF, Spectrum/latent upscale, and 2K local were never v3 and stay out.

---

## What is actually missing

History is a list plus a player. There is no in/out trim — PiMP had a light
editor and PLAN.md parked it for v3. Adapters are **import a file you already
have**; the Generate Music empty-state still says community adapters are rare.
Repeat generate on official CUDA is fast only while Studio stays open (the
worker dies with the window). Comfy INT8 already stays warm if the user left
ComfyUI running.

That is the v3 product: **trim a take, find a LoRA, don't reload 30 GB for
the next prompt** — without becoming a node graph, a cloud GPU broker, or a
second trainer.

---

## Locked v3 decisions

| Decision | Choice |
|---|---|
| Theme | Library extras. No new training loop, no Comfy embed, no wires |
| Trim | `ffmpeg` on PATH (already required for H3 mux). Writes a **new** History row; the original take is never mutated. Provenance `trimmed-from: <id>` |
| Trim scope | In/out points only. No DSP, no inpaint, no beat grid, no titles. Video snaps to the H3 frame grid (same 24 fps / 17k+5 rule as Generate) |
| Adapter catalog | **Curated list** in `catalog.py` (same shape as model packs: repo, license, markers, territory). Not a live scrape of Hugging Face |
| Sharing | Still no "publish this LoRA". H3 Community License V.3 and PLAN-V2's legal pass stand. Download-with-notice is the same risk class as Models |
| Repeat generate | Measure first (v3 S0). Default is "leave Studio or Comfy open." A detached **warm worker** is in scope if the measurement says quit-and-relaunch is the pain. **SGLang is after v3** unless S0 shows official CUDA cold-start is worse than that |
| GUI invariant | GUI still never imports torch. Trim and catalog live in the worker; ffmpeg is a subprocess like SimpleTuner |
| Hardware | Trim is CPU/ffmpeg, all platforms. Catalog is download. Warm-worker is CUDA (and MLX Music on Mac) — same generate backends as v1 |

---

## Slices, in order

Same vertical-slice discipline as v1/v2. Off-GPU and CI-coverable first;
metal only where a take has to move.

### S0 — Gate (off-GPU + one measurement)

> Nothing in S1–S3 ships until this page has answers written down.

1. **Adapter census.** Count real MiniMax Music 3 and H3 LoRAs on Hugging
   Face that a catalog row could point at (filename, license, size). If Music
   is still "two experimental reggae tests," S2 is Music-honest: empty state
   stays "import a file," and the catalog ships H3 (with territory notice)
   and/or Turbo-class add-ons only. Do not invent a bazaar.
2. **ffmpeg trim PoC.** Worker helper: in/out seconds → new wav/mp4, no GUI.
   Tests with a stub binary and a tiny fixture. Named errors when ffmpeg is
   missing (same honesty as H3 preflight).
3. **Repeat-generate timing** (optional, this machine). Official CUDA H3 or
   Music: second job in the same worker vs kill worker and reload. Write the
   two numbers into this plan. That decides whether S3 is a slice or a skip.

**Exit = this document updated with the census + the two timings (or "S3
skipped").** Abort S2 if there is nothing honest to list. Abort S3 if warm
pipe already wins by a lot.

> **Status (0.2.43): S0 closed.** Census 2026-09-01. Trim helper
> `worker/trim.py` (stub + real-ffmpeg wav cut). Repeat-generate: in-session
> pipe cache already exists; official H3 CUDA cannot be timed on this box
> (no transformer shard); Music CUDA is on disk but a 63 GB cold load was
> not burned for S0. S3 (warm worker across GUI quit) is **deferred after
> S1**, not skipped and not SGLang. S2 is **not aborted** — H3 has a real
> catalog; Music stays thin.

#### S0 census (Hugging Face, 2026-09-01)

**Music 3** — 22 adapter-tagged repos on `MiniMaxAI/MiniMax-Music3`. Honest
generate LoRAs, not a bazaar:

| Repo | What it is | Catalog? |
|---|---|---|
| `bghira/minimax-music-suno-reggae-rank128-v2` (v1 too) | The two experimental reggae tests PLAN.md named | Maybe one row; still a test |
| `SimpleTuner/minimaxmusic-reggae-test-lora-comfyui-v1-4k` | ComfyUI export of the same idea | Duplicate-ish |
| `guillaume127/MiniMax-Music-3-Turbo-FP8` | 8-step turbo LoRA (~180 MB) plus FP8 checkpoints | Turbo LoRA yes; FP8 weights are not a LoRA |
| `ntc-ai/minimax-music3-concept-sliders` | Bipolar LM sliders (gender/energy/…) | Different UX — not S2 |
| `RareConcepts/*`, `MiniMaxMusicTraining/soad-*`, tournament dumps | Identity/SOAD research | No |

Music empty-state ("import a file") stays honest. S2 may add reggae-v2 and
the 8-step Music turbo, not a storefront.

**H3** — 40 adapter-tagged repos on `MiniMaxAI/MiniMax-H3`. Enough to curate:

| Repo | What it is | Catalog? |
|---|---|---|
| `fal/MiniMax-H3-Realism-People-LoRA` | People/faces, trigger `r34l1sm`, H3 community license, ~46k downloads | **Yes** — the obvious first row |
| `lovis93/studio-1939-old-animation-lora-minimax-h3` | Animation style | Yes if the file is a plain `.safetensors` |
| `MATLOWAI/MiniMax-H3-Motion-Adapter` | Rank-16 motion, FL2VA+Ref2VA | Yes |
| `ostris/minimax_h3_ref2va_jacked_lora` | Joke/style (muscles) | Optional |
| Comfy-Org / larryvrh / lightx2v Turbo | Fast mode | **Already a Models pack** (`h3-turbo`) — do not duplicate |
| Alibaba PAI Acc / PDD 8-step | Extra head bank, dedicated nodes | Not S2 (not a plain LoRA picker file) |
| `bghira/minimax-h3-anyflow-wip`, SVD-delta experiments | WIP | No |
| NSFW AfterMidnight et al. | Explicit | Not the default catalog |

S2 proceeds: H3 style LoRAs with the same territory notice as `h3-train`.
Turbo stays on Models.

#### S0 timings

| Question | Answer |
|---|---|
| Second job, same worker | Already cached (`runtime.h3_pipe` / `runtime.music_pipe`). No GPU run needed to know this. |
| Kill worker and reload | Official H3 CUDA: **cannot** — this machine has the training tree, not `transformer/diffusion_pytorch_model-00001-of-00014.safetensors`. Music CUDA: pack is on disk; cold load is ~63 GB VRAM/RAM. Not measured in S0. |
| Comfy INT8 | Already warm if the user left ComfyUI running (v1). |
| S3 | **Deferred.** In-session is solved. Detached warm-worker is quit-and-relaunch, still a product slice after History trim. SGLang stays after v3. |

### S1 — History trim

The PiMP leftover people feel every day.

- History grows **Trim…** (in/out on the selected take). Preview in the
  existing player if cheap; otherwise the new file is the preview.
- Worker writes a sibling file next to the take, then a new History entry:
  same prompt/lyrics/loras/mode, `trimmed-from` set, duration the new length.
- Restore-to-Generate on the child still works (it is just another take).
- Delete of the child does not delete the parent; delete of the parent does
  not chase children.
- Music: wav via `ffmpeg -ss -to`. H3: mp4+audio, re-mux, snap out-point to
  the frame grid so a trimmed take can still be a first/last frame later.
- Missing ffmpeg: the button names the install, does not grey out as a
  mystery.

Tests: stub ffmpeg, History containment (new file stays under the history
root), provenance round-trip, restore still emits. Metal: trim one Music wav
and one H3 mp4, play both, restore the child.

### S2 — Adapter catalog

Browse and download, not a store.

- Models-page cousin on Adapters: a short list of **known** LoRAs (id, title,
  summary, repo, approx GB, family, license, territory notice). Download uses
  the same snapshot + disk-space guard as packs.
- Land in `models/loras/` (or the Comfy loras root we already search).
  `import_lora` / `record_imported` so the picker and Audition see them
  without a second path.
- H3 rows show the same US/EU/UK/KR text as `h3-train`. Music rows show the
  Music 3 credit line.
- Empty catalog (S0 census came back empty) → S2 is a skip, not a fake shop.

Tests: catalog rows, disk guard, import-after-download, territory string.
Metal: one small adapter download if S0 found one; otherwise skip.

### S3 — Repeat generate (measurement-gated)

Only if S0 said quit-and-relaunch is the expensive part.

- **In scope:** keep the generate worker alive when the window quits (same
  idea as training: detached process, GUI reconnects on launch, named in the
  status bar). One GPU still. Training preflight already refuses while a
  generate job is live — a warm worker with no job is idle, not a fight.
- **Out of scope:** SGLang as a third server, RunPod, "generate on another
  machine."
- If S0 timings show the in-process pipe cache is enough, **write "S3
  skipped"** here and stop. Do not build a daemon for theory.

---

## Architecture notes

- **Trim is a worker job**, not a GUI `QProcess`. Same queue honesty as
  generate (one GPU job at a time). Trim itself should not need the GPU;
  it can run while a train is live if we keep it CPU-only. If we ever
  GPU-trim, it joins the mutual-exclusion line.
- **New History row, not in-place.** In-place edit would break Restore,
  dataset-from-History copies, and audition badges that point at a take.
- **Catalog is packs, not search.** Live HF search is how you get surprise
  20 GB files and license-less dumps. Curated rows match Models.
- **Warm worker (S3) must not become a second Studio.** One listener, one
  token, reconnect like `train_runs` already does for a detached trainer.

## Test strategy

CI stays off-GPU, all of it.

- Trim: stub `ffmpeg` argv + fixture bytes; refuse when the binary is missing
- History: child entry schema, `trimmed-from`, delete isolation
- Catalog: pack_status markers, disk guard, import registry row
- Warm worker (only if S3 is not skipped): reconnect against a stub process

Metal checklists at the bottom of S1 (and S2/S3 if they ship), pasted into
the release notes.

## Risks & abort criteria

| Risk | Mitigation / abort |
|---|---|
| ffmpeg re-encode ruins H3 audio sync | Prefer `-c copy` when in/out land on keyframes; if copy cannot hit the frame grid, re-encode and **listen** on metal before shipping. Abort the video half of S1 if muxed audio drifts |
| Music/H3 LoRA catalog is empty or license-messy | S0 census; skip S2 rather than scrape |
| H3 adapter download treated as "we distribute weights" | Same territory notice + license file as model packs; no upload |
| Warm worker fights training or Comfy for the GPU | Idle warm worker holds weights in RAM, not a running job. Preflight already names active generate jobs. If idle VRAM reservation starves training, S3 is a skip |
| Trim-in-place temptation | Locked: new row only |

## Not in v3

From PLAN.md / PLAN-V2, unchanged and still later:

- MCP
- RunPod (or any remote worker)
- Mac training
- LyCORIS / full-rank training UI
- RVQ-encoder training
- Drift-distillation tuning UI
- Dataset pack sharing / voice-clone datasets (licensing first)
- Tensorboard embed
- SGLang persistent server (unless S0 forces it — default is after v3)
- GGUF / 8 GB H3 preview
- Spectrum, latent upscale, motion-context packs
- In-app DSP editor, latent inpaint, 2K local
- Auto-audition (declined 2026-09-01)

## Open questions (product, not engineering)

1. Catalog source of truth: hand-curated `catalog.py` rows (recommended) vs
   a small JSON file you can edit without a release.
2. Trim UI: a dialog with two timestamps, or in/out marks on the existing
   player timeline? Dialog is the smaller slice; timeline is nicer.
3. Should a trimmed take be offered as "add to dataset" with the parent
   caption, or is that just Restore + Datasets as today?

Recommended defaults: curated Python rows, timestamp dialog, no new dataset
shortcut (History → Add from History already copies).

## Next step

**S1 — History trim.** S0 is closed (0.2.43). Wire `trim_media` to a History
**Trim…** dialog: new row, `trimmed-from`, original kept.

# Code review — Qwen 0.2.28–0.2.34 training stack

Review of the committed work on `main` (working tree was clean). Focus is
PLAN-V2 S0–S5 / S4a: datasets, train config, detached runs, adapters, Build UI.

H3 config keys were checked against SimpleTuner **4.8.0** example configs
(`minimaxh3-fl2va-convrot-int8-24g.peft-lora+ramtorch`,
`minimaxh3-fl2va-convrot-int8.peft-lora`,
`minimaxh3-fl2va-convrot-int8-32g.peft-lora`,
`minimaxh3-fl2va-convrot-int8-48g.peft-lora`,
`minimaxh3-t2v-fp8-141g.peft-lora+anyflow`) and
`simpletuner/helpers/models/minimaxh3/model.py`.

Training-stack issues 1–29 are **fixed**. Generate-path follow-up 30–37 is **fixed**.
Music / poll follow-up 40–46 is **fixed**. GUI jobs tick / import-kind 47–50 are **fixed**.
Shipped as **0.2.35**.

0.2.35 rest-of-app pass (issues 51–67) is **fixed**.
Shipped as **0.2.36**.

## Summary

The off-GPU scaffolding is coherent: family gating, `dataset_spec` on resume,
Storage dry-run, and the H3 validator's "warn when unmeasured" rule are real.
The H3 writer did not match the pinned trainer's contract — a 24 GB H3 run
would have started with the wrong `model_family`, a made-up flavour, and a
RamTorch key SimpleTuner never reads. Installing that run would then register
the adapter as Music. Those contracts now follow SimpleTuner 4.8.0's own
example configs.

## Issues

### Issue 1 -- Severity: bug
- File: src/minimax_studio/worker/train_config.py:314
- Description: H3 configs write `model_family: "minimax_h3"`. SimpleTuner 4.8.0's family id is `minimaxh3` (every official example, and `MiniMaxH3.NAME` registration). An unknown family fails at launch, not "later on metal".
- Suggestion: Write `minimaxh3`.
- Status: fixed

### Issue 2 -- Severity: bug
- File: src/minimax_studio/worker/train_config.py:315
- Description: H3 configs write `model_flavour: "h3"`. SimpleTuner flavours are `fl2va` (official bf16), `convrot-int8`, `ref2va`, `fp8-e4m3fn`, etc. `"h3"` is not a flavour.
- Suggestion: INT8 tiers (`h3-24g`/`32g`/`48g`) → `convrot-int8`. `h3-80g` → `fl2va`.
- Status: fixed

### Issue 3 -- Severity: bug
- File: src/minimax_studio/worker/train_config.py:320
- Description: 24 GB H3 writes `"ram_torch": true`. SimpleTuner JSON keys match CLI flags; the flag is `--ramtorch`. Official 24G example is `"ramtorch": true`. The underscore key is ignored, so the floor tier does not actually offload.
- Suggestion: Write `ramtorch` (and `ramtorch_text_encoder` on the 24G tier, matching the example).
- Status: fixed

### Issue 4 -- Severity: bug
- File: src/minimax_studio/worker/train_config.py:92
- Description: H3 INT8 tiers write `base_model_precision: "int8-quanto"`. Official convrot-int8 examples use `"no_change"` because the flavour already loads INT8 ConvRot weights. Quanto on top is the wrong 8-bit. `h3-80g` writes `"bf16"`, which is not a SimpleTuner precision name (`no_change` loads the official FL2VA bf16 weights as-is).
- Suggestion: All H3 presets write `no_change`; flavour carries the weight format.
- Status: fixed

### Issue 5 -- Severity: bug
- File: src/minimax_studio/worker/train_config.py:333
- Description: Drift distillation is deliberately unwritten on the assumption SimpleTuner leaves it on. Official H3 LoRA examples all set `"distillation_method": "h3_drift"` with a config block — the default without the key is off. PLAN-V2's safety net for the audio heads never engages.
- Suggestion: Write the same `h3_drift` block the examples use. Also write `flow_schedule_shift: 12` / `audio_flow_schedule_shift: 3` and `resolution_type: "pixel_area"` (SimpleTuner warns if video shift inherits the global 3.0 default).
- Status: fixed

### Issue 6 -- Severity: bug
- File: src/minimax_studio/worker/adapters.py:242
- Description: `record_trained` hardcodes `kind: "music"` and `base_pack: "music3-cuda"`. An installed H3 adapter is listed as Music, `can_audition` becomes true, and Audition queues a 30 s song.
- Suggestion: Derive kind/base_pack from `run_state["family"]` / `dataset_kind`.
- Status: fixed

### Issue 7 -- Severity: bug
- File: src/minimax_studio/worker/adapters.py:36
- Description: `dataset_fingerprint` only counts audio/video clip suffixes. An H3 stills-only set fingerprints as `clip_count: 0` with no hash — provenance of the thing S4 exists to train.
- Suggestion: Include `STILL_EXTS` (png/jpg/jpeg/webp).
- Status: fixed

### Issue 8 -- Severity: bug
- File: src/minimax_studio/worker/datasets.py:575
- Description: `set_h3_target_mode(..., "av")` treats unmeasured clips as silent (`has_audio` defaults False). Without ffmpeg it accuses clips of having no audio — the opposite of the validator's "warn, don't accuse" rule. `test_target_mode_route_refuses_with_a_reason` even accepts either wording.
- Suggestion: If files were unmeasured, refuse with the measurement warning, not "have none".
- Status: fixed

### Issue 9 -- Severity: bug
- File: src/minimax_studio/worker/train_runs.py:266
- Description: `install_adapter(path=...)` does not require the file to live under the run dir. `resume_run` does. An absolute path to any `.safetensors` is copied and credited to this run.
- Suggestion: Same containment check as resume (`run_dir` must be a parent).
- Status: fixed

### Issue 10 -- Severity: bug
- File: src/minimax_studio/worker/train_runs.py:254
- Description: `progress()["checkpoints"]` uses `str(path.relative_to(run_dir))` (Windows backslashes). Storage/prune/export were fixed to `.as_posix()` in 0.2.33; the progress payload used by the Train page was not.
- Suggestion: `.as_posix()`.
- Status: fixed

### Issue 11 -- Severity: bug
- File: src/minimax_studio/worker/train_runs.py:579
- Description: Resume writes `resume_from_checkpoint` as the path to a `.safetensors` LoRA file. SimpleTuner's option accepts `latest` or a **checkpoint directory** (optimizer + weights), not an exported adapter file. `--init_lora` is the safetensors-only path and cannot combine with resume.
- Suggestion: Default resume → `"latest"`. Explicit pick → the checkpoint folder that contains the file.
- Status: fixed

### Issue 12 -- Severity: bug
- File: src/minimax_studio/worker/train_config.py:186
- Description: Cheap music preflight only globs `*.wav`/`*.flac` and says "No audio files (*.wav)". `MEDIA_BY_KIND["music"]` includes `.mp3`. An mp3-only folder fails the cheap check and never reaches the real validator.
- Suggestion: Use `MEDIA_BY_KIND["music"]`.
- Status: fixed

### Issue 13 -- Severity: bug
- File: src/minimax_studio/ui/pages/datasets_page.py:46
- Description: History picker for a Video dataset only offers `.mp4/.mov/.webm`. The worker accepts stills (`.png/.jpg/.webp`) and S4 is "stills first". T2I takes in History cannot be added.
- Suggestion: Mirror worker `MEDIA_BY_KIND["video"]`.
- Status: fixed

### Issue 14 -- Severity: bug
- File: src/minimax_studio/ui/pages/train_page.py:667
- Description: After installing an H3 adapter the dialog and tooltip still say "Generate Music picker".
- Suggestion: Branch on the run's `family`.
- Status: fixed

### Issue 15 -- Severity: bug
- File: src/minimax_studio/worker/server.py:570
- Description: `GET /datasets/{id}` does `json.loads` on `validation.json` with no try/except. A truncated report 500s the detail view. Manifest reads elsewhere are tolerant.
- Suggestion: Swallow `OSError`/`JSONDecodeError` and return `validation: null`.
- Status: fixed

### Issue 16 -- Severity: suggestion
- File: src/minimax_studio/worker/datasets.py:120
- Description: `get_dataset` / `get_run` join the id onto the root with no check. FastAPI `{id}` can be `..` and then `datasets_root() / ".."` is the parent of the datasets folder (`delete_dataset` would `rmtree` it).
- Suggestion: Reject ids that are not a single path segment.
- Status: fixed

### Issue 17 -- Severity: suggestion
- File: src/minimax_studio/worker/datasets.py:132
- Description: `list_entries` unions every music and video extension, so a stray `.wav` in an H3 folder shows as an entry but is never validated.
- Suggestion: Filter by the manifest kind when present.
- Status: fixed

### Issue 18 -- Severity: suggestion
- File: src/minimax_studio/ui/pages/help_page.py:102
- Description: Help still says Install adapter puts the file in the Generate Music picker, after S4a added H3 training.
- Suggestion: Name both pickers.
- Status: fixed

### Issue 19 -- Severity: bug
- File: src/minimax_studio/worker/train_runs.py:680
- Description: `import_run` joins `state.json`'s `id` onto the runs root with no slug check. An absolute id on POSIX (`"/tmp/evil"`) replaces the root; `".."` walks out.
- Suggestion: Reject non-segment ids; require the resolved target to stay under `runs_root()`.
- Status: fixed

### Issue 20 -- Severity: bug
- File: src/minimax_studio/worker/train_config.py:362
- Description: Music validator blesses clips up to 300 s; the written SimpleTuner backend has `max_duration_seconds: 60`, so a "ready" 90 s wav is skipped at discovery. Mystery drop the validator exists to prevent.
- Suggestion: Write `max_duration_seconds` from `datasets.MAX_SECONDS`.
- Status: fixed

### Issue 21 -- Severity: bug
- File: src/minimax_studio/ui/pages/train_page.py:54
- Description: `_FALLBACK_PRESETS` has no `family` and is Music-shaped. An H3 dataset before preflight returns, or after a preflight failure, shows "24 GB — conservative LoRA".
- Suggestion: Per-family fallbacks; never substitute a Music tier when the dataset is H3.
- Status: fixed

### Issue 22 -- Severity: bug
- File: src/minimax_studio/ui/pages/train_page.py:476
- Description: Changing VRAM preset only updates rank. The Ready/problem list stays about the previous tier until Check again.
- Suggestion: Call `preflight()` from `_preset_changed`.
- Status: fixed

### Issue 23 -- Severity: bug
- File: src/minimax_studio/worker/datasets.py:424
- Description: Music mp3/flac without ffprobe become file *problems* (`cannot read audio: install ffmpeg`). H3 treats the same fact as a warning. Docstring/Help promise the warning.
- Suggestion: Missing ffprobe → report warning; keep "cannot read" for a real decode failure.
- Status: fixed

### Issue 24 -- Severity: bug
- File: src/minimax_studio/worker/datasets.py:306
- Description: `probe_video`/`probe_audio` ignore returncode and do not catch `TimeoutExpired`. Empty stdout on a failed still probe looks like a successful measurement with no size. One hung file 500s validate.
- Suggestion: Non-zero returncode / timeout → RuntimeError. Catch `SubprocessError` in the validators.
- Status: fixed

### Issue 25 -- Severity: bug
- File: src/minimax_studio/ui/pages/datasets_page.py:329
- Description: Dataset `warnings` are never rendered. An unmeasured H3 set looks identical to a clean one; Help says the page names what could not be measured.
- Suggestion: Show warnings on the status line.
- Status: fixed

### Issue 26 -- Severity: suggestion
- File: src/minimax_studio/worker/datasets.py:200
- Description: `import_folder` counts `.txt` and `.lyrics` as `captions`, so the page math `missing = copied - captions` under-reports missing captions. Also copies non-files (directories named like media).
- Suggestion: Count media that had a `.txt`; skip non-files.
- Status: fixed

### Issue 27 -- Severity: bug
- File: src/minimax_studio/worker/train_runs.py:808
- Description: `_refresh` treats any exit 0 as `completed`, then cancel. SimpleTuner often SIGTERM-exits 0, so Cancel would badge a half-trained run as success.
- Suggestion: `cancel_requested` wins over a clean exit.
- Status: fixed

### Issue 28 -- Severity: bug
- File: src/minimax_studio/worker/train_runs.py:791
- Description: PLAN-V2's log-mtime heartbeat (stale > 5 min → lost) was never implemented. A hung pid with a silent log stayed "running" forever.
- Suggestion: If the process is alive but the log (and start time) are older than 5 minutes, status `lost`. Still live for GPU/storage; Cancel still works. Log movement restores `running`.
- Status: fixed

### Issue 29 -- Severity: bug
- File: src/minimax_studio/worker/loras.py:65
- Description: `import_lora` copied to `models/loras/<source.name>` and overwrote. SimpleTuner's usual export is `pytorch_lora_weights.safetensors`, so a second Install adapter replaced the first run's file and registry row.
- Suggestion: Prefix trained installs with the run name; `_free_lora_path` if that name is taken. Never overwrite.
- Status: fixed

## Generate-path follow-up (from the rest-of-app pass)

### Issue 30 -- Severity: bug
- File: src/minimax_studio/worker/jobs.py:273
- Description: History did not persist `assets` / `ref_image_size`, so Restore dropped frames.
- Suggestion: Write them on `record_entry`.
- Status: fixed

### Issue 31 -- Severity: bug
- File: src/minimax_studio/ui/ready.py:27
- Description: Generate confirm omitted `resolution`, skipping the 2K-on-local preflight block.
- Suggestion: Pass the combo value through.
- Status: fixed

### Issue 32 -- Severity: bug
- File: src/minimax_studio/worker/backends/h3_comfy.py:427
- Description: LoRA 2+ graph nodes used `Path(name).name`, stripping Comfy subfolders after resolve.
- Suggestion: Keep the resolved `lora_name`.
- Status: fixed

### Issue 33 -- Severity: bug
- File: src/minimax_studio/worker/backends/h3.py:116
- Description: Fast on official diffusers appended Turbo and used 8 steps in Ref2VA; Comfy prepends Turbo and uses 4.
- Suggestion: Insert Turbo first; Ref2VA Fast = 4 steps.
- Status: fixed

### Issue 34 -- Severity: bug
- File: src/minimax_studio/worker/backends/h3.py:301
- Description: `_apply_loras` swallowed load failures, so a selected adapter could silently not apply.
- Suggestion: Fail the job with the LoRA filename in the error.
- Status: fixed

### Issue 35 -- Severity: bug
- File: src/minimax_studio/worker/backends/h3_api.py:133
- Description: Cancel only stopped the local poll. Queued MiniMax tasks can be DELETE'd with no charge.
- Suggestion: DELETE `/v2/video_generation/{task_id}` on cancel.
- Status: fixed

### Issue 36 -- Severity: bug
- File: src/minimax_studio/worker/backends/h3_api.py:174
- Description: Every reference was Base64 with no size check against the 64 MB API body cap.
- Suggestion: Refuse files over ~45 MB with a named number.
- Status: fixed

### Issue 37 -- Severity: bug
- File: src/minimax_studio/worker/downloads.py:145
- Description: `snapshot_download` ignored the cancel Event. Cancel waited until the whole pack finished.
- Suggestion: Run the Hugging Face snapshot in a child process and terminate it on cancel.
- Status: fixed

### Issue 38 -- Severity: bug
- File: src/minimax_studio/worker/datasets.py:546
- Description: With `h3_target_mode: av`, unmeasured clips still left `report.ok` true (captions only). Training could start without ever seeing audio.
- Suggestion: If av is on and files were unmeasured, the report is not ok.
- Status: fixed

### Issue 39 -- Severity: bug
- File: src/minimax_studio/ui/pages/train_page.py:720
- Description: Resume confirmed a lexicographic `progress.checkpoints[-1]` while the worker resumes `latest` (mtime). The box could name the wrong file.
- Suggestion: Confirm “newest checkpoint”; Storage… is the picker for a specific one.
- Status: fixed

## Music / poll follow-up

### Issue 40 -- Severity: bug
- File: src/minimax_studio/worker/backends/music_api.py
- Description: Empty lyrics forced `is_instrumental: true`. MiniMax Music 3.0's empty-lyrics + `lyrics_optimizer` path writes lyrics from the prompt; local CUDA/Comfy already did that. API takes became instrumentals.
- Suggestion: Blank lyrics → `lyrics_optimizer: true` and `is_instrumental: false`.
- Status: fixed

### Issue 41 -- Severity: bug
- File: src/minimax_studio/worker/backends/music_comfy.py, music.py
- Description: CUDA applied Music LoRAs; Comfy ignored them; API and MLX dropped them silently. A selected adapter could do nothing.
- Suggestion: Stack `LoraLoaderModelOnly` on the Comfy graph. API/MLX refuse with a named reason.
- Status: fixed

### Issue 42 -- Severity: bug
- File: src/minimax_studio/worker/backends/music_comfy.py
- Description: INT8-missing fallback hardcoded the FP16 DiT basename, ignoring the name Comfy actually lists (subfolders).
- Suggestion: Resolve INT8 or FP16 the same way missing-file checks do; retry with the resolved FP16 name.
- Status: fixed

### Issue 43 -- Severity: bug
- File: src/minimax_studio/ui/pages/train_page.py
- Description: The 2 s Train tick listed runs and tailed the log on the GUI thread. A stuck worker froze the window. A list failure also wiped the last view.
- Suggestion: `poll()` off-thread; keep the last list on failure; constructor `refresh()` stays sync for tests.
- Status: fixed

### Issue 44 -- Severity: bug
- File: src/minimax_studio/ui/pages/datasets_page.py, adapters_page.py, main_window.py
- Description: Datasets and Adapters 2 s ticks still did HTTP on the GUI thread (and rebuilt widgets even when nothing changed).
- Suggestion: Same `poll()` path as Train; keep the last list on failure; skip widget rebuilds when the signature is unchanged.
- Status: fixed

### Issue 45 -- Severity: bug
- File: src/minimax_studio/worker/backends/music_api.py
- Description: Cancel during the Music API's 180 s POST only flipped local status; the HTTP call ran to completion.
- Suggestion: Close the httpx client when the job is cancelled.
- Status: fixed

### Issue 46 -- Severity: bug
- File: src/minimax_studio/worker/adapters.py
- Description: Untracked `.safetensors` in `h3-comfy` / `minimax-h3` folders defaulted to `kind: music`, so Audition queued a 30 s song. Imports from those folders did the same.
- Suggestion: Infer `h3` from those folder names; `can_audition` stays Music-only.
- Status: fixed

### Issue 47 -- Severity: bug
- File: src/minimax_studio/ui/main_window.py
- Description: The 500 ms tick called `list_jobs` and `list_downloads` on the GUI thread (downloads twice, once from job-status). A slow worker froze Generate and the status bar.
- Suggestion: One off-thread fetch shared by both pages and the bar; keep the last snapshot; apply when it returns.
- Status: fixed

### Issue 48 -- Severity: bug
- File: src/minimax_studio/worker/loras.py, ui/ready.py
- Description: Importing an H3 `.safetensors` from Downloads (or any folder not named `h3-comfy` / `minimax-h3`) recorded `kind: music`, so Audition queued a song.
- Suggestion: `import_lora(..., kind=)`; the Import dialog asks Music vs H3 when the folder does not already say H3.
- Status: fixed

### Issue 49 -- Severity: suggestion
- File: src/minimax_studio/worker/backends/music.py
- Description: The stub backend still cannot apply LoRAs (it writes a tone). Tests and `MINIMAX_STUDIO_STUB=1` rely on that.
- Suggestion: Say so in the job message (`stub skips LoRAs`) rather than looking like the adapter ran.
- Status: fixed

### Issue 50 -- Severity: bug
- File: src/minimax_studio/ui/pages/models_page.py
- Description: While the Models page is showing, the 2 s tick called `list_packs` (tree walk) on the GUI thread.
- Suggestion: Same `poll()` off-thread path; constructor `refresh()` stays sync.
- Status: fixed

## 0.2.35 rest-of-app pass

Review of committed `main` at 0.2.35 (working tree clean). 1–50 stay fixed.
Focus: generate CUDA state, history containment, GUI-thread leftovers the 0.2.35
poll work missed, train GPU selection, download child lifetime, persistence.

Not in this list (already tracked elsewhere): H3 SimpleTuner keys / encoder path
(`H3_UNVERIFIED_KEYS`, PLAN-V2 S0 steps 3–5 on metal), H3 still-pair audition.

### Issue 51 -- Severity: bug
- File: src/minimax_studio/worker/backends/h3.py, music.py, runtime.py
- Description: `runtime.h3_pipe` / `runtime.music_pipe` are reused across jobs. `_apply_loras` only loads; an empty LoRA list returns immediately. Pipes are cleared only on Settings save. Fast then Quality on official CUDA still has Turbo; LoRA A then none still sounds like A; A then B can stack. Comfy/API are fine (fresh graph).
- Suggestion: Unload/disable adapters at the start of every CUDA job, or include the LoRA set in the pipe cache key and reload when it changes.
- Status: fixed

### Issue 52 -- Severity: bug
- File: src/minimax_studio/worker/history.py
- Description: `delete_entry` / `get_entry` join `entry_id` onto `history_root()` with no segment check. Datasets and train runs already use `_require_id`. `DELETE /history/..` is `rmtree(output_dir)`; `DELETE /history/../models` wipes the models tree. GUI encodes segments so it will not send this; `--worker-only` and anything with the token will. `add_from_history` uses the same `get_entry`.
- Suggestion: Same `_require_id` as datasets/runs — reject ids that are not a single path segment.
- Status: fixed

### Issue 53 -- Severity: bug
- File: src/minimax_studio/ui/pages/music_page.py, video_page.py
- Description: The 500 ms tick fetches jobs off-thread, then `poll(snapshot)` falls back to sync `get_job` when the live id is missing. Worker crash → snapshot `[]` → `get_job` with a 30 s httpx timeout on the GUI thread, every tick, while a job is tracked. The 0.2.35 freeze fix has a hole.
- Suggestion: If a snapshot was passed and the id is not in it, treat it as unknown / worker down. Do not round-trip. `poll()` with no snapshot (tests / constructor) may still `get_job`.
- Status: fixed

### Issue 54 -- Severity: bug
- File: src/minimax_studio/worker/probe.py, train_runs.py, device.py
- Description: `select_cuda_device()` only pins in-process torch. SimpleTuner is a subprocess with no `CUDA_VISIBLE_DEVICES` / `--gpu_ids`. Preflight `free_vram_gb` is max free across all GPUs, not the selected one. Two-GPU box, Studio on GPU 1: generate uses 1, train lands on 0, VRAM check can pass on the wrong card. Help already says Comfy uses `--default-device`.
- Suggestion: Pass the Settings CUDA index into the trainer env. Probe free VRAM on that GPU, not `max()` across the list.
- Status: fixed

### Issue 55 -- Severity: bug
- File: src/minimax_studio/worker/downloads.py, app.py
- Description: `_cancellable_hf_snapshot` uses `stderr=PIPE` and only reads it after exit — `huggingface_hub` is chatty; a full pipe deadlocks the child. The child is also `start_new_session=True`. Quitting Studio SIGTERMs the worker; daemon threads skip `finally`; a 130 GB pull keeps going. Training surviving quit is the product; a pack download is not.
- Suggestion: `DEVNULL` or a log file for stderr. Kill the snapshot process group from `_stop_worker` (or a non-daemon tracker).
- Status: fixed

### Issue 56 -- Severity: bug
- File: src/minimax_studio/worker/backends/music.py
- Description: Blank lyrics on the Music API now uses `lyrics_optimizer` (issue 40). CUDA passes empty lyrics through. MLX still does `lyrics=request.lyrics or "[instrumental]"`. Same form, three backends, three meanings.
- Suggestion: Match CUDA/API (empty lyrics are not an instrumental), or say so on the inspector the way the Music API already does for duration/seed.
- Status: fixed

### Issue 57 -- Severity: bug
- File: src/minimax_studio/ui/ready.py
- Description: Generate confirm still calls `/preflight` on the GUI thread. Preflight can hit Comfy `/object_info` and freeze the window until it returns. The inspector route check already moved off-thread; the click path did not.
- Suggestion: Same off-thread preflight as `_refresh_route`, then show the confirm/block dialog on the result. Or reuse the last route snapshot when it is fresh.
- Status: fixed

### Issue 58 -- Severity: bug
- File: src/minimax_studio/ui/main_window.py
- Description: `_refresh_train_status` (4 s tick when not on Train) calls `list_train_runs` on the GUI thread — a run-dir walk. The bar also ignores `lost` (only `running` / `queued`), so a hung trainer disappears from the chrome while generate preflight still warns via `live_runs()`.
- Suggestion: Off-thread like the jobs tick. Include `lost` in the status-bar live set, matching Train page `_LIVE`.
- Status: fixed

### Issue 59 -- Severity: bug
- File: src/minimax_studio/ui/pages/train_page.py, datasets_page.py
- Description: Train **Start** runs `train_preflight` + `validate_dataset` (30 min timeout) + `start_train_run` (5 min) on the GUI thread. Datasets **Validate** is the same 30 min call. A 500-clip folder freezes the shell until ffprobe finishes.
- Suggestion: Those two actions off-thread; disable the button and keep a status line. Constructor `refresh()` stays sync for tests.
- Status: fixed

### Issue 60 -- Severity: suggestion
- File: src/minimax_studio/worker/jobs.py
- Description: `runtime.jobs` only grows. `list_jobs` every 500 ms ships every prompt generated this session. History already has the takes.
- Suggestion: Keep active jobs plus the last N terminal ones, or drop a job once History has recorded it.
- Status: fixed

### Issue 61 -- Severity: suggestion
- File: src/minimax_studio/worker/history.py
- Description: `index.jsonl` is append-only and fully rewritten on delete, not atomically. `list_history` reads the whole file, then slices 200. A crash mid-rewrite loses the index (takes still exist as folders).
- Suggestion: Write temp + `os.replace`. Bound the list, or rebuild from `history/*/meta.json` when the index is gone.
- Status: fixed

### Issue 62 -- Severity: bug
- File: src/minimax_studio/config.py, worker/presets.py, worker/history.py
- Description: Train/dataset *reads* tolerate `JSONDecodeError`. `load_config`, `list_presets`, and `get_entry` do not — a crash mid-save makes the next launch die, and truncated `meta.json` 500s History / add-from-history. Same class as issue 15 (`validation.json`).
- Suggestion: Swallow `OSError` / `JSONDecodeError` on those reads (empty config / no presets / missing entry). Prefer temp + replace on write.
- Status: fixed

### Issue 63 -- Severity: bug
- File: src/minimax_studio/worker/backends/h3.py
- Description: `_apply_loras` TypeError path (no `adapter_name`) loads the first LoRA, `break`s, never stacks LoRA 2, then skips `set_adapters` (`names = []`). `set_adapters` errors are swallowed, so strength can silently become 1.0 even on the success path.
- Suggestion: If `adapter_name` is unsupported, say so and load one, or fail the job. Do not swallow `set_adapters` — a selected adapter that does not apply is issue 34 again.
- Status: fixed

### Issue 64 -- Severity: bug
- File: src/minimax_studio/worker/backends/h3.py, music.py
- Description: CUDA generate `TypeError` fallback calls `pipe(**kwargs)` with no step callback, so Cancel does nothing until sampling finishes. The callback path already raises `CancelledError`.
- Suggestion: Keep the cancel callback on the fallback, or poll `is_cancelled` around the call and raise `CancelledError` if it flipped.
- Status: fixed

### Issue 65 -- Severity: suggestion
- File: src/minimax_studio/worker/loras.py
- Description: `list_loras` keys on filename only. Two different `pytorch_lora_weights.safetensors` in different folders collapse to one. Install prefixes the run name now (issue 29), so this is leftover Comfy dumps and hand copies.
- Suggestion: Key on resolved path, or skip a file whose name collided and say so.
- Status: fixed

### Issue 66 -- Severity: suggestion
- File: src/minimax_studio/worker_client.py
- Description: Every call opens a new `httpx.Client` (no pooling). The 500 ms jobs tick is two GETs × handshake. The inspector route check and Build polls add more.
- Suggestion: One client per `WorkerClient` (or per GUI thread). Close it when the window dies.
- Status: fixed

### Issue 67 -- Severity: suggestion
- File: src/minimax_studio/worker/train_runs.py
- Description: `export_run` copies file-by-file with no cancel and no cleanup on failure. A failed export leaves a dest folder that the next export refuses to overwrite (`already exists — move or delete it first`).
- Suggestion: Copy into a temp name, then rename. On failure, remove the partial dest.
- Status: fixed

Still open: none from this review list. Next metal session should confirm the H3 SimpleTuner keys on a real 24 GB run (`H3_UNVERIFIED_KEYS` warning stays until then).

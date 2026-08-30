from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget

from minimax_studio import __version__

HELP = """
<h2>MiniMax Studio {version}</h2>
<p>Point-and-click local studio for <b>MiniMax H3</b> and <b>MiniMax-Music3</b>.
Weights are downloaded after install. This app does not ship model files.</p>

<h3>Models</h3>
<ul>
<li><b>MiniMax-Music3</b> — MiniMax-Music3 Community License. Commercial use allowed
with UI credit. Over USD 20M/year needs MiniMax authorization. No geographic carve-out.</li>
<li><b>MiniMax H3</b> — MiniMax H3 Community License. Open weights are not authorized
in the US, EU, UK, or South Korea unless MiniMax grants a separate license.
The MiniMax hosted API remains globally available.</li>
</ul>

<p>Version is also shown in the window title and on <code>GET /health</code>.</p>

<h3>Local LLM</h3>
<p>Enhance caption / prompt talks to an OpenAI-compatible server
(default <code>http://127.0.0.1:8080/v1</code>). Thinking is set to
<b>medium</b> (<code>reasoning_effort=medium</code>, 512-token budget).</p>

<h3>Backends</h3>
<p>Inspector <b>Fast</b> on Generate Video needs the Turbo LoRA pack (Models).
Without it, Generate tells you to download Turbo or switch back to Quality.
Fast uses 8 steps (4 in Ref2VA). Fast is disabled on Generate Music.
<b>LoRA 2</b> stacks after LoRA 1; Fast still prepends Turbo.</p>
Video length chips set 5 / 8 / 15 seconds. A failed generate offers Retry.
History can copy the prompt.</p>
<p>Inspector: <b>Auto</b> prefers Comfy-Org INT8 via a running ComfyUI when the
selected GPU has under 24 GB VRAM, or when PyTorch is missing from the Studio
venv. Official diffusers is used when VRAM is large enough and torch is installed.
Then the MiniMax API if a key is set.</p>
<p>The Comfy-Org INT8 files use Comfy <code>convrot</code> kernels. Diffusers cannot
load them. Point Settings at your ComfyUI models folder (Studio auto-detects
<code>~/ai/ComfyUI/models</code>, <code>~/models</code>, and Comfy
<code>extra_model_paths.yaml</code>) and keep ComfyUI at
<code>http://127.0.0.1:8188</code> to generate from those weights.</p>
<p>Reference mode on INT8 uses the Ref2VA checkpoint via ComfyUI. Name files in the
prompt as <code>&lt;Picture 1&gt;</code> / <code>&lt;Video 1&gt;</code> /
<code>&lt;Audio 1&gt;</code> in the order you added them.</p>
<p>Inspector <b>Sage</b> attention patches the Comfy graph (KJNodes
<code>PathchSageAttentionKJ</code>). Official diffusers still uses PyTorch attention.</p>
<p>MiniMax-Music3 INT8 works the same way: if those files are in a Comfy models
folder and ComfyUI is running, Generate Music can submit to it.</p>
<p>Music LoRAs load on CUDA and ComfyUI. The MiniMax Music API and MLX cannot
apply them — Generate refuses rather than dropping the adapter. Blank lyrics
on the API is not an instrumental: the endpoint writes lyrics from the prompt
(<code>lyrics_optimizer</code>).</p>
<p>Inspector <b>CUDA GPU</b> is for in-process diffusers only. ComfyUI uses the
device it was launched with (<code>--default-device</code>). Generate runs a
preflight check and will tell you if ComfyUI or PyTorch is missing.</p>
<p>Generate Video accepts drag-and-drop on frame/reference rows. Quality is
Preview (~480px) or Native 768. Structure inserts a local shot-list scaffold
(Enhance/Context-IR still do the rewrite). Models Remove only deletes Studio’s
copy, never a ComfyUI folder.</p>
<p>First launch picks an output folder, then a welcome screen lists GPU and any
packs already on disk. H3 duration in the inspector snaps to the model’s
5–15 s / 24 fps frame grid.</p>
<p>The status bar shows the live generate job (Cancel stops it). History
<b>Show in folder</b> opens the take in the file manager. File menu opens the
output and models folders. H3 local generate wants <code>ffmpeg</code> on PATH
for MP4 mux.</p>
<p>Go menu: Ctrl+1…7 switches pages (Build pages: Ctrl+Shift+D / T / A),
Ctrl+Enter generates on Video/Music.
<b>Start ComfyUI</b> (Welcome, Settings, or Go) launches a detected
<code>main.py</code> install as a separate process and waits until it answers
— extra args in Settings (for example
<code>--listen 0.0.0.0 --default-device 1</code>). History can
filter by Video/Music and prompt text.</p>
<p>Generate asks before continuing if preflight has warnings (for example
missing ffmpeg). The status bar also shows an active model download.
Models and History can both reveal the file on disk.</p>
<p>One job runs at a time. Further Generate clicks queue (up to 8), including
from the same page. Cancel on a queued take drops it; Cancel on the status
bar stops the running job. Presets can be filtered like History.</p>
<p>Settings can store API tokens in the OS keychain instead of
<code>config.json</code> when the optional <code>keyring</code> package is
installed.</p>

<h3>Build — datasets and LoRA training (experimental)</h3>
<p><b>Datasets</b> (Ctrl+Shift+D) keeps a training set in SimpleTuner’s own
layout — <code>track.wav</code> plus a <code>track.txt</code> caption and
optionally <code>track.lyrics</code> for Music, or <code>shot.png</code> stills
and short <code>shot.mp4</code> clips for H3. Imports <b>copy</b> files, so
cleaning up your originals never guts a dataset; <b>Add from History</b> brings
a good take’s caption and lyrics with it. <b>Validate</b> names the problem on
every entry that can’t train — a missing caption, a 12.0s clip over the 8s cap,
a 128×96 still under the 256 px floor — and starting a run refuses a dataset
that doesn’t validate. Without ffmpeg it says what it could not measure instead
of blaming your clips.</p>
<p>An <b>H3 dataset</b> trains stills and short clips. Clips are capped at 8
seconds because dialogue footage waits for proof it does not wreck the audio
heads. <b>Audio+video (av) mode</b> is a checkbox, never the default: it pays
extra VRAM and disk, needs an audio stream in every clip, and cannot include
stills at all — refusing it names the clips that made it impossible.</p>
<p><b>Train LoRA</b> (Ctrl+Shift+T) writes SimpleTuner’s two config files and launches
the pinned <code>simpletuner</code> as <b>its own process</b>: closing Studio
does not stop a run, and the page reattaches to it from the run folder.
Cancel signals the process group; checkpoints already written stay, and a run
resumes from the last one. <b>Install adapter</b> copies the newest checkpoint
into the LoRA folder — Music adapters in the Generate Music picker (try ~0.8
strength), H3 adapters in Generate Video.</p>
<p>A long run is <b>Storage…</b> on the Train page, and it names the gigabytes
before anything goes. <b>Prune checkpoints</b> keeps the newest N <i>plus</i>
every checkpoint you installed as an adapter; <b>Clear caches</b> frees the VAE
and text-embedding caches, which the next run rebuilds. <b>Resume from
selected</b> continues a stopped run from any checkpoint it wrote — same folder,
same caches, new process. Nothing destructive is offered at all while a run is
live, and <b>Export…</b> / <b>Import run folder</b> move a run (weights, config,
log — not its caches) to another disk or another machine.</p>
<p>Training is <b>experimental</b>: CUDA only, Python 3.12 with
<code>pip install -e ".[train]"</code>, plus the weights the run needs — the
<b>Music 3 Training Encoder</b> pack for Music LoRAs, the <b>H3 diffusers</b>
weights for video ones (the Comfy packs will not do; the trainer reads the
diffusers folder). 24 GB is the floor for Music. H3 offers 24 GB with RamTorch
CPU-offload, then 32, 48 and 80 GB, and the preset list changes with the dataset
you pick — the two trainers never share a run. Preflight names the VRAM, packs,
active jobs and free cache disk before anything starts; no mystery OOM.</p>
<p><b>Said plainly:</b> H3 LoRA training has not run on this build yet. Its
config keys are written from SimpleTuner’s documentation rather than its output,
and preflight says so on every H3 preset until a real run confirms them — watch
the first minutes and report anything odd. Auditioning is still Music-only: an
H3 adapter loads in the picker and works on Generate Video, but it has no
one-click preview.</p>
<p>A live training run warns you on Generate (and shows in the status bar) —
one GPU, two hungry things.</p>

<h3>Adapters — provenance and auditioning</h3>
<p><b>Adapters</b> (Ctrl+Shift+A) lists every LoRA the picker can load and says
where each came from: <b>trained here</b> (dataset name, clip count, a hash of
those clip names and sizes, run id, preset, rank, steps, the pinned
SimpleTuner version), <b>imported</b> (you brought the file), or <b>found on
disk</b> — files Studio never registered but the picker loads anyway, which is
worth saying out loud rather than hiding.</p>
<p><b>Audition</b> queues one 30-second render at 0.8 strength with the caption
that clip set used most, so the answer to “was that worth three hours?” arrives
in the History list badged as an audition — and <b>Restore to Generate</b>
works on it like any other take. It refuses honestly when there is nothing to
sing with: a hand-imported adapter or a deleted dataset has no caption, so type
a prompt.</p>
<p><b>Forget</b> drops Studio’s provenance row and leaves the
<code>.safetensors</code> on disk and in the picker. Deleting a file leaves the
row behind, flagged “file is gone”, because the story of an adapter is still
worth reading after the file is not.</p>
"""


class HelpPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Help")
        title.setObjectName("pageTitle")
        body = QTextBrowser()
        body.setOpenExternalLinks(True)
        body.setHtml(HELP.format(version=__version__))
        layout.addWidget(title)
        layout.addWidget(body, 1)

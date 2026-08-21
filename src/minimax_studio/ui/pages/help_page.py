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
<p>Go menu: Ctrl+1…7 switches pages, Ctrl+Enter generates on Video/Music.
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

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
<p>Inspector: <b>Auto</b> prefers official diffusers on this GPU, then Comfy-Org INT8
via a running ComfyUI, then the MiniMax API if a key is set.</p>
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
<p>First launch picks an output folder, then a welcome screen lists GPU and any
packs already on disk. H3 duration in the inspector snaps to the model’s
5–15 s / 24 fps frame grid.</p>
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

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget

HELP = """
<h2>MiniMax Studio</h2>
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

<h3>Local LLM</h3>
<p>Enhance caption / prompt talks to an OpenAI-compatible server
(default <code>http://127.0.0.1:8080/v1</code>). Thinking is set to
<b>medium</b> (<code>reasoning_effort=medium</code>, 512-token budget).</p>

<h3>Backends</h3>
<p>Inspector: Auto prefers a local pack on this GPU, then the MiniMax API if a key is set.</p>
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
        body.setHtml(HELP)
        layout.addWidget(title)
        layout.addWidget(body, 1)

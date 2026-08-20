from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


def placeholder_page(title: str, subtitle: str) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(28, 24, 28, 24)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("pageTitle")
    sub = QLabel(subtitle)
    sub.setObjectName("pageSubtitle")
    sub.setWordWrap(True)
    layout.addWidget(heading)
    layout.addWidget(sub)
    layout.addStretch(1)
    page.setProperty("pageTitle", title)
    return page


def build_pages() -> dict[str, QWidget]:
    return {
        "video": placeholder_page(
            "Generate Video",
            "MiniMax H3 — text, first/last frame, or references. Point and click. Coming next after Music 3.",
        ),
        "music": placeholder_page(
            "Generate Music",
            "MiniMax-Music3 — caption + lyrics. This is the first generate slice once the downloader lands.",
        ),
        "history": placeholder_page("History", "Takes, playback, restore-to-generate, export."),
        "presets": placeholder_page("Presets", "Saved generation settings."),
        "models": placeholder_page(
            "Models",
            "Download packs after install. Weights are never bundled with the app.",
        ),
        "settings": placeholder_page(
            "Settings",
            "Output folder, Hugging Face token, MiniMax API key, GPU.",
        ),
    }

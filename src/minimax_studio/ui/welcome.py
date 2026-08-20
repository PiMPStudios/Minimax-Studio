from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from minimax_studio.worker_client import WorkerClient


class WelcomeDialog(QDialog):
    """One-shot after first launch: GPU, packs already on disk, Comfy status."""

    def __init__(self, client: WorkerClient) -> None:
        super().__init__()
        self.setWindowTitle("MiniMax Studio")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        title = QLabel("You're set up")
        title.setObjectName("pageTitle")
        body = QLabel(_welcome_text(client))
        body.setObjectName("pageSubtitle")
        body.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Open studio")
        buttons.accepted.connect(self.accept)
        recommended = QPushButton("Download recommended")
        recommended.setToolTip("Pull only the consumer packs for this GPU, not the 130 GB official H3 tree.")
        recommended.clicked.connect(lambda: _download_recommended(client, body, recommended))
        buttons.addButton(recommended, QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addWidget(buttons)


def _welcome_text(client: WorkerClient) -> str:
    lines: list[str] = []
    probe: dict[str, Any] = {}
    ping: dict[str, Any] = {}
    try:
        probe = client.probe()
    except Exception:
        probe = {}
    try:
        ping = client.ping()
    except Exception:
        ping = {}

    gpus = probe.get("gpus") or []
    if gpus:
        names = " + ".join(f"{item.get('name')} ({item.get('vram_gb')} GB)" for item in gpus)
        lines.append(f"GPU: {names}.")
        if not probe.get("torch_available"):
            lines.append(
                "PyTorch is not installed in the Studio venv, so in-process "
                "diffusers generate is unavailable. Comfy-Org INT8 still works "
                "when ComfyUI is running."
            )
    elif probe.get("cuda"):
        lines.append(
            f"GPU: {probe.get('cuda_name') or 'CUDA'} ({probe.get('vram_gb')} GB)."
        )
    elif probe.get("apple_silicon"):
        lines.append("Apple Silicon. Local Music 3 is MLX; local H3 is experimental.")
    else:
        lines.append("No CUDA GPU detected. Use the MiniMax API or a machine with NVIDIA.")
    if probe.get("ram_gb"):
        lines.append(f"RAM: {probe.get('ram_gb')} GB.")
    if probe.get("sageattention"):
        lines.append("SageAttention is installed (Inspector → Attention → Sage, Comfy path).")

    titles = probe.get("packs_ready_titles") or []
    if titles:
        lines.append("")
        lines.append("Already on disk:")
        for title in titles:
            lines.append(f"• {title}")
        comfy_n = probe.get("packs_from_comfy") or 0
        if comfy_n:
            lines.append(
                f"{comfy_n} of those were found in a ComfyUI models folder — no re-download needed."
            )
    else:
        lines.append("")
        lines.append(
            "No packs on disk yet. Open Models and download only what you need "
            "(Comfy-Org INT8 is the consumer CUDA default, not the 130 GB official H3 tree)."
        )

    comfy = ping.get("comfy") if isinstance(ping.get("comfy"), dict) else {}
    lines.append("")
    if comfy.get("ok"):
        lines.append("ComfyUI is reachable. Auto generate can use INT8 packs through it.")
    else:
        lines.append(
            "ComfyUI is not running. Start it at http://127.0.0.1:8188 to generate from "
            "Comfy-Org INT8 files, or download official diffusers packs for in-process generate."
        )
    return "\n".join(lines)


def _download_recommended(client: WorkerClient, body: QLabel, button: QPushButton) -> None:
    try:
        packs = client.list_packs()
    except Exception as exc:
        body.setText(body.text() + f"\n\nCould not list packs: {exc}")
        return
    started = 0
    for pack in packs:
        if not pack.get("recommended") or pack.get("ready"):
            continue
        try:
            client.start_download(pack["id"])
            started += 1
        except Exception:
            continue
    button.setEnabled(False)
    if started:
        body.setText(
            body.text()
            + f"\n\nStarted {started} recommended download{'s' if started != 1 else ''}. "
            "Watch progress on the Models page."
        )
    else:
        body.setText(body.text() + "\n\nNothing to download — recommended packs are already ready.")

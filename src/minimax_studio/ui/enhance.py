from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from minimax_studio.worker_client import WorkerClient


def on_main(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a callback so it runs on the GUI thread even when emitted from a
    worker thread (plain functions connected to signals run on the emitter)."""

    def wrapper(*args: Any) -> None:
        QTimer.singleShot(0, lambda: fn(*args))

    return wrapper


class EnhanceWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, client: WorkerClient, kind: str, text: str, extra: str = "") -> None:
        super().__init__()
        self._client = client
        self._kind = kind
        self._text = text
        self._extra = extra

    def run(self) -> None:
        try:
            payload = self._client.enhance(self._kind, self._text, self._extra)
            self.finished.emit(str(payload.get("text") or ""))
        except Exception as exc:
            self.failed.emit(str(exc))


def start_enhance(
    parent: QObject,
    client: WorkerClient,
    kind: str,
    text: str,
    extra: str,
    on_done,
    on_fail,
) -> QThread:
    thread = QThread(parent)
    worker = EnhanceWorker(client, kind, text, extra)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(on_main(on_done))
    worker.failed.connect(on_main(on_fail))
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    # Strong ref: signal connections only weakly reference the worker.
    thread._worker = worker
    thread.start()
    parent._enhance_thread = thread  # keep a ref
    return thread

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from minimax_studio.worker_client import WorkerClient


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
    worker.finished.connect(on_done)
    worker.failed.connect(on_fail)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    parent._enhance_thread = thread  # keep a ref
    return thread

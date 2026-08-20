from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

from minimax_studio.worker_client import WorkerClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minimax-studio")
    parser.add_argument(
        "--worker-only",
        action="store_true",
        help="Run the FastAPI worker in this process (no Qt window).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    args = parser.parse_args(argv)

    port = args.port or _free_port(args.host)
    if args.worker_only:
        return worker_main_with_port(args.host, port)

    worker = _start_worker(args.host, port)
    client = WorkerClient(f"http://{args.host}:{port}")
    try:
        _wait_for_health(client)
        return _run_ui(client)
    finally:
        _stop_worker(worker)


def worker_main_with_port(host: str, port: int) -> int:
    import uvicorn

    uvicorn.run(
        "minimax_studio.worker.server:app",
        host=host,
        port=port,
        log_level="info",
    )
    return 0


def _run_ui(client: WorkerClient) -> int:
    from PySide6.QtWidgets import QApplication

    from minimax_studio.ui.main_window import MainWindow
    from minimax_studio.ui.theme import apply_theme

    app = QApplication(sys.argv)
    app.setApplicationName("MiniMax Studio")
    apply_theme(app)
    window = MainWindow(client)
    window.show()
    return app.exec()


def _start_worker(host: str, port: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "minimax_studio.worker",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
    )


def _stop_worker(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _wait_for_health(client: WorkerClient, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            payload = client.health()
            if payload.get("ok"):
                return
        except Exception as exc:  # noqa: BLE001 — retry until timeout
            last = exc
        time.sleep(0.15)
    raise RuntimeError(f"worker did not become healthy: {last}")


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main())

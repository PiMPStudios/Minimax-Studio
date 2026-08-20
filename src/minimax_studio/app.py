from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from minimax_studio.config import (
    AppConfig,
    default_config_path,
    load_config,
    save_config,
)
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
    return _run_ui(args.host, port)


def worker_main_with_port(host: str, port: int) -> int:
    import uvicorn

    uvicorn.run(
        "minimax_studio.worker.server:app",
        host=host,
        port=port,
        log_level="info",
    )
    return 0


def _run_ui(host: str, port: int) -> int:
    from PySide6.QtWidgets import QApplication, QDialog

    from minimax_studio.ui.first_launch import FirstLaunchDialog
    from minimax_studio.ui.main_window import MainWindow
    from minimax_studio.ui.theme import apply_theme
    from minimax_studio.ui.welcome import WelcomeDialog

    app = QApplication(sys.argv)
    app.setApplicationName("MiniMax Studio")
    apply_theme(app)

    config = load_config()
    if not config.has_output_dir():
        dialog = FirstLaunchDialog()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return 0
        config.output_dir = dialog.output_dir
        config.models_dir = str(Path(dialog.output_dir) / "models")
        save_config(config)
    config.ensure_dirs()

    worker = _start_worker(host, port)
    client = WorkerClient(f"http://{host}:{port}")
    try:
        _wait_for_health(client)
        window = MainWindow(client, config)
        if not config.welcome_seen:
            WelcomeDialog(client).exec()
            config.welcome_seen = True
            save_config(config)
        window.show()
        return app.exec()
    finally:
        _stop_worker(worker)


def _start_worker(host: str, port: int) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["MINIMAX_STUDIO_CONFIG"] = str(default_config_path())
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
        env=env,
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
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.15)
    raise RuntimeError(f"worker did not become healthy: {last}")


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main())

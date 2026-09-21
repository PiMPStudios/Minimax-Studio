"""The agent read surface, driven over real HTTP against a real worker.

M0 of the MCP thread. Everything an MCP server needs to be except the MCP
wrapper: an out-of-process client, authenticated with the same per-launch
secret the GUI hands its worker, reading the same JSON the GUI reads. What this
deliberately does *not* solve is discovery — see
test_missing_endpoint_is_exit_two_naming_the_env_var, which is the missing
handoff written down as an assertion.

The worker runs as a subprocess here (not TestClient) on purpose: an
in-process client proves the route works, not that a second process can reach
it, and that difference is the whole thread.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from minimax_studio import __version__, agent_handoff
from minimax_studio.agent_cli import NO_ENDPOINT, TOKEN_ENV, URL_ENV, main
from minimax_studio.config import load_config, save_config

TOKEN = "agent-cli-spike-token"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def tokened_worker(studio_home: Path, monkeypatch: pytest.MonkeyPatch):
    """A live worker on a fixed port, gated by a token we control.

    ``--worker-only`` generates no token of its own; the middleware reads
    MINIMAX_STUDIO_WORKER_TOKEN from the environment, which is exactly how the
    GUI gates its child (app.py) — so setting it here reproduces a normal
    launch rather than the open dev mode.
    """
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    port = _free_port()
    log = studio_home / "worker.log"
    with log.open("wb") as handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "minimax_studio",
                "--worker-only",
                "--port",
                str(port),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + 60.0
        while time.time() < deadline:
            try:
                # 401 means the process is up and the gate is armed: both are
                # facts this suite depends on before it calls anything.
                if httpx.get(f"{url}/health", timeout=1.0).status_code in (200, 401):
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            proc.terminate()
            pytest.fail(f"worker never answered; {log.read_text()[-2000:]}")
        try:
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover - CI safety only
                proc.kill()
                proc.wait(timeout=5)


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def advertised_worker(studio_home: Path):
    """A launch with “Allow agent access” switched on.

    The URL and token go into the *child's* environment only — exactly what
    app.py does — so nothing in this process has them. Anything that connects
    has to find them the way an outside agent would: the handoff file.
    """
    config = load_config()
    config.allow_agent_access = True
    save_config(config)
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    log = studio_home / "worker.log"
    with log.open("wb") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "minimax_studio", "--worker-only", "--port", str(port)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                agent_handoff.TOKEN_ENV: TOKEN,
                agent_handoff.URL_ENV: url,
            },
        )
        deadline = time.time() + 60.0
        handoff = agent_handoff.handoff_path()
        found = False
        while time.time() < deadline:
            try:
                live = httpx.get(
                    f"{url}/health",
                    headers={"X-Minimax-Studio-Token": TOKEN},
                    timeout=1.0,
                ).status_code
            except httpx.HTTPError:
                live = 0
            if live == 200 and handoff.is_file():
                found = True
                break
            time.sleep(0.2)
        if not found:
            proc.terminate()
            pytest.fail(f"worker or handoff never appeared; {log.read_text()[-2000:]}")
        try:
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover - CI safety only
                proc.kill()
                proc.wait(timeout=5)


def test_the_cli_finds_the_worker_from_the_handoff_alone(
    advertised_worker: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """M2's whole claim: no flags, no exported port, no exported secret."""
    monkeypatch.delenv(URL_ENV, raising=False)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    # Not a formality: if any fixture ever leaks either value into this
    # process, this test stops proving discovery and starts proving plumbing.
    assert URL_ENV not in os.environ and TOKEN_ENV not in os.environ
    code, out, err = _run(["status"], capsys)
    assert code == 0, err
    payload = json.loads(out)
    assert payload["connected_via"] == "handoff"
    assert payload["worker"]["version"] == __version__
    assert TOKEN not in out


def test_a_handoff_from_a_quit_studio_says_so(
    studio_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The likeliest real failure: the switch is on, the launch is gone."""
    monkeypatch.delenv(URL_ENV, raising=False)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    path = agent_handoff.sync(True, url="http://127.0.0.1:1", token="stale-token")
    assert path is not None
    code, out, err = _run(["--timeout", "2", "status"], capsys)
    assert code == 1
    assert out == ""
    assert "handoff" in err.lower()
    assert str(path) in err


def test_the_switch_being_off_is_what_makes_the_last_two_work(
    studio_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the premise of the two tests above: with the switch off, an
    outside process has nothing to find — which is the security claim."""
    monkeypatch.delenv(URL_ENV, raising=False)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    assert not agent_handoff.handoff_path().exists()
    assert agent_handoff.read() is None


def test_status_answers_over_http_with_the_launch_token(
    tokened_worker: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(URL_ENV, tokened_worker)
    code, out, err = _run(["status"], capsys)
    assert code == 0, err
    payload = json.loads(out)
    assert payload["worker"]["version"] == __version__
    assert payload["worker"]["service"] == "minimax-studio-worker"
    # studio_home is empty, so "can I generate right now" must be honest.
    assert payload["hardware"]["packs_ready"] == []
    assert payload["can_generate_now"] is False
    assert payload["studio_version"] == __version__
    # The whole point of the token is that it does not travel into output.
    assert TOKEN not in out
    assert TOKEN not in err


def test_packs_names_the_catalog_without_downloading_anything(
    tokened_worker: str,
    studio_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv(URL_ENV, tokened_worker)
    code, out, err = _run(["packs"], capsys)
    assert code == 0, err
    payload = json.loads(out)
    ids = {row["id"] for row in payload["packs"]}
    assert {"music3-cuda", "h3-fl2va", "music3-mlx"} <= ids
    assert all(row["ready"] is False for row in payload["packs"])
    assert all("approx_gb" in row for row in payload["packs"])
    assert isinstance(payload["adapters"], list)
    # ensure_dirs() makes models/ on first boot, so emptiness — not existence —
    # is what proves a read stayed a read.
    models = studio_home / "models"
    assert models.is_dir() and not any(models.iterdir())


def test_the_token_never_reaches_stdout_even_when_it_is_wrong(
    tokened_worker: str, capsys: pytest.CaptureFixture
) -> None:
    """A 401 is the interesting leak: the failure copy must not echo the secret."""
    code, out, err = _run(
        ["--url", tokened_worker, "--token", "not-the-launch-token", "status"], capsys
    )
    assert code == 1
    assert out == ""
    assert "not-the-launch-token" not in err
    assert "token" in err.lower()


def test_unreachable_worker_names_the_fix_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(URL_ENV, "http://127.0.0.1:1")  # nothing listens on port 1
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    code, out, err = _run(["--timeout", "2", "status"], capsys)
    assert code == 1
    assert "scripts/run.sh" in err


def test_missing_endpoint_is_exit_two_naming_the_env_var(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """M2's job is to make this path impossible; until then it must be legible."""
    monkeypatch.delenv(URL_ENV, raising=False)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    code, out, err = _run(["status"], capsys)
    assert code == 2
    assert out == ""
    assert URL_ENV in err
    assert URL_ENV in NO_ENDPOINT


def test_the_agent_surface_imports_neither_qt_nor_torch() -> None:
    """AGENTS.md: anything heavy stays behind a lazy import in worker/backends.

    The agent CLI is the entry point an agent touches, so it has to start fast
    and never drag inference into the picture.
    """
    probe = (
        "import sys, minimax_studio.agent_cli as m; "
        "print(any(n.split('.')[0] in {'PySide6','torch','diffusers','mlx','simpletuner'} "
        "for n in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", proc.stdout

"""The agent handoff: written only while the switch is on, gone otherwise.

The worker keeps an ephemeral port and an in-memory token on purpose. This is
the one deliberate hole in that, so the tests are mostly about closing it: off
must leave nothing on disk, un-toggling must remove what toggling created,
quitting must remove it too, and a tokenless dev worker must never advertise
itself at all.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimax_studio import __version__, agent_handoff
from minimax_studio.config import AppConfig, load_config, save_config
from minimax_studio.worker.runtime import runtime
from minimax_studio.worker.server import app

URL = "http://127.0.0.1:54321"
TOKEN = "launch-token-for-handoff-tests"


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_off_leaves_no_file_at_all(studio_home: Path) -> None:
    """The point of only-while-on: nothing stale can outlive the intent."""
    assert agent_handoff.sync(False, url=URL, token=TOKEN) is None
    assert not agent_handoff.handoff_path().exists()


def test_on_writes_the_launch_coordinates(studio_home: Path) -> None:
    path = agent_handoff.sync(True, url=URL, token=TOKEN)
    assert path is not None and path.is_file()
    data = _payload(path)
    assert data["url"] == URL
    assert data["token"] == TOKEN
    assert data["service"] == agent_handoff.SERVICE
    assert data["worker_version"] == __version__
    assert data["pid"] == os.getpid()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_the_handoff_is_created_private_not_opened_then_tightened(
    studio_home: Path,
) -> None:
    """A chmod after the write is a window; the mode must be set at creation."""
    path = agent_handoff.sync(True, url=URL, token=TOKEN)
    assert path is not None
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # No world-readable leftovers from the atomic replace either.
    assert not any(p.name.endswith(".tmp") for p in path.parent.iterdir())


def test_a_tokenless_dev_worker_never_advertises_itself(
    studio_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--worker-only is open to every local process already; publishing that
    next to config.json would make a dev run look like a normal install."""
    monkeypatch.delenv(agent_handoff.TOKEN_ENV, raising=False)
    monkeypatch.setenv(agent_handoff.URL_ENV, URL)
    assert agent_handoff.sync(True) is None
    assert not agent_handoff.handoff_path().exists()


def test_read_rejects_anything_that_is_not_ours(studio_home: Path) -> None:
    path = agent_handoff.handoff_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    for junk in (
        "not json at all",
        json.dumps(["a", "list"]),
        json.dumps({"url": URL, "token": "t"}),  # wrong service
        json.dumps({"service": agent_handoff.SERVICE, "url": URL}),  # no token
    ):
        path.write_text(junk, encoding="utf-8")
        assert agent_handoff.read() is None
    path.unlink()
    assert agent_handoff.read() is None


def test_saving_settings_turns_the_file_on_and_off(
    studio_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(agent_handoff.URL_ENV, URL)
    monkeypatch.setenv(agent_handoff.TOKEN_ENV, TOKEN)
    # Setting the token env above arms the real gate for this in-process app,
    # so the client has to carry it the way the GUI does.
    client = TestClient(app, headers={"X-Minimax-Studio-Token": TOKEN})

    assert not agent_handoff.handoff_path().exists()
    saved = client.post("/settings", json={"allow_agent_access": True})
    assert saved.status_code == 200
    assert saved.json()["allow_agent_access"] is True
    path = agent_handoff.handoff_path()
    assert path.is_file()
    assert _payload(path)["url"] == URL

    saved = client.post("/settings", json={"allow_agent_access": False})
    assert saved.json()["allow_agent_access"] is False
    assert not path.exists()
    # And it stayed off across a config reload, not just in memory.
    assert load_config().allow_agent_access is False


def test_startup_honours_a_saved_switch_and_shutdown_clears_it(
    studio_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The switch is a setting, so a normal Studio launch must respect it
    without anyone opening the Settings page."""
    config = load_config()
    config.allow_agent_access = True
    save_config(config)
    # runtime.config is read at import; the startup hook reads the singleton.
    # Forgetting this reload is the flake AGENTS.md warns about.
    runtime.reload_config()
    monkeypatch.setenv(agent_handoff.URL_ENV, URL)
    monkeypatch.setenv(agent_handoff.TOKEN_ENV, TOKEN)

    path = agent_handoff.handoff_path()
    with TestClient(
        app, headers={"X-Minimax-Studio-Token": TOKEN}
    ) as client:  # entering fires the startup hook
        assert path.is_file()
        assert _payload(path)["token"] == TOKEN
        assert client.get("/health").json()["ok"] is True
    assert not path.exists()


def test_the_settings_switch_is_what_sends_the_flag(studio_home: Path) -> None:
    """No one hand-assembles this payload, so the checkbox itself is the
    contract: unchecked must send False, not omit the key and leave it on."""
    from PySide6.QtWidgets import QApplication

    from minimax_studio.ui.pages.settings_page import SettingsPage

    QApplication.instance() or QApplication([])

    class Stub:
        def __init__(self) -> None:
            self.payload: dict = {}

        def put_settings(self, payload: dict) -> dict:
            self.payload = payload
            return {**AppConfig().model_dump(), **payload}

        def probe(self) -> dict:
            return {"gpus": [], "cuda": False, "apple_silicon": False}

        def ping(self) -> dict:
            return {"minimax": {}, "llm": {}, "comfy": {}}

        def comfy_status(self) -> dict:
            return {"root": None}

    worker = Stub()
    config = AppConfig(output_dir=str(studio_home))
    page = SettingsPage(worker, config)  # type: ignore[arg-type]

    # Drained after every save, not just at the end: each _save starts a
    # QThread and assigns it to page._ping_thread, so draining once would
    # leave the first thread running while its QThread object is replaced and
    # garbage-collected — the failure test_main_window._drain_window exists to
    # prevent, and the kind that surfaces minutes later in someone else's test.
    page.allow_agent.setChecked(True)
    assert page._save() is True
    from tests.dialogs import wait_background

    wait_background(page)
    assert worker.payload["allow_agent_access"] is True

    page.allow_agent.setChecked(False)
    assert page._save() is True
    wait_background(page)
    assert worker.payload["allow_agent_access"] is False


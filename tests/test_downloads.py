import time
from pathlib import Path

from minimax_studio.worker.catalog import PACKS
from minimax_studio.worker.downloads import delete_pack, get_download, start_download


def test_h3_train_pack_is_the_63gb_slice_not_the_transformer() -> None:
    pack = PACKS["h3-train"]
    assert pack.local_dir == "h3-diffusers"
    assert pack.repo_id == "MiniMaxAI/MiniMax-H3"
    assert 50 <= pack.approx_gb <= 70
    assert pack.allow_patterns is not None
    assert "audio_vae/*" in pack.allow_patterns
    assert "text_encoder/*" in pack.allow_patterns
    assert "tokenizer/*" in pack.allow_patterns
    assert not any(
        "transformer" in pattern and pattern.endswith("safetensors")
        for pattern in pack.allow_patterns
    )
    assert "audio_vae/diffusion_pytorch_model.safetensors" in pack.marker_files
    assert "text_encoder/model.safetensors.index.json" in pack.marker_files
    assert "text_encoder/model-00001-of-00014.safetensors" in pack.marker_files
    generate = PACKS["h3-diffusers-fl2va"]
    assert generate.local_dir == pack.local_dir
    assert (
        "transformer/diffusion_pytorch_model-00001-of-00014.safetensors"
        in generate.marker_files
    )


def test_h3_train_markers_do_not_make_the_generate_pack_ready(tmp_path: Path) -> None:
    from minimax_studio.worker.model_paths import pack_status

    models = tmp_path / "models"
    dest = models / "h3-diffusers"
    _touch(dest, PACKS["h3-train"].marker_files)
    train = pack_status(PACKS["h3-train"], models, extra_roots=[models])
    generate = pack_status(PACKS["h3-diffusers-fl2va"], models, extra_roots=[models])
    assert train["ready"] is True
    assert generate["ready"] is False


def test_full_official_fl2va_already_counts_as_h3_train(tmp_path: Path) -> None:
    from minimax_studio.worker.model_paths import pack_status

    models = tmp_path / "models"
    dest = models / "h3-diffusers"
    _touch(dest, PACKS["h3-train"].marker_files)
    _touch(dest, PACKS["h3-diffusers-fl2va"].marker_files)
    train = pack_status(PACKS["h3-train"], models, extra_roots=[models])
    generate = pack_status(PACKS["h3-diffusers-fl2va"], models, extra_roots=[models])
    assert train["ready"] is True
    assert generate["ready"] is True


def test_delete_h3_train_wipes_folder_when_generate_pack_is_not_installed(
    studio_home: Path,
) -> None:
    dest = studio_home / "models" / "h3-diffusers"
    _touch(dest, PACKS["h3-train"].marker_files)
    result = delete_pack("h3-train")
    assert not dest.exists()
    assert result["folder_kept"] is False
    assert result["removed_bytes"] > 0


def test_delete_h3_train_keeps_folder_when_official_fl2va_is_installed(
    studio_home: Path,
) -> None:
    dest = studio_home / "models" / "h3-diffusers"
    _touch(dest, PACKS["h3-train"].marker_files)
    _touch(dest, PACKS["h3-diffusers-fl2va"].marker_files)
    result = delete_pack("h3-train")
    assert dest.is_dir()
    assert result["folder_kept"] is True
    assert result["removed"] is False
    assert any("FL2VA" in title for title in result["kept_for"])
    for marker in PACKS["h3-train"].marker_files:
        assert (dest / marker).is_file()
    for marker in PACKS["h3-diffusers-fl2va"].marker_files:
        assert (dest / marker).is_file()


def test_list_packs_recommends_h3_train_from_24gb(
    studio_home: Path, monkeypatch
) -> None:
    from minimax_studio.worker.downloads import list_packs

    monkeypatch.setattr(
        "minimax_studio.worker.probe.probe",
        lambda: {
            "cuda": True,
            "vram_gb": 24.0,
            "ram_gb": 32.0,
            "apple_silicon": False,
        },
    )
    rows = {item["id"]: item for item in list_packs()}
    assert rows["h3-train"]["recommended"] is True
    assert rows["h3-diffusers-fl2va"]["recommended"] is False


def test_download_uses_injected_snapshot(studio_home: Path) -> None:
    def snapshot(**kwargs):
        dest = Path(kwargs["local_dir"])
        (dest / "modular_model_index.json").write_text("{}", encoding="utf-8")
        return str(dest)

    # force=True: CI runners have less free disk than the 63 GB this pack
    # claims; these tests exercise the snapshot flow, not the disk guard.
    record = start_download("music3-cuda", snapshot=snapshot, force=True)
    deadline = time.time() + 5
    while time.time() < deadline:
        current = get_download(record["id"])
        if current["status"] in {"done", "error"}:
            break
        time.sleep(0.05)
    current = get_download(record["id"])
    assert current["status"] == "done"
    assert (studio_home / "models" / "music3-cuda" / "modular_model_index.json").is_file()


def test_cancel_download_marks_cancelling(studio_home: Path) -> None:
    from minimax_studio.worker.downloads import cancel_download

    def snapshot(**kwargs):
        import time as _time

        _time.sleep(0.2)
        dest = Path(kwargs["local_dir"])
        (dest / "modular_model_index.json").write_text("{}", encoding="utf-8")
        return str(dest)

    # force=True: CI runners have less free disk than the 63 GB this pack
    # claims; these tests exercise the snapshot flow, not the disk guard.
    record = start_download("music3-cuda", snapshot=snapshot, force=True)
    rec = cancel_download(record["id"])
    assert rec["status"] in {"cancelling", "cancelled", "done"}


def test_delete_pack_removes_studio_copy(studio_home: Path) -> None:
    dest = studio_home / "models" / PACKS["music3-cuda"].local_dir
    dest.mkdir(parents=True)
    (dest / "modular_model_index.json").write_text("{}", encoding="utf-8")
    result = delete_pack("music3-cuda")
    assert result["ok"] is True
    assert not dest.exists()


def _touch(root: Path, markers: tuple[str, ...]) -> None:
    for marker in markers:
        path = root / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 4096)


def test_delete_pack_keeps_files_installed_packs_need(studio_home: Path) -> None:
    from minimax_studio.worker.downloads import delete_pack

    h3dir = studio_home / "models" / "h3-comfy"
    _touch(h3dir, PACKS["h3-fl2va"].marker_files)
    _touch(h3dir, PACKS["h3-ref2va"].marker_files)
    _touch(h3dir, PACKS["h3-turbo"].marker_files)
    result = delete_pack("h3-fl2va")
    assert result["folder_kept"] is True
    assert any("Ref2VA" in title for title in result["kept_for"])
    # Ref2VA requires FL2VA's encoder/VAE/UNET — all kept, and said so.
    for marker in PACKS["h3-fl2va"].marker_files:
        assert (h3dir / marker).is_file()
        assert marker in result["kept_files"]
    assert (h3dir / PACKS["h3-turbo"].marker_files[0]).is_file()


def test_delete_pack_removes_when_only_turbo_shares(studio_home: Path) -> None:
    from minimax_studio.worker.downloads import delete_pack

    h3dir = studio_home / "models" / "h3-comfy"
    _touch(h3dir, PACKS["h3-fl2va"].marker_files)
    _touch(h3dir, PACKS["h3-turbo"].marker_files)
    result = delete_pack("h3-fl2va")
    assert result["removed_bytes"] > 0
    assert not (h3dir / PACKS["h3-fl2va"].marker_files[0]).exists()
    assert (h3dir / PACKS["h3-turbo"].marker_files[0]).is_file()


def test_delete_pack_wipes_folder_when_nothing_shares(studio_home: Path) -> None:
    from minimax_studio.worker.downloads import delete_pack

    h3dir = studio_home / "models" / "h3-comfy"
    _touch(h3dir, PACKS["h3-fl2va"].marker_files)
    result = delete_pack("h3-fl2va")
    assert not h3dir.exists()
    assert result["removed_bytes"] > 0
    assert result["folder_kept"] is False


def test_delete_pack_delete_shared_wipes_shared_folder(
    studio_home: Path,
) -> None:
    from minimax_studio.worker.downloads import delete_pack

    h3dir = studio_home / "models" / "h3-comfy"
    _touch(h3dir, PACKS["h3-fl2va"].marker_files)
    _touch(h3dir, PACKS["h3-ref2va"].marker_files)
    first = delete_pack("h3-fl2va")
    assert first["folder_kept"] is True
    second = delete_pack("h3-fl2va", delete_shared=True)
    assert not h3dir.exists()
    assert second["deleted_shared"] is True


def test_start_download_blocks_without_free_space(
    studio_home: Path, monkeypatch
) -> None:
    import shutil
    from collections import namedtuple

    from minimax_studio.worker.downloads import start_download

    Usage = namedtuple("usage", "total used free")  # shutil.disk_usage shape
    tiny = Usage(1024**3, 0, int(0.2 * 1024**3))
    monkeypatch.setattr(shutil, "disk_usage", lambda path: tiny)
    from minimax_studio.errors import InsufficientDisk

    try:
        start_download("h3-fl2va")
        raise AssertionError("expected a free-disk error")
    except InsufficientDisk as exc:
        assert isinstance(exc, RuntimeError)
        assert "Not enough free disk" in str(exc)


def test_start_download_force_bypasses_free_space(
    studio_home: Path, monkeypatch
) -> None:
    import shutil
    import time
    from collections import namedtuple

    from minimax_studio.worker.downloads import get_download, start_download

    Usage = namedtuple("usage", "total used free")
    tiny = Usage(1024**3, 0, int(0.2 * 1024**3))
    monkeypatch.setattr(shutil, "disk_usage", lambda path: tiny)

    def fake_snapshot(repo_id, local_dir, token, allow_patterns, ignore_patterns):
        pack = PACKS["h3-turbo"]
        for marker in pack.marker_files:
            path = Path(local_dir) / marker
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        return local_dir

    record = start_download("h3-turbo", snapshot=fake_snapshot, force=True)
    deadline = time.time() + 5
    while time.time() < deadline:
        status = get_download(record["id"])["status"]
        if status in {"done", "error", "cancelled"}:
            break
        time.sleep(0.05)
    final = get_download(record["id"])
    assert final["status"] == "done", final.get("error")


class _HttpError:
    """Stand-in for httpx.Response in WorkerClient._raise tests."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.is_success = False
        self.status_code = status_code
        self._detail = detail

    def json(self) -> dict:
        return {"detail": self._detail}


def test_download_route_and_client_raise_insufficient_disk(
    studio_home: Path, monkeypatch
) -> None:
    """The hatch is a type + HTTP 507, not a substring of user-facing copy."""
    import shutil
    from collections import namedtuple

    from fastapi.testclient import TestClient

    from minimax_studio.errors import InsufficientDisk
    from minimax_studio.worker.server import app
    from minimax_studio.worker_client import WorkerClient

    Usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: Usage(1024**3, 0, int(0.2 * 1024**3))
    )
    http = TestClient(app)
    response = http.post("/downloads", json={"pack_id": "h3-fl2va"})
    assert response.status_code == 507
    assert "Not enough free disk" in response.json()["detail"]

    try:
        WorkerClient._raise(_HttpError(507, response.json()["detail"]))  # type: ignore[arg-type]
        raise AssertionError("expected InsufficientDisk")
    except InsufficientDisk as exc:
        assert "Not enough free disk" in str(exc)


def test_conflict_status_is_not_insufficient_disk() -> None:
    """409 is train/audition/dataset conflicts. Mapping it in _raise
    would turn 'run already busy' into Download anyway?."""
    from minimax_studio.errors import InsufficientDisk
    from minimax_studio.worker_client import WorkerClient

    try:
        WorkerClient._raise(_HttpError(409, "a run is already in progress"))  # type: ignore[arg-type]
        raise AssertionError("expected RuntimeError")
    except InsufficientDisk:
        raise AssertionError("409 must stay RuntimeError")
    except RuntimeError as exc:
        assert "already in progress" in str(exc)


def test_ui_does_not_substring_match_disk_copy() -> None:
    from pathlib import Path

    root = Path("src/minimax_studio/ui")
    hits = [
        str(path)
        for path in root.rglob("*.py")
        if "Not enough free disk" in path.read_text(encoding="utf-8")
    ]
    assert hits == []
    models = (root / "pages" / "models_page.py").read_text(encoding="utf-8")
    adapters = (root / "pages" / "adapters_page.py").read_text(encoding="utf-8")
    assert "confirm_and_download" in models
    assert "confirm_and_download" in adapters
    assert "territory_notice" not in models
    assert "territory_notice" not in adapters
    assert "Download this pack anyway" not in models
    assert "Download this adapter anyway" not in adapters


def test_confirm_and_download_asks_the_license_then_starts(monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    from minimax_studio.ui import download as download_mod
    from minimax_studio.ui.download import confirm_and_download
    from tests.dialogs import Dialogs

    calls: list[str] = []

    class Client:
        def start_download(self, pack_id: str, force: bool = False) -> dict:
            calls.append(pack_id)
            return {"id": "dl", "pack_id": pack_id, "status": "queued"}

    pack = {
        "id": "h3-fl2va",
        "license_name": "MiniMax H3 Community License",
        "territory_notice": "US/EU/UK/KR need a separate grant.",
    }
    dialogs = Dialogs(
        monkeypatch, {"question": QMessageBox.StandardButton.Yes}, download_mod
    )
    job = confirm_and_download(None, Client(), pack, noun="pack")  # type: ignore[arg-type]
    assert job is not None and job["id"] == "dl"
    assert calls == ["h3-fl2va"]
    assert dialogs.kinds() == ["question"]
    assert "US/EU/UK/KR" in dialogs.last()[2]
    assert "Download this pack anyway?" in dialogs.last()[2]


def test_confirm_and_download_no_skips_the_start(monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    from minimax_studio.ui import download as download_mod
    from minimax_studio.ui.download import confirm_and_download
    from tests.dialogs import Dialogs

    class Client:
        def start_download(self, pack_id: str, force: bool = False) -> dict:
            raise AssertionError("declining the license must not start a download")

    Dialogs(monkeypatch, {"question": QMessageBox.StandardButton.No}, download_mod)
    job = confirm_and_download(  # type: ignore[arg-type]
        None,
        Client(),
        {"id": "h3-motion", "territory_notice": "US/EU/UK/KR"},
        noun="adapter",
    )
    assert job is None


def test_confirm_and_download_skips_the_box_when_there_is_no_notice(
    monkeypatch,
) -> None:
    from minimax_studio.ui import download as download_mod
    from minimax_studio.ui.download import confirm_and_download
    from tests.dialogs import Dialogs

    class Client:
        def start_download(self, pack_id: str, force: bool = False) -> dict:
            return {"id": "dl", "pack_id": pack_id, "status": "queued"}

    dialogs = Dialogs(monkeypatch, {}, download_mod)
    job = confirm_and_download(  # type: ignore[arg-type]
        None, Client(), {"id": "music3-cuda", "territory_notice": None}
    )
    assert job is not None and job["id"] == "dl"
    assert dialogs.kinds() == []


def test_start_download_or_ask_retries_when_copy_is_reworded(monkeypatch) -> None:
    """The hatch is the type. The old substring is not in this message."""
    from PySide6.QtWidgets import QMessageBox

    from minimax_studio.errors import InsufficientDisk
    from minimax_studio.ui import download as download_mod
    from minimax_studio.ui.download import start_download_or_ask
    from tests.dialogs import Dialogs

    calls: list[tuple[str, bool]] = []

    class Client:
        def start_download(self, pack_id: str, force: bool = False) -> dict:
            calls.append((pack_id, force))
            if not force:
                raise InsufficientDisk("The models volume is full.")
            return {"id": "dl", "pack_id": pack_id, "status": "queued"}

    dialogs = Dialogs(
        monkeypatch, {"question": QMessageBox.StandardButton.Yes}, download_mod
    )
    job = start_download_or_ask(None, Client(), "h3-fl2va")  # type: ignore[arg-type]
    assert job is not None and job["id"] == "dl"
    assert calls == [("h3-fl2va", False), ("h3-fl2va", True)]
    assert "Download anyway?" in dialogs.last()[2]


def test_start_download_or_ask_no_does_not_force(monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    from minimax_studio.errors import InsufficientDisk
    from minimax_studio.ui import download as download_mod
    from minimax_studio.ui.download import start_download_or_ask
    from tests.dialogs import Dialogs

    calls: list[tuple[str, bool]] = []

    class Client:
        def start_download(self, pack_id: str, force: bool = False) -> dict:
            calls.append((pack_id, force))
            raise InsufficientDisk("The models volume is full.")

    Dialogs(monkeypatch, {"question": QMessageBox.StandardButton.No}, download_mod)
    job = start_download_or_ask(None, Client(), "h3-fl2va")  # type: ignore[arg-type]
    assert job is None
    assert calls == [("h3-fl2va", False)]


def test_start_download_or_ask_other_errors_are_warnings(monkeypatch) -> None:
    from minimax_studio.ui import download as download_mod
    from minimax_studio.ui.download import start_download_or_ask
    from tests.dialogs import Dialogs

    class Client:
        def start_download(self, pack_id: str, force: bool = False) -> dict:
            raise RuntimeError("network down")

    dialogs = Dialogs(monkeypatch, {}, download_mod)
    job = start_download_or_ask(None, Client(), "h3-fl2va")  # type: ignore[arg-type]
    assert job is None
    assert dialogs.kinds() == ["warning"]
    assert "network down" in dialogs.last()[2]

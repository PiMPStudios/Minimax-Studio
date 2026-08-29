import time
from pathlib import Path

from minimax_studio.worker.catalog import PACKS
from minimax_studio.worker.downloads import delete_pack, get_download, start_download


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
    try:
        start_download("h3-fl2va")
        raise AssertionError("expected a free-disk error")
    except RuntimeError as exc:
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

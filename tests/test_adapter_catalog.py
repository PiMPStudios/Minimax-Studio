"""PLAN-V3 S2: curated adapter catalog — not a live scrape, not on Models."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from minimax_studio.worker.catalog import ADAPTERS, H3_TERRITORY, PACKS, Pack, pack_or_raise
from minimax_studio.worker.downloads import (
    delete_pack,
    list_adapter_catalog,
    list_packs,
    start_download,
)


def test_catalog_is_h3_only_and_not_on_models() -> None:
    assert set(ADAPTERS) == {"h3-realism-people", "h3-motion"}
    for pack in ADAPTERS.values():
        assert pack.kind == "lora"
        assert pack.family == "h3"
        assert pack.local_dir == "loras"
        assert pack.territory_notice == H3_TERRITORY
        assert pack.allow_patterns
        assert pack.revision and len(pack.revision) == 40
        assert pack.sha256 and len(pack.sha256) == 64
        assert set(pack.sha256) <= set("0123456789abcdef")
        assert pack.min_bytes and pack.max_bytes
        assert pack.min_bytes < pack.max_bytes
        assert not any(
            pattern.endswith(".mp4") or "LICENSE" in pattern
            for pattern in pack.allow_patterns
        )
    assert "h3-realism-people" not in PACKS
    assert pack_or_raise("h3-realism-people").repo_id.startswith("fal/")


def test_adapter_ids_do_not_collide_with_pack_ids() -> None:
    # pack_or_raise checks PACKS first — a collision would silently shadow a row.
    assert not set(ADAPTERS) & set(PACKS)


def _pin_dummy_digest(monkeypatch: pytest.MonkeyPatch, pack: Pack, payload: bytes) -> None:
    monkeypatch.setitem(
        ADAPTERS, pack.id, replace(pack, sha256=hashlib.sha256(payload).hexdigest())
    )


def test_list_packs_does_not_include_catalog_loras(studio_home: Path) -> None:
    ids = {row["id"] for row in list_packs()}
    assert "h3-realism-people" not in ids
    assert "h3-fl2va" in ids


def test_catalog_status_and_download_registers_catalog(
    studio_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minimax_studio.worker import adapters

    rows = {row["id"]: row for row in list_adapter_catalog()}
    assert rows["h3-realism-people"]["ready"] is False
    assert "US" in (rows["h3-realism-people"]["territory_notice"] or "")

    pack = ADAPTERS["h3-realism-people"]
    marker = pack.marker_files[0]
    payload = b"x" * pack.min_bytes
    _pin_dummy_digest(monkeypatch, pack, payload)

    def snapshot(**kwargs):
        assert kwargs.get("revision") == pack.revision
        dest = Path(kwargs["local_dir"])
        path = dest / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(dest)

    record = start_download("h3-realism-people", snapshot=snapshot, force=True)
    import time

    deadline = time.time() + 5
    from minimax_studio.worker.downloads import get_download

    while time.time() < deadline:
        current = get_download(record["id"])
        if current["status"] in {"done", "error"}:
            break
        time.sleep(0.05)
    assert get_download(record["id"])["status"] == "done"
    dest = studio_home / "models" / "loras" / marker
    assert dest.is_file()
    listed = {row["id"]: row for row in adapters.list_adapters()}
    assert listed[marker]["source"] == "catalog"
    assert listed[marker]["kind"] == "h3"
    assert listed[marker]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert listed[marker]["revision"] == pack.revision
    catalog = {row["id"]: row for row in list_adapter_catalog()}
    assert catalog["h3-realism-people"]["ready"] is True
    assert catalog["h3-realism-people"]["verified"] is True


def test_delete_catalog_lora_does_not_wipe_the_loras_folder(
    studio_home: Path,
) -> None:
    loras = studio_home / "models" / "loras"
    loras.mkdir(parents=True)
    marker = ADAPTERS["h3-realism-people"].marker_files[0]
    (loras / marker).write_bytes(b"lora")
    trained = loras / "my-trained.safetensors"
    trained.write_bytes(b"keep-me")
    result = delete_pack("h3-realism-people")
    assert result["removed"] is True
    assert not (loras / marker).exists()
    assert trained.is_file()
    assert loras.is_dir()


def test_delete_catalog_lora_missing_marker_is_not_removed(studio_home: Path) -> None:
    loras = studio_home / "models" / "loras"
    loras.mkdir(parents=True)
    result = delete_pack("h3-motion")
    assert result["removed"] is False
    assert result["removed_bytes"] == 0
    assert loras.is_dir()


def test_delete_catalog_lora_forgets_the_registry_row(studio_home: Path) -> None:
    from minimax_studio.worker import adapters

    marker = ADAPTERS["h3-motion"].marker_files[0]
    loras = studio_home / "models" / "loras"
    loras.mkdir(parents=True)
    path = loras / marker
    path.write_bytes(b"lora")
    adapters.record_imported({"path": str(path), "kind": "h3"}, source="catalog")
    assert marker in {row["file"] for row in adapters.load_registry()}
    delete_pack("h3-motion")
    assert marker not in {row["file"] for row in adapters.load_registry()}
    assert not path.exists()


def test_delete_catalog_lora_does_not_swallow_a_registry_write_failure(
    studio_home: Path, monkeypatch
) -> None:
    from minimax_studio.worker import adapters

    marker = ADAPTERS["h3-motion"].marker_files[0]
    loras = studio_home / "models" / "loras"
    loras.mkdir(parents=True)
    path = loras / marker
    path.write_bytes(b"lora")
    adapters.record_imported({"path": str(path), "kind": "h3"}, source="catalog")

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk is full")

    monkeypatch.setattr(adapters, "save_registry", boom)
    with pytest.raises(RuntimeError, match="disk is full"):
        delete_pack("h3-motion")


def test_start_download_refuses_a_second_in_flight(
    studio_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time

    from minimax_studio.worker.downloads import cancel_download, get_download
    from minimax_studio.worker.runtime import runtime

    pack = ADAPTERS["h3-motion"]
    marker = pack.marker_files[0]
    payload = b"x" * pack.min_bytes
    _pin_dummy_digest(monkeypatch, pack, payload)
    # Hold the snapshot until the second start_download has been refused.
    release = threading.Event()

    def snapshot(**kwargs):
        release.wait(timeout=5)
        dest = Path(kwargs["local_dir"])
        path = dest / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(dest)

    first = start_download("h3-motion", snapshot=snapshot, force=True)
    try:
        with pytest.raises(RuntimeError, match="already downloading"):
            start_download("h3-motion", snapshot=snapshot, force=True)
    finally:
        release.set()
        cancel_download(first["id"])
        deadline = time.time() + 5
        while time.time() < deadline:
            if get_download(first["id"])["status"] in {"done", "error", "cancelled"}:
                break
            time.sleep(0.05)
        assert first["id"] not in runtime.download_stops


def test_catalog_download_refuses_a_too_small_marker(studio_home: Path) -> None:
    pack = ADAPTERS["h3-realism-people"]
    marker = pack.marker_files[0]

    def snapshot(**kwargs):
        dest = Path(kwargs["local_dir"])
        path = dest / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tiny")
        return str(dest)

    record = start_download("h3-realism-people", snapshot=snapshot, force=True)
    import time

    from minimax_studio.worker.downloads import get_download

    deadline = time.time() + 5
    while time.time() < deadline:
        current = get_download(record["id"])
        if current["status"] in {"done", "error"}:
            break
        time.sleep(0.05)
    final = get_download(record["id"])
    assert final["status"] == "error"
    assert "vouches for" in (final.get("error") or "")
    assert not (studio_home / "models" / "loras" / marker).exists()


def test_catalog_download_refuses_a_wrong_digest(studio_home: Path) -> None:
    pack = ADAPTERS["h3-motion"]
    marker = pack.marker_files[0]
    loras = studio_home / "models" / "loras"
    loras.mkdir(parents=True)
    kept = loras / "keep-me.safetensors"
    kept.write_bytes(b"mine")

    def snapshot(**kwargs):
        dest = Path(kwargs["local_dir"])
        path = dest / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * pack.min_bytes)
        return str(dest)

    record = start_download("h3-motion", snapshot=snapshot, force=True)
    import time

    from minimax_studio.worker.downloads import get_download

    deadline = time.time() + 5
    while time.time() < deadline:
        current = get_download(record["id"])
        if current["status"] in {"done", "error"}:
            break
        time.sleep(0.05)
    final = get_download(record["id"])
    assert final["status"] == "error"
    error = final.get("error") or ""
    assert pack.sha256 in error
    assert "deleted the copy" in error
    assert "models/loras/" in error
    assert not (loras / marker).exists()
    assert kept.is_file()


def test_catalog_row_without_a_recorded_digest_is_unverified(studio_home: Path) -> None:
    pack = ADAPTERS["h3-motion"]
    marker = pack.marker_files[0]
    path = studio_home / "models" / "loras" / marker
    path.parent.mkdir(parents=True)
    path.write_bytes(b"pre-pin-bytes" * 80)
    from minimax_studio.worker import adapters

    adapters.record_imported({"path": str(path), "kind": "h3"}, source="catalog")
    rows = {row["id"]: row for row in list_adapter_catalog()}
    assert rows["h3-motion"]["ready"] is True
    assert rows["h3-motion"]["verified"] is False


def test_catalog_background_hash_verifies_bytes_that_match_the_pin(
    studio_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minimax_studio.worker import adapters
    from minimax_studio.worker.downloads import join_catalog_verifies

    pack = ADAPTERS["h3-motion"]
    marker = pack.marker_files[0]
    payload = b"x" * pack.min_bytes
    _pin_dummy_digest(monkeypatch, pack, payload)
    path = studio_home / "models" / "loras" / marker
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    rows = {row["id"]: row for row in list_adapter_catalog()}
    assert rows["h3-motion"]["ready"] is True
    assert rows["h3-motion"]["verified"] is False
    join_catalog_verifies()
    rows = {row["id"]: row for row in list_adapter_catalog()}
    assert rows["h3-motion"]["verified"] is True
    listed = adapters.get_adapter(marker)
    assert listed["sha256"] == hashlib.sha256(payload).hexdigest()
    assert listed["revision"] == pack.revision

"""PLAN-V3 S2: curated adapter catalog — not a live scrape, not on Models."""

from pathlib import Path

from minimax_studio.worker.catalog import ADAPTERS, H3_TERRITORY, PACKS, pack_or_raise
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
        assert not any(
            pattern.endswith(".mp4") or "LICENSE" in pattern
            for pattern in pack.allow_patterns
        )
    assert "h3-realism-people" not in PACKS
    assert pack_or_raise("h3-realism-people").repo_id.startswith("fal/")


def test_list_packs_does_not_include_catalog_loras(studio_home: Path) -> None:
    ids = {row["id"] for row in list_packs()}
    assert "h3-realism-people" not in ids
    assert "h3-fl2va" in ids


def test_catalog_status_and_download_registers_imported(
    studio_home: Path,
) -> None:
    from minimax_studio.worker import adapters

    rows = {row["id"]: row for row in list_adapter_catalog()}
    assert rows["h3-realism-people"]["ready"] is False
    assert "US" in (rows["h3-realism-people"]["territory_notice"] or "")

    pack = ADAPTERS["h3-realism-people"]
    marker = pack.marker_files[0]

    def snapshot(**kwargs):
        dest = Path(kwargs["local_dir"])
        path = dest / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"lora")
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
    assert listed[marker]["source"] == "imported"
    assert listed[marker]["kind"] == "h3"
    assert list_adapter_catalog()[0]["ready"] is True or any(
        row["id"] == "h3-realism-people" and row["ready"] for row in list_adapter_catalog()
    )


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

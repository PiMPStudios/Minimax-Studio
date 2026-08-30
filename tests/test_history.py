from pathlib import Path

import pytest

from minimax_studio.worker.history import (
    delete_entry,
    get_entry,
    list_history,
    record_entry,
)


def test_history_ids_cannot_walk_out_of_the_root(studio_home: Path) -> None:
    with pytest.raises(KeyError):
        get_entry("..")
    with pytest.raises(KeyError):
        delete_entry("..")
    with pytest.raises(KeyError):
        get_entry("foo/bar")
    with pytest.raises(KeyError):
        delete_entry("../models")


def test_corrupt_meta_json_is_a_missing_entry(studio_home: Path) -> None:
    record_entry(
        {
            "id": "deadbeefcafe",
            "kind": "music",
            "prompt": "x",
            "output_path": str(studio_home / "history" / "deadbeefcafe" / "a.wav"),
        }
    )
    meta = studio_home / "history" / "deadbeefcafe" / "meta.json"
    meta.write_text("{not json", encoding="utf-8")
    with pytest.raises(KeyError):
        get_entry("deadbeefcafe")


def test_list_rebuilds_when_the_index_is_gone(studio_home: Path) -> None:
    record_entry(
        {
            "id": "aaa111aaa111",
            "kind": "music",
            "prompt": "keep me",
            "output_path": str(studio_home / "history" / "aaa111aaa111" / "a.wav"),
        }
    )
    index = studio_home / "history" / "index.jsonl"
    index.unlink()
    rows = list_history()
    assert any(row.get("id") == "aaa111aaa111" for row in rows)
    assert index.is_file()

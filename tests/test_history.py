from pathlib import Path

import pytest

from minimax_studio.worker.history import (
    delete_entry,
    get_entry,
    list_history,
    record_entry,
    trim_entry,
)

FFMPEG_STUB = """
import pathlib, sys
out = pathlib.Path(sys.argv[-1])
out.write_bytes(b"trimmed")
"""


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


@pytest.fixture
def ffmpeg_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "ffmpeg_stub.py"
    script.write_text(FFMPEG_STUB, encoding="utf-8")
    import sys

    monkeypatch.setenv(
        "MINIMAX_STUDIO_FFMPEG_BIN",
        f'"{sys.executable}" "{script}"',
    )


def _parent_take(studio_home: Path, stem: str = "parentparent") -> Path:
    folder = studio_home / "history" / stem
    folder.mkdir(parents=True)
    src = folder / "audio.wav"
    src.write_bytes(b"RIFF-original")
    record_entry(
        {
            "id": stem,
            "kind": "music",
            "backend": "stub",
            "prompt": "folk song",
            "lyrics": "[Verse]\nhello",
            "duration_s": 8,
            "loras": [{"id": "one.safetensors", "strength": 0.8}],
            "output_path": str(src),
            "media_type": "audio",
        }
    )
    return src


def test_trim_writes_a_new_row_and_keeps_the_parent(
    studio_home: Path, ffmpeg_stub
) -> None:
    src = _parent_take(studio_home)
    child = trim_entry("parentparent", 1.5, 4.0)
    assert child["id"] != "parentparent"
    assert child["trimmed_from"] == "parentparent"
    assert child["prompt"] == "folk song"
    assert child["lyrics"] == "[Verse]\nhello"
    assert child["loras"] == [{"id": "one.safetensors", "strength": 0.8}]
    assert child["duration_s"] == 2.5
    dest = Path(child["output_path"])
    assert dest.is_file()
    assert dest.read_bytes() == b"trimmed"
    root = (studio_home / "history").resolve()
    assert root in dest.resolve().parents
    assert src.read_bytes() == b"RIFF-original"
    parent = get_entry("parentparent")
    assert parent["duration_s"] == 8
    assert parent.get("trimmed_from") is None


def test_delete_child_does_not_chase_the_parent(
    studio_home: Path, ffmpeg_stub
) -> None:
    _parent_take(studio_home)
    child = trim_entry("parentparent", 0, 2)
    delete_entry(child["id"])
    assert get_entry("parentparent")["prompt"] == "folk song"
    with pytest.raises(KeyError):
        get_entry(child["id"])

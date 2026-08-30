from pathlib import Path

from minimax_studio.worker.loras import import_lora, list_loras


def test_import_lists_safetensors(studio_home: Path) -> None:
    src = studio_home / "adapter.safetensors"
    src.write_bytes(b"not-a-real-lora")
    imported = import_lora(str(src))
    assert Path(imported["path"]).is_file()
    names = {item["id"] for item in list_loras()}
    assert "adapter.safetensors" in names


def test_same_filename_in_two_folders_both_list(studio_home: Path) -> None:
    from minimax_studio.worker.runtime import runtime

    root = runtime.config.models_root()
    one = root / "loras" / "style.safetensors"
    two = root / "h3-comfy" / "loras" / "style.safetensors"
    one.parent.mkdir(parents=True, exist_ok=True)
    two.parent.mkdir(parents=True, exist_ok=True)
    one.write_bytes(b"a")
    two.write_bytes(b"b")
    paths = {item["path"] for item in list_loras()}
    assert str(one) in paths
    assert str(two) in paths

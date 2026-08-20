from pathlib import Path

from minimax_studio.worker.loras import import_lora, list_loras


def test_import_lists_safetensors(studio_home: Path) -> None:
    src = studio_home / "adapter.safetensors"
    src.write_bytes(b"not-a-real-lora")
    imported = import_lora(str(src))
    assert Path(imported["path"]).is_file()
    names = {item["id"] for item in list_loras()}
    assert "adapter.safetensors" in names

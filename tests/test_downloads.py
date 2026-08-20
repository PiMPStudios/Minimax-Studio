from pathlib import Path
import time

from minimax_studio.worker.catalog import PACKS
from minimax_studio.worker.downloads import delete_pack, get_download, start_download


def test_download_uses_injected_snapshot(studio_home: Path) -> None:
    def snapshot(**kwargs):
        dest = Path(kwargs["local_dir"])
        (dest / "modular_model_index.json").write_text("{}", encoding="utf-8")
        return str(dest)

    record = start_download("music3-cuda", snapshot=snapshot)
    deadline = time.time() + 5
    while time.time() < deadline:
        current = get_download(record["id"])
        if current["status"] in {"done", "error"}:
            break
        time.sleep(0.05)
    current = get_download(record["id"])
    assert current["status"] == "done"
    assert (studio_home / "models" / "music3-cuda" / "modular_model_index.json").is_file()


def test_delete_pack_removes_studio_copy(studio_home: Path) -> None:
    dest = studio_home / "models" / PACKS["music3-cuda"].local_dir
    dest.mkdir(parents=True)
    (dest / "modular_model_index.json").write_text("{}", encoding="utf-8")
    result = delete_pack("music3-cuda")
    assert result["ok"] is True
    assert not dest.exists()

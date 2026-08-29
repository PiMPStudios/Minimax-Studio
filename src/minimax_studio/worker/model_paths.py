from __future__ import annotations

import os
from pathlib import Path

from minimax_studio.worker.catalog import Pack
from minimax_studio.worker.fsutil import dir_bytes

_BYTES_TTL_S = 5.0
_BYTES_CACHE: dict[str, tuple[float, int]] = {}


def reset_bytes_cache() -> None:
    """Forget cached directory sizes (after downloads/removals)."""
    _BYTES_CACHE.clear()


def _dir_bytes_cached(path: Path) -> int:
    """dir_bytes with a short TTL. File *existence* stays live; only the
    expensive whole-tree size walk is cached (Models page and preflight
    refresh this constantly)."""
    import time

    key = str(path)
    now = time.monotonic()
    hit = _BYTES_CACHE.get(key)
    if hit and now - hit[0] < _BYTES_TTL_S:
        return hit[1]
    value = dir_bytes(path)
    _BYTES_CACHE[key] = (now, value)
    return value

_NESTED = ("", "h3-comfy", "minimax-h3", "minimax-music-3", "music3-comfy")
_FOLDER_KEYS = {
    "checkpoints",
    "configs",
    "loras",
    "vae",
    "text_encoders",
    "clip",
    "clip_vision",
    "diffusion_models",
    "unet",
    "embeddings",
    "controlnet",
    "style_models",
    "upscale_models",
    "latent_upscale_models",
    "vae_approx",
    "gligen",
    "hypernetworks",
    "photomaker",
    "model_patches",
    "audio_encoders",
}


def guess_comfy_model_dirs() -> list[Path]:
    home = Path.home()
    guesses: list[Path] = [
        home / "ai" / "ComfyUI" / "models",
        home / "ComfyUI" / "models",
        home / "Documents" / "ComfyUI" / "models",
        home / "models",
        home / "models" / "minimax-h3",
    ]
    env = os.environ.get("COMFYUI_PATH") or os.environ.get("COMFYUI_MODELS")
    if env:
        path = Path(env).expanduser()
        guesses.append(path)
        if path.name != "models":
            guesses.append(path / "models")
    seen: set[str] = set()
    out: list[Path] = []
    for item in guesses:
        try:
            resolved = item.expanduser().resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        out.append(resolved)
    return out


def search_roots(models_root: Path, comfy_models_dir: str | None = None) -> list[Path]:
    seen: set[str] = set()
    roots: list[Path] = []

    def add(path: Path | None) -> None:
        if path is None:
            return
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            return
        seen.add(key)
        roots.append(resolved)

    add(models_root)
    if comfy_models_dir:
        add(Path(comfy_models_dir))
    else:
        for guessed in guess_comfy_model_dirs():
            add(guessed)
        for extra in extra_model_path_roots():
            add(extra)
    return roots


def extra_model_path_files() -> list[Path]:
    home = Path.home()
    files = [
        home / "ai" / "ComfyUI" / "extra_model_paths.yaml",
        home / "ComfyUI" / "extra_model_paths.yaml",
        home / "Documents" / "ComfyUI" / "extra_model_paths.yaml",
    ]
    env = os.environ.get("COMFYUI_PATH")
    if env:
        files.append(Path(env).expanduser() / "extra_model_paths.yaml")
    return files


def extra_model_path_roots() -> list[Path]:
    roots: list[Path] = []
    for path in extra_model_path_files():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        roots.extend(parse_extra_model_paths(text))
    return roots


def parse_extra_model_paths(text: str) -> list[Path]:
    """Lightweight YAML subset for Comfy extra_model_paths.yaml."""
    roots: list[Path] = []
    section_base: Path | None = None
    collecting = False
    collect_indent = 0

    def add_rel(rel: str) -> None:
        if not section_base or not rel:
            return
        roots.append(section_base / rel.strip().rstrip("/"))

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if collecting:
            if indent > collect_indent and ":" not in stripped.split(" ", 1)[0]:
                add_rel(stripped)
                continue
            collecting = False
        if indent == 0 and stripped.endswith(":") and "|" not in stripped:
            section_base = None
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "base_path" and value:
            try:
                section_base = Path(value).expanduser()
            except OSError:
                section_base = None
            else:
                roots.append(section_base)
            continue
        if value == "|":
            if key in _FOLDER_KEYS:
                collecting = True
                collect_indent = indent
            continue
        if value and key in _FOLDER_KEYS:
            add_rel(value)
    return roots


def marker_candidates(root: Path, marker: str) -> list[Path]:
    rel = Path(marker)
    name = rel.name
    paths = [root / rel, root / name]
    for nest in _NESTED:
        base = root / nest if nest else root
        paths.append(base / rel)
        paths.append(base / name)
        if len(rel.parts) > 1:
            paths.append(base / rel.parts[0] / name)
    # unique while preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def find_marker(roots: list[Path], marker: str) -> Path | None:
    for root in roots:
        for path in marker_candidates(root, marker):
            if path.is_file():
                return path
    return None


def pack_status(
    pack: Pack,
    models_root: Path,
    extra_roots: list[Path] | None = None,
) -> dict:
    dest = models_root / pack.local_dir
    roots = extra_roots if extra_roots is not None else [models_root]
    found: dict[str, Path] = {}
    missing: list[str] = []
    studio_hits = 0
    if pack.marker_files:
        for marker in pack.marker_files:
            studio_path = dest / marker
            if studio_path.is_file():
                found[marker] = studio_path
                studio_hits += 1
                continue
            extra = find_marker(roots, marker)
            if extra is not None:
                found[marker] = extra
            else:
                missing.append(marker)
        ready = not missing
    else:
        ready = dest.is_dir() and any(dest.iterdir())
        if ready:
            studio_hits = 1

    if ready and found:
        path = _common_parent([item.parent for item in found.values()])
        bytes_on_disk = _dir_bytes_cached(path)
    else:
        path = dest
        bytes_on_disk = _dir_bytes_cached(dest) if dest.exists() else 0

    source = "studio"
    if ready and pack.marker_files and studio_hits < len(pack.marker_files):
        source = "comfy"
    elif ready and pack.kind == "comfy" and studio_hits == 0:
        source = "comfy"

    return {
        "id": pack.id,
        "title": pack.title,
        "summary": pack.summary,
        "repo_id": pack.repo_id,
        "family": pack.family,
        "kind": pack.kind,
        "approx_gb": pack.approx_gb,
        "license_name": pack.license_name,
        "territory_notice": pack.territory_notice,
        "path": str(path),
        "ready": ready,
        "missing": missing,
        "bytes_on_disk": bytes_on_disk,
        "partial": (not ready) and dest.exists() and _dir_bytes_cached(dest) > 1024 * 1024,
        "source": source,
        "files": {key: str(value) for key, value in found.items()},
    }


def _common_parent(paths: list[Path]) -> Path:
    if not paths:
        raise ValueError("no paths")
    if len(paths) == 1:
        return paths[0]
    common = os.path.commonpath([str(path) for path in paths])
    return Path(common)

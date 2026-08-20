"""Optional OS keychain for API tokens. File storage remains the default."""

from __future__ import annotations

from typing import Any

SERVICE = "minimax-studio"
SECRET_FIELDS = ("hf_token", "minimax_api_key", "llm_api_key")


def keyring_available() -> bool:
    try:
        ring = _keyring()
        ring.get_password(SERVICE, "__probe__")
        return True
    except Exception:
        return False


def persist_secrets(values: dict[str, str | None]) -> None:
    ring = _keyring()
    for key in SECRET_FIELDS:
        value = (values.get(key) or "").strip()
        if value:
            ring.set_password(SERVICE, key, value)
            continue
        try:
            ring.delete_password(SERVICE, key)
        except Exception:
            pass


def load_secrets() -> dict[str, str | None]:
    ring = _keyring()
    out: dict[str, str | None] = {}
    for key in SECRET_FIELDS:
        try:
            out[key] = ring.get_password(SERVICE, key)
        except Exception:
            out[key] = None
    return out


def clear_secrets() -> None:
    persist_secrets({key: None for key in SECRET_FIELDS})


def _keyring() -> Any:
    import keyring

    return keyring

"""CUDA LoRA apply — the pipe is cached, so Fast then Quality must be Quality."""

from __future__ import annotations

import pytest

from minimax_studio.worker.backends.h3 import _apply_loras


class _PeftPipe:
    def __init__(self) -> None:
        self.loaded: list[tuple[str, str | None]] = []
        self.active: tuple[list[str], list[float] | None] | None = None
        self.unloads = 0

    def load_lora_weights(self, path, adapter_name=None):  # noqa: ANN001
        self.loaded.append((str(path), adapter_name))

    def unload_lora_weights(self) -> None:
        self.unloads += 1
        self.loaded.clear()
        self.active = None

    def set_adapters(self, names, adapter_weights=None):  # noqa: ANN001
        self.active = (list(names), adapter_weights)


class _FuseOnlyPipe:
    """Older diffusers: load_lora_weights(path) only, no adapter_name."""

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.unloads = 0

    def load_lora_weights(self, path):  # noqa: ANN001
        self.loaded.append(str(path))

    def unload_lora_weights(self) -> None:
        self.unloads += 1
        self.loaded.clear()


class _NoLoraPipe:
    pass


def test_empty_list_unloads_previous_adapters() -> None:
    pipe = _PeftPipe()
    _apply_loras(pipe, [{"id": "/models/loras/turbo.safetensors", "strength": 1.0}])
    assert pipe.unloads == 1
    assert pipe.active == (["adapter0"], [1.0])
    _apply_loras(pipe, [])
    assert pipe.unloads == 2
    assert pipe.loaded == []
    assert pipe.active is None


def test_second_job_replaces_instead_of_stacking() -> None:
    pipe = _PeftPipe()
    _apply_loras(pipe, [{"id": "/a/one.safetensors", "strength": 0.8}])
    _apply_loras(pipe, [{"id": "/b/two.safetensors", "strength": 0.5}])
    assert pipe.unloads == 2
    assert pipe.loaded == [("/b/two.safetensors", "adapter0")]
    assert pipe.active == (["adapter0"], [0.5])


def test_two_loras_stack_with_strengths() -> None:
    pipe = _PeftPipe()
    _apply_loras(
        pipe,
        [
            {"id": "/a/turbo.safetensors", "strength": 1.0},
            {"id": "/a/style.safetensors", "strength": 0.8},
        ],
    )
    assert pipe.loaded == [
        ("/a/turbo.safetensors", "adapter0"),
        ("/a/style.safetensors", "adapter1"),
    ]
    assert pipe.active == (["adapter0", "adapter1"], [1.0, 0.8])


def test_fuse_only_pipe_cannot_stack() -> None:
    pipe = _FuseOnlyPipe()
    with pytest.raises(RuntimeError, match="cannot stack"):
        _apply_loras(
            pipe,
            [
                {"id": "/a/one.safetensors", "strength": 1.0},
                {"id": "/a/two.safetensors", "strength": 1.0},
            ],
        )


def test_fuse_only_pipe_loads_one_after_unload() -> None:
    pipe = _FuseOnlyPipe()
    _apply_loras(pipe, [{"id": "/a/turbo.safetensors"}])
    _apply_loras(pipe, [{"id": "/a/style.safetensors"}])
    assert pipe.unloads == 2
    assert pipe.loaded == ["/a/style.safetensors"]


def test_set_adapters_failure_names_the_file() -> None:
    pipe = _PeftPipe()

    def boom(names, adapter_weights=None):  # noqa: ANN001, ARG001
        raise RuntimeError("no peft")

    pipe.set_adapters = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="style.safetensors"):
        _apply_loras(pipe, [{"id": "/a/style.safetensors", "strength": 0.8}])


def test_pipe_without_lora_api_refuses_a_selected_adapter() -> None:
    with pytest.raises(RuntimeError, match="cannot load LoRAs"):
        _apply_loras(_NoLoraPipe(), [{"id": "/a/style.safetensors"}])


def test_pipe_without_lora_api_accepts_an_empty_list() -> None:
    _apply_loras(_NoLoraPipe(), [])


def test_load_failure_names_the_file() -> None:
    pipe = _PeftPipe()

    def boom(path, adapter_name=None):  # noqa: ANN001, ARG001
        raise ValueError("bad weights")

    pipe.load_lora_weights = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="broken.safetensors"):
        _apply_loras(pipe, [{"id": "/a/broken.safetensors"}])

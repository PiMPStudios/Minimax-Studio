"""Trainer-process startup hook (PYTHONPATH, not imported by the GUI).

SimpleTuner 4.8.0's ``mark_cudagraph_step_begin`` treats ``dynamo_backend="no"``
as truthy and then ``import torch._inductor``, which on torch 2.13 raises
``TypeError: 'CustomDecompTable' object is not a mapping``. Metal 2026-08-30:
VAE cache and a baseline validation sample both ran; the first train step died
here. Skip the mark when Dynamo is off.
"""

from __future__ import annotations

import sys


def _is_simpletuner_process() -> bool:
    """PYTHONPATH is set for every trainer Popen, including pytest stubs.

    Importing sdnq/simpletuner here would load torch in those stubs and hang
    the train-run tests. Only patch the real trainer (and accelerate's
    launch of train.py).
    """
    for arg in sys.argv:
        lowered = arg.replace("\\", "/").lower()
        if lowered.endswith("simpletuner/train.py") or lowered.endswith("/simpletuner"):
            return True
        if lowered.rsplit("/", 1)[-1] in {"simpletuner", "simpletuner.exe"}:
            return True
    return False


def _patch_dynamo() -> None:
    try:
        import simpletuner.helpers.training.dynamo as dynamo
    except Exception:
        return
    orig = dynamo.mark_cudagraph_step_begin
    off = {"", "no", "none", "false", "disabled", "0"}

    def wrapped(config) -> None:
        backend = str(getattr(config, "dynamo_backend", "") or "").strip().lower()
        if backend in off:
            return
        orig(config)

    dynamo.mark_cudagraph_step_begin = wrapped  # type: ignore[method-assign]


def _patch_sdnq() -> None:
    """SimpleTuner 4.8.0's ConvRot loader omits args sdnq 0.2.6 requires.

    Metal 2026-08-30: H3 VAE cache succeeded, then
    MiniMaxH3Transformer3DModel.from_single_file on the Comfy INT8 DiT died in
    SDNQDequantizer.__init__ (codebook_steps, use_codebook).
    """
    try:
        from sdnq.dequantizer import SDNQDequantizer
    except Exception:
        return
    orig = SDNQDequantizer.__init__

    def wrapped(self, *args, **kwargs):
        kwargs.setdefault("codebook_steps", 8)
        kwargs.setdefault("use_codebook", False)
        return orig(self, *args, **kwargs)

    SDNQDequantizer.__init__ = wrapped  # type: ignore[method-assign]


if _is_simpletuner_process():
    _patch_dynamo()
    _patch_sdnq()

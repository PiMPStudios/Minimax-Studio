from __future__ import annotations

from fastapi import FastAPI

from minimax_studio import __version__
from minimax_studio.worker.probe import probe

app = FastAPI(title="MiniMax Studio Worker", version=__version__)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "version": __version__, "service": "minimax-studio-worker"}


@app.get("/probe")
def hardware_probe() -> dict[str, object]:
    return probe()

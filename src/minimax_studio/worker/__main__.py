from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax Studio worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    # If MINIMAX_STUDIO_WORKER_TOKEN is set in the environment (the GUI does
    # this per launch), every route requires the X-Minimax-Studio-Token
    # header. Started bare, the worker stays open for development.
    uvicorn.run(
        "minimax_studio.worker.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

"""Agent-facing read surface over the Studio worker (MCP thread, M0 slice).

Why a CLI before the MCP server, when MCP is the thing agents actually speak:

* the MCP server will be a thin translator over exactly these calls, so the
  tool schema is written once and every client — Claude Desktop, Codex, Cursor,
  and the pi "skills plus a CLI" crowd — reads the same JSON;
* this repo's CI is stub-backends-only, no GPU, no network, and no MCP client on
  the runner. A CLI is testable against a spawned worker with ``subprocess``;
  an MCP server is not, so an MCP-only surface would be a demo, not a feature;
* it enforces the rule that the worker owns all job/model/disk state: this
  module holds none of it and imports no Qt, no torch, no diffusers.

stdout is pure JSON — no preamble an agent has to strip. Prose and failures go
to stderr. Exit codes: 0 good answer, 1 the worker refused or failed, 2 nobody
told us where the worker is.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from minimax_studio import __version__, agent_handoff
from minimax_studio.worker_client import WorkerClient

URL_ENV = agent_handoff.URL_ENV
# Same name the worker's middleware reads (worker/server.py AUTH_ENV), so one
# exported variable is both the client's address-book entry and the server's
# gate value.
TOKEN_ENV = agent_handoff.TOKEN_ENV

NO_ENDPOINT = (
    f"No MiniMax Studio worker to talk to. Set {URL_ENV} (for example "
    "http://127.0.0.1:8756) or pass --url. Neither was found, and there was "
    "no usable agent handoff next to config.json — Studio only writes that "
    "file while Settings → Allow agent access is on."
)
AUTH_FAILURE = (
    "The worker answered with 401: it was launched with a token and this "
    f"request did not carry it. Set {TOKEN_ENV} to the value Studio started "
    "with (it is regenerated each launch), or pass --token. Studio's worker is "
    "deliberately not open to other local processes without it."
)


def resolve_endpoint(
    url: str | None = None, token: str | None = None
) -> tuple[str | None, str | None, str | None, str | None]:
    """(url, token, how we found them, problem). Flags beat environment,
    which beats the handoff Studio writes when agent access is switched on.
    """
    resolved_url = (url or "").strip()
    resolved_token = (token or "").strip()
    if resolved_url:
        return resolved_url, (resolved_token or None), "flag", None
    resolved_url = (os.environ.get(URL_ENV) or "").strip()
    resolved_token = (os.environ.get(TOKEN_ENV) or "").strip()
    if resolved_url:
        return resolved_url, (resolved_token or None), "env", None
    handoff = agent_handoff.read()
    if handoff:
        return str(handoff["url"]), handoff.get("token") or None, "handoff", None
    return None, None, None, NO_ENDPOINT


def status(client: WorkerClient, via: str = "unknown") -> dict[str, Any]:
    """What this machine can run right now: worker, hardware, installed packs."""
    health = client.health()
    hardware = client.probe()
    return {
        "studio_version": __version__,
        "connected_via": via,
        "worker": health,
        "hardware": hardware,
        "can_generate_now": bool(hardware.get("packs_ready")),
    }


def packs(client: WorkerClient) -> dict[str, Any]:
    """Downloadable packs and curated LoRAs, with what is already on disk."""
    rows = client.list_packs()
    catalog: list[dict[str, Any]] = []
    try:
        catalog = client.list_adapter_catalog()
    except Exception:
        # The catalog is a network read on the worker. Its absence is a
        # narrower answer, not a failed one.
        catalog = []
    return {
        "packs": [
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "kind": row.get("kind"),
                "ready": bool(row.get("ready")),
                "recommended": bool(row.get("recommended")),
                "approx_gb": row.get("approx_gb"),
                "source": row.get("source"),
            }
            for row in rows
        ],
        "adapters": [
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "ready": bool(row.get("ready")),
                "verified": bool(row.get("verified")),
            }
            for row in catalog
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minimax-studio-agent",
        description="Read-only agent surface over a running MiniMax Studio worker.",
    )
    parser.add_argument("--url", help=f"worker URL (falls back to {URL_ENV})")
    parser.add_argument("--token", help=f"worker token (falls back to {TOKEN_ENV})")
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="per-request timeout, seconds"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="worker, hardware, and what is ready to run")
    sub.add_parser("packs", help="downloadable packs and curated LoRAs")
    return parser


_COMMANDS = {"status", "packs"}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    url, token, via, problem = resolve_endpoint(args.url, args.token)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    client = WorkerClient(str(url), timeout=args.timeout, token=token)
    try:
        payload = status(client, via) if args.command == "status" else packs(client)
    except Exception as exc:  # worker boundary: one sentence, one exit code
        detail = str(exc)
        if "401" in detail:
            detail = AUTH_FAILURE
        print(
            f"minimax-studio-agent {args.command}: {detail}\n"
            f"(worker: {url} found by {via} — is Studio running? scripts/run.sh)",
            file=sys.stderr,
        )
        if via == "handoff":
            # The likeliest reason a handoff URL does not answer is that the
            # launch which wrote it has since exited.
            print(
                f"(that address came from {agent_handoff.handoff_path()}; "
                "it is stale if Studio has quit since)",
                file=sys.stderr,
            )
        return 1
    finally:
        client.close()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# MiniMax Studio runs on exactly one Python: .python-version is the source of
# truth (pyproject's requires-python agrees, CI runs only that version). This
# launcher finds it, builds .venv on it, moves aside a .venv built on anything
# else, and then runs the app. No more silent 3.14 venvs that cannot install
# the pinned [train] extra.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

want="$(tr -d '[:space:]' <.python-version)"
maj="${want%%.*}"
min="${want##*.}"

is_wanted() {
  "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==($maj,$min) else 1)" \
    >/dev/null 2>&1
}

# 1. Which interpreter is Python $want? MINIMAX_STUDIO_PYTHON overrides.
py="${MINIMAX_STUDIO_PYTHON:-}"
if [[ -n "$py" ]] && ! is_wanted "$py"; then
  echo "MINIMAX_STUDIO_PYTHON=$py is not Python $want." >&2
  exit 1
fi
if [[ -z "$py" ]]; then
  for cand in "python$want" python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && is_wanted "$cand"; then
      py="$cand"
      break
    fi
  done
fi
if [[ -z "$py" ]]; then
  echo "MiniMax Studio needs Python $want — not found on PATH." >&2
  echo "Install it (apt/brew/python.org) and re-run, or set" >&2
  echo "MINIMAX_STUDIO_PYTHON to the full path of a $want interpreter." >&2
  exit 1
fi

# 2. A .venv built on another Python looks ready and silently cannot install
#    [train]. Move it aside — never delete someone's work — and rebuild.
if [[ -d .venv ]] && ! is_wanted .venv/bin/python; then
  stale="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown)"
  echo ".venv is Python $stale, not $want — moving it to .venv.pre-$stale" >&2
  echo "and rebuilding on $py. Delete that folder when you are done with it." >&2
  rm -rf ".venv.pre-$stale"
  mv .venv ".venv.pre-$stale"
fi

if [[ ! -d .venv ]]; then
  "$py" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
is_wanted python # the venv we are about to run in must be the pinned one
pip install -e ".[dev]" 2>/dev/null || pip install -e .
exec python -m minimax_studio "$@"

#!/usr/bin/env bash
# MiniMax Studio runs on exactly one Python: .python-version is the source of
# truth (pyproject's requires-python agrees, CI runs only that version). Since
# 0.2.67 this launcher *provides* that interpreter instead of demanding it: uv
# reads the pin, reuses a system Python 3.12 when one exists, and otherwise
# downloads a standalone build it keeps under ~/.local/share/uv. Nobody has to
# install Python 3.12 to run Studio any more — but the pin itself did not
# loosen, because simpletuner==4.8.0 ships nothing outside >=3.12,<3.14 and
# 0.2.28 is the release that landed a [train] extra nothing could install.
#
# MINIMAX_STUDIO_PYTHON=/path/to/python3.12 still bypasses uv entirely (offline
# shops, distro packagers). Nothing else picks a different version.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

want="$(tr -d '[:space:]' <.python-version)"
maj="${want%%.*}"
min="${want##*.}"

UV_INSTALL="curl -LsSf https://astral.sh/uv/install.sh | sh"
install_uv=0
print_plan=0
app_args=()
for arg in "$@"; do
  case "$arg" in
    --print-runtime) print_plan=1 ;;
    --install-uv) install_uv=1 ;;
    *) app_args+=("$arg") ;;
  esac
done

is_wanted() {
  "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==($maj,$min) else 1)" \
    >/dev/null 2>&1
}

venv_python() {
  [[ -x .venv/bin/python ]] && .venv/bin/python -c \
    'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "none"
}

# 1. Where is uv? MINIMAX_STUDIO_UV_BIN wins, then PATH, then the two places the
#    official installers put it (so a fresh install works without a new shell).
uv="${MINIMAX_STUDIO_UV_BIN:-}"
if [[ -z "$uv" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv="$(command -v uv)"
  elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv="$HOME/.local/bin/uv"
  elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
    uv="$HOME/.cargo/bin/uv"
  fi
fi

# --print-runtime answers "what is Studio actually going to use?" without
# building anything, so a support thread can be settled from one paste.
if [[ "$print_plan" == 1 ]]; then
  echo "wants python   $want  (.python-version)"
  echo "uv             ${uv:-not found}"
  echo "override       ${MINIMAX_STUDIO_PYTHON:-none}"
  echo ".venv python   $(venv_python)"
  exit 0
fi

# 2. Apple Silicon Music 3 needs mlx-audio (mlx_audio.music, >=0.5.0). Intel
#    Macs, Linux, and Windows stay on [dev] — mlx has no CUDA wheels here.
extras="dev"
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  extras="dev,mlx"
fi

# A .venv built on another Python looks ready and silently cannot install
# [train]. Move it aside — never delete someone's work — and rebuild.
retire_stale_venv() {
  if [[ -d .venv ]] && ! is_wanted .venv/bin/python; then
    stale="$(venv_python)"
    echo ".venv is Python $stale, not $want — moving it to .venv.pre-$stale" >&2
    echo "and rebuilding on Python $want. Delete that folder when you are done with it." >&2
    rm -rf ".venv.pre-$stale"
    mv .venv ".venv.pre-$stale"
  fi
}

install_with_pip() {
  python -m pip install -e ".[$extras]" 2>/dev/null || python -m pip install -e .
}

# 3. The documented opt-out: an interpreter you chose yourself, venv built the
#    slow way, uv never required.
if [[ -n "${MINIMAX_STUDIO_PYTHON:-}" ]]; then
  if ! is_wanted "$MINIMAX_STUDIO_PYTHON"; then
    echo "MINIMAX_STUDIO_PYTHON=$MINIMAX_STUDIO_PYTHON is not Python $want." >&2
    exit 1
  fi
  retire_stale_venv
  [[ -d .venv ]] || "$MINIMAX_STUDIO_PYTHON" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  is_wanted python
  install_with_pip
  exec python -m minimax_studio "${app_args[@]+"${app_args[@]}"}"
fi

# 4. The default path: uv resolves the pinned interpreter.
if [[ -z "$uv" ]]; then
  if [[ "$install_uv" == 1 ]]; then
    echo "Installing uv: $UV_INSTALL"
    bash -c "$UV_INSTALL"
    uv="$HOME/.local/bin/uv"
    [[ -x "$uv" ]] || { echo "uv still not found after installing." >&2; exit 1; }
  elif [[ -t 0 ]]; then
    echo "MiniMax Studio uses uv to build its environment on Python $want."
    echo "It is not installed. Install it now? This runs:"
    echo "    $UV_INSTALL"
    read -r -p "[y/N] " reply
    if [[ "$reply" == [yY]* ]]; then
      bash -c "$UV_INSTALL"
      uv="$HOME/.local/bin/uv"
      [[ -x "$uv" ]] || { echo "uv still not found after installing." >&2; exit 1; }
    fi
  fi
fi
if [[ -z "$uv" ]]; then
  echo "MiniMax Studio needs uv (or set MINIMAX_STUDIO_PYTHON to a Python $want)." >&2
  echo "Install uv:  $UV_INSTALL" >&2
  echo "Then re-run scripts/run.sh." >&2
  exit 1
fi

retire_stale_venv
# --seed so .venv/bin/pip exists: AGENTS.md tells people to run
# `pip install -e ".[train]"` inside the venv, and a uv venv has no pip without it.
[[ -d .venv ]] || "$uv" venv --seed --python "$want" .venv
# shellcheck disable=SC1091
source .venv/bin/activate
is_wanted python
"$uv" pip install -e ".[$extras]" 2>/dev/null || "$uv" pip install -e .
exec python -m minimax_studio "${app_args[@]+"${app_args[@]}"}"

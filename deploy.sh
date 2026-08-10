#!/usr/bin/env bash
# Rsync this project to a robot (or any) host over SSH, then optionally provision.
#
# The remote path defaults to ~/candy-shop; override with a second arg.
#
# Usage:
#   ./deploy.sh <host> [remote-path] [--sync]
#
#   <host>          ssh target, e.g. robotuser@robot.local
#   [remote-path]   destination on the host (default: ~/candy-shop)
#   --sync          run `uv sync` on the remote after the copy so the host is
#                   ready to run. Omit to just copy the files.
#
# Examples:
#   ./deploy.sh robotuser@robot.local
#   ./deploy.sh robotuser@robot.local --sync
#   ./deploy.sh robotuser@robot.local ~/demos/candy-shop --sync
#
# `.env` IS synced so the host inherits the shared config; per-machine overrides
# go in `.env.local`, which is NOT synced. Venvs, caches, model weights
# (checkpoints/), recorded datasets (data/, outputs/) stay on the machine that
# made them — see the EXCLUDES list below.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <host> [remote-path] [--sync]" >&2
    echo "  e.g. $0 robotuser@robot.local --sync" >&2
    exit 1
fi

HOST="$1"; shift
REMOTE_PATH="~/candy-shop"
DO_SYNC=0

for arg in "$@"; do
    case "$arg" in
        --sync) DO_SYNC=1 ;;
        -*) echo "unknown flag '$arg'" >&2; exit 1 ;;
        *)  REMOTE_PATH="$arg" ;;
    esac
done

command -v rsync >/dev/null 2>&1 || { echo "rsync not found on this machine" >&2; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"   # repo root (this script lives at the root)

# Excludes (same syntax as .gitignore). Patterns without a leading slash match at
# any depth. Everything else — including .env — ships.
EXCLUDES=(
    # Version control / Python / caches.
    ".git/"
    ".venv/"
    "__pycache__/"
    "*.pyc"
    "*.egg-info/"
    ".pytest_cache/"
    ".ruff_cache/"
    ".mypy_cache/"
    # Recorded corpora and weights stay on the machine that made them.
    "data/"
    "outputs/"
    "checkpoints/"
    # Agent tooling — not needed on the host.
    ".claude/"
    # Editor / OS noise.
    ".vscode/"
    ".idea/"
    ".DS_Store"
    # Per-machine overrides; each machine keeps its own.
    ".env.local"
)

EXCLUDE_ARGS=()
for pat in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=(--exclude "$pat")
done

echo "[deploy] syncing $HERE/ -> $HOST:$REMOTE_PATH/"
# Unquoted $REMOTE_PATH in the remote command so the remote shell expands ~.
ssh "$HOST" "mkdir -p $REMOTE_PATH"
rsync -azP --delete "${EXCLUDE_ARGS[@]}" "$HERE/" "$HOST:$REMOTE_PATH/"

if [[ "$DO_SYNC" -eq 1 ]]; then
    echo "[deploy] running 'uv sync' on $HOST (first run is slow) ..."
    if ! ssh "$HOST" "cd $REMOTE_PATH && uv sync"; then
        echo "[deploy] WARN: remote 'uv sync' failed — run 'uv sync' in $REMOTE_PATH on the host yourself" >&2
    fi
fi

cat <<EOF

[deploy] done. On the host:

  cd ${REMOTE_PATH}
$( [[ "$DO_SYNC" -eq 0 ]] && echo "  uv sync                 # once, if not deployed with --sync" )
  uv run robot            # robot host
  # uv run teleoperator   # or the teleoperator, policy, move-to, reward operators

EOF

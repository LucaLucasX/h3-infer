#!/usr/bin/env bash
# Stop :8190 and remove EXP custom_node symlink. Leaves vendor build intact.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/sparse_attn"
CP="$ROOT/ComfyUI-master_cp"
PORT="${H3_SPARGE_PORT:-8190}"
VENV="${VENV:-$ROOT/.venv_sage_bench}"
NODE_LINK="$CP/custom_nodes/H3-SpargeAttn-Exp"

pkill -f "${VENV}/bin/python main.py --listen 0.0.0.0 --port ${PORT}" 2>/dev/null || true
if [[ -L "$NODE_LINK" ]]; then
  rm -f "$NODE_LINK"
  echo "removed symlink $NODE_LINK"
elif [[ -e "$NODE_LINK" ]]; then
  echo "WARN: $NODE_LINK exists but is not a symlink — not removing" >&2
fi
echo "teardown done (:${PORT} stopped). :8188 untouched."

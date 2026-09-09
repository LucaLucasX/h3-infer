#!/usr/bin/env bash
# Stop ComfyUI processes started from this deployment directory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT="${COMFY_PORT:-8188}"
PIDS=()

add_pid() {
  local pid="${1:-}"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  [[ "$pid" == "$$" ]] && return 0
  kill -0 "$pid" 2>/dev/null || return 0
  PIDS+=("$pid")
}

# 1) Listeners on ComfyUI port
if command -v lsof >/dev/null 2>&1; then
  while read -r pid; do
    add_pid "$pid"
  done < <(lsof -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)
fi
if command -v fuser >/dev/null 2>&1; then
  for pid in $(fuser "${PORT}/tcp" 2>/dev/null || true); do
    add_pid "$pid"
  done
fi

# 2) python main.py whose cwd is this ROOT (start_comfyui.sh: cd ROOT && exec python main.py)
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
  [[ "$cwd" == "$ROOT" ]] && add_pid "$pid"
done < <(pgrep -f '[Pp]ython.*main\.py' 2>/dev/null || true)

# 3) Absolute-path launch
while read -r pid; do
  add_pid "$pid"
done < <(pgrep -f "$ROOT/main\.py" 2>/dev/null || true)

# Deduplicate
if [[ ${#PIDS[@]} -gt 0 ]]; then
  mapfile -t PIDS < <(printf '%s\n' "${PIDS[@]}" | awk 'NF && !seen[$0]++')
fi

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "No ComfyUI process found for $ROOT (port ${PORT})"
  exit 0
fi

echo "Stopping ComfyUI PIDs: ${PIDS[*]} (port ${PORT})"
kill -TERM "${PIDS[@]}" 2>/dev/null || true

for _ in $(seq 1 20); do
  alive=()
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive+=("$pid")
  done
  [[ ${#alive[@]} -eq 0 ]] && break
  sleep 0.5
done

alive=()
for pid in "${PIDS[@]}"; do
  kill -0 "$pid" 2>/dev/null && alive+=("$pid")
done
if [[ ${#alive[@]} -gt 0 ]]; then
  echo "Force killing: ${alive[*]}"
  kill -KILL "${alive[@]}" 2>/dev/null || true
fi

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi

echo "Stopped ComfyUI for $ROOT (port ${PORT})"

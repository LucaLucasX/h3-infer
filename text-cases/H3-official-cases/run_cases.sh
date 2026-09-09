#!/bin/bash
# usage: run_cases.sh <port> [tasklist] [outdir] [steps]
set -u
PORT=$1
HERE=$(cd "$(dirname "$0")" && pwd)
LIST=${2:-$HERE/tasklist_all.list}
OUT=${3:-$HERE/results}
STEPS=${4:-30}
PY=${PY:-python3}
if ! "$PY" -c 'import requests' >/dev/null 2>&1; then
  echo "error: $PY lacks the requests package; set PY to the project virtualenv Python" >&2
  exit 2
fi
mkdir -p "$OUT"

while IFS='|' read -r name dur aspect; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  out="$OUT/${name}_s${STEPS}.mp4"
  if [ -s "$out" ]; then echo "=== SKIP $name (exists) ==="; continue; fi
  "$PY" "$HERE/../../fl2va.py" --prompt-file "$HERE/${name}.txt" \
    --duration "$dur" --aspect-ratio "$aspect" --steps "$STEPS" \
    --port "$PORT" --out-dir "$OUT" --name "${name}_s${STEPS}" --poll-interval 10
done < "$LIST"

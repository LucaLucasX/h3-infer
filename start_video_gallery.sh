#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
pkill -f "python3 ${ROOT}/video_gallery.py" 2>/dev/null || true
nohup python3 "${ROOT}/video_gallery.py" --host 0.0.0.0 --port 8765 \
  > "${ROOT}/video_gallery.log" 2>&1 &
echo $! > "${ROOT}/video_gallery.pid"
echo "video gallery started: http://127.0.0.1:8765"

#!/usr/bin/env bash
# Submit the same 12s Chinese Bund dialogue scene used in exp_cn_accel_12s.
set -euo pipefail

ROOT=/mnt/luca/H3_infer/experiments/sglang_h3
OUT="${ROOT}/output"
mkdir -p "${OUT}"
PORT="${PORT:-30010}"
BASE="http://127.0.0.1:${PORT}"
STEPS="${STEPS:-20}"
SEED="${SEED:-42}"
DURATION="${DURATION:-12}"

# wait ready
for i in $(seq 1 120); do
  if curl -sS -m 2 "${BASE}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
curl -sS -m 5 "${BASE}/health" >/dev/null

PROMPT=$(python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "/mnt/luca/H3_infer")
sys.path.insert(0, "/mnt/luca/H3_infer/report")
from benchmark_cn_accel_12s import CN_VOICE_TEXT
print(CN_VOICE_TEXT)
PY
)

PAYLOAD=$(python3 - <<PY
import json
prompt = """${PROMPT}"""
print(json.dumps({
  "model": "MiniMaxAI/MiniMax-H3",
  "prompt": prompt,
  "task": "t2va",
  "seconds": ${DURATION},
  "target": {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": float(${DURATION}),
  },
  "num_outputs_per_prompt": 1,
  "num_inference_steps": ${STEPS},
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": ${SEED},
  "fps": 24,
}, ensure_ascii=False))
PY
)

echo "POST ${BASE}/v1/videos  steps=${STEPS} duration=${DURATION}s"
RESP=$(curl -sS -X POST "${BASE}/v1/videos" -H 'Content-Type: application/json' -d "${PAYLOAD}")
echo "${RESP}" | python3 -m json.tool | head -40
JOB=$(echo "${RESP}" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("id",""))')
if [[ -z "${JOB}" ]]; then
  echo "failed to get job id" >&2
  exit 1
fi

T0=$(date +%s)
while true; do
  INFO=$(curl -sS "${BASE}/v1/videos/${JOB}")
  STATUS=$(echo "${INFO}" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("status",""))')
  PROG=$(echo "${INFO}" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("progress",""))')
  echo "[$(date +%H:%M:%S)] status=${STATUS} progress=${PROG}"
  if [[ "${STATUS}" == "completed" || "${STATUS}" == "succeeded" || "${STATUS}" == "success" ]]; then
    echo "${INFO}" >"${OUT}/cn_12s_job.json"
    # copy/download result
    python3 - <<PY
import json, shutil, urllib.request
from pathlib import Path
info = json.loads(Path("${OUT}/cn_12s_job.json").read_text())
out = Path("${OUT}") / f"cn_12s_seed${SEED}_steps${STEPS}.mp4"
src = info.get("file_path") or info.get("url")
if src and str(src).startswith("http"):
    urllib.request.urlretrieve(src, out)
elif src and Path(src).exists():
    shutil.copy2(src, out)
else:
    # try content endpoint
    import urllib.request
    urllib.request.urlretrieve(f"${BASE}/v1/videos/${JOB}/content", out)
print("saved", out, "bytes", out.stat().st_size if out.exists() else 0)
print("wall_sec", int($(date +%s) - ${T0}))
print(json.dumps({k: info.get(k) for k in ("id","status","file_path","url","seconds","size","error")}, ensure_ascii=False, indent=2))
PY
    break
  fi
  if [[ "${STATUS}" == "failed" || "${STATUS}" == "error" ]]; then
    echo "${INFO}" >&2
    exit 1
  fi
  sleep 10
done

#!/usr/bin/env bash
# Orchestrate Comfy (EXP :8190) then SGLang TP2 speed for triway 10-scene bench.
# Stops the other stack so GPU ownership is exclusive / timings fair.
set -euo pipefail
ROOT=/mnt/luca/H3_infer
LOGDIR="$ROOT/experiments/sparse_attn/output"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/triway_12s_10scenes_run.log"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date -Is) triway start ========"

echo "=== stop SGLang (free both GPUs for Comfy) ==="
bash "$ROOT/experiments/sglang_h3/launch_tp2.sh" stop || true
sleep 3

echo "=== start EXP Comfy :8190 on GPU1 ==="
bash "$ROOT/experiments/sparse_attn/scripts/run_comfy_exp.sh" stop || true
sleep 2
bash "$ROOT/experiments/sparse_attn/scripts/run_comfy_exp.sh"
# wait health
for i in $(seq 1 90); do
  if curl -sS -m 2 http://127.0.0.1:8190/system_stats >/dev/null 2>&1; then
    echo "Comfy ready"
    break
  fi
  sleep 2
  (( i == 90 )) && { echo "Comfy failed to start"; tail -40 "$ROOT/experiments/sparse_attn/output/comfyui_exp_8190.log"; exit 1; }
done

echo "=== Comfy phase (warmup + 10 scenes × 2 configs) ==="
"$ROOT/.venv_sage_bench/bin/python" "$ROOT/report/benchmark_triway_12s_10scenes.py" --phase comfy

echo "=== stop Comfy; start SGLang speed ==="
bash "$ROOT/experiments/sparse_attn/scripts/run_comfy_exp.sh" stop || true
sleep 3
bash "$ROOT/experiments/sglang_h3/launch_tp2.sh"
PID=$(cat "$ROOT/experiments/sglang_h3/output/sglang_tp2.pid")
LOGS="$ROOT/experiments/sglang_h3/output/sglang_tp2_sage_turbo_speed.log"
for i in $(seq 1 180); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "SGLang died"; tail -40 "$LOGS"; exit 1
  fi
  if grep -q 'fired up and ready to roll' "$LOGS" 2>/dev/null; then
    echo "SGLang ready"
    break
  fi
  sleep 5
  (( i == 180 )) && { echo TIMEOUT; exit 1; }
done

echo "=== SGLang phase (warmup + 10 scenes) ==="
"$ROOT/experiments/sglang_h3/.venv/bin/python" "$ROOT/report/benchmark_triway_12s_10scenes.py" --phase sglang

echo "======== $(date -Is) triway done ========"
echo "videos: $ROOT/ComfyUI-master_cp/output/exp_triway_12s_10scenes/"
echo "meta:   $ROOT/experiments/sparse_attn/output/ab_triway_12s_10scenes.json"
echo "log:    $LOG"

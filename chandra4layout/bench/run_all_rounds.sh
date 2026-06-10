#!/usr/bin/env bash
# Chạy round2 plain (server đang chạy) → restart optimized → round2 optimized → summarize.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/bench"
PY="${PYTHON:-python}"

echo "=== Round 2 plain (vLLM đang chạy) ==="
"$PY" "$BENCH/run_vllm_bench.py" \
  --scenarios-file "$BENCH/scenarios_round2.yaml" \
  --tag round2_plain \
  --vllm-profile plain

echo ""
echo "=== Restart vLLM optimized ==="
pkill -f 'vllm serve.*8000' 2>/dev/null || true
sleep 3

source /media/drive-2t/miniconda3/etc/profile.d/conda.sh
conda activate personal_lab
chmod +x "$BENCH/scripts/start_vllm_optimized.sh"
CUDA_VISIBLE_DEVICES=1 "$BENCH/scripts/start_vllm_optimized.sh" \
  2>&1 | tee "$BENCH/results/vllm_optimized_server.log" &
VLLM_PID=$!

echo "Đợi vLLM optimized khởi động..."
for i in $(seq 1 120); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "vLLM ready sau ${i}x5s"
    break
  fi
  sleep 5
done

if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
  echo "ERROR: vLLM optimized không lên được" >&2
  exit 1
fi

echo ""
echo "=== Round 2 optimized ==="
"$PY" "$BENCH/run_vllm_bench.py" \
  --scenarios-file "$BENCH/scenarios_round2_optimized.yaml" \
  --tag round2_optimized \
  --vllm-profile optimized

echo ""
echo "=== Summarize ==="
"$PY" "$BENCH/summarize_improvements.py"

echo ""
echo "Báo cáo: $BENCH/results/improvement_report.json"

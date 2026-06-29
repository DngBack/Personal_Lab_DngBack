#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Start a stock vLLM server (NO dng_opt patch) for baseline measurement.
#
# Environment variables (all optional — defaults shown):
#   MODEL                   Qwen/Qwen3.5-30B-A3B
#   TENSOR_PARALLEL_SIZE    1
#   MAX_MODEL_LEN           32768
#   GPU_MEM_UTIL            0.90
#   PORT                    8000
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-30B-A3B}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM="${GPU_MEM_UTIL:-0.90}"
PORT="${PORT:-8000}"

# Disable the packed recurrent decode fast-path so the baseline uses the
# general fused_sigmoid_gating_delta_rule_update path (same as the competition
# server default).  Remove this line to benchmark the already-optimised path.
export VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1

echo "=== dng-opt: baseline server ==="
echo "  model   : $MODEL"
echo "  TP      : $TP"
echo "  max-len : $MAX_LEN"
echo "  port    : $PORT"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --port "$PORT" \
    --mamba-cache-mode align \
    --disable-log-requests

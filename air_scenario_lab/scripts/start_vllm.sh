#!/usr/bin/env bash
# Khởi động vLLM và ghi log ra file text (đồng thời in ra terminal).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="${VLLM_LOG_DIR:-$ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${VLLM_LOG_FILE:-$LOG_DIR/vllm_${TIMESTAMP}.log}"
LATEST_LINK="${LOG_DIR}/vllm_latest.log"

if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION}"
else
  GPU_UTIL="0.85"
fi

mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="${GPUS}"

{
  echo "=== vLLM start $(date -Iseconds) ==="
  echo "Model: ${MODEL}"
  echo "max-model-len: ${MAX_MODEL_LEN}"
  echo "GPU(s): ${CUDA_VISIBLE_DEVICES}  TP=${TP}  mem_util=${GPU_UTIL}  port=${PORT}"
  echo "Log file: ${LOG_FILE}"
  echo "==================================="
} | tee "$LOG_FILE"

ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LINK"

vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype auto \
  2>&1 | tee -a "$LOG_FILE"

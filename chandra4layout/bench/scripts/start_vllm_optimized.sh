#!/usr/bin/env bash
# vLLM với tuning cơ bản cho extraction workload (không quantize).
set -euo pipefail
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${VLLM_PORT:-8000}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

export CUDA_VISIBLE_DEVICES="${GPU}"
echo "=== vLLM optimized serve ==="
echo "Model: ${MODEL}  GPU: ${GPU}  port: ${PORT}"
echo "max-model-len: ${MAX_MODEL_LEN}  max-num-seqs: ${MAX_NUM_SEQS}  mem_util: ${GPU_UTIL}"
exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --dtype auto

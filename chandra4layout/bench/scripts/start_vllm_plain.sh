#!/usr/bin/env bash
# vLLM baseline — không thêm flag tối ưu (chỉ model + port).
set -euo pipefail
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${VLLM_PORT:-8000}"
echo "=== vLLM plain serve ==="
echo "Model: ${MODEL}  Port: ${PORT}"
exec vllm serve "${MODEL}" --port "${PORT}" --host 0.0.0.0

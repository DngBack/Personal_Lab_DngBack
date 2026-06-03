#!/usr/bin/env bash
# Dừng process vLLM cũ (thường sót sau lần start_vllm thất bại) để giải phóng VRAM.
set -euo pipefail

echo "=== GPU processes (trước khi dọn) ==="
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>/dev/null || true

pids="$(pgrep -f 'vllm serve|VLLM::Worker|EngineCore' 2>/dev/null || true)"
if [[ -z "${pids}" ]]; then
  echo "Không thấy process vLLM — không cần dọn."
  exit 0
fi

echo "Sẽ gửi SIGTERM tới: ${pids}"
kill ${pids} 2>/dev/null || true
sleep 3

still="$(pgrep -f 'vllm serve|VLLM::Worker|EngineCore' 2>/dev/null || true)"
if [[ -n "${still}" ]]; then
  echo "Còn process, gửi SIGKILL: ${still}"
  kill -9 ${still} 2>/dev/null || true
  sleep 1
fi

echo "=== GPU sau khi dọn ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

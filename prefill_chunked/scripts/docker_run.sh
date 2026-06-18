#!/usr/bin/env bash
# Run built image (needs 4 GPUs + model mount).
set -euo pipefail

IMAGE="${IMAGE:-llm-opt-challenge:tp2-short-long}"
MODEL_MOUNT="${MODEL_MOUNT:-./models/gpt-oss-20b}"
HOST_PORT="${HOST_PORT:-8000}"

if [[ "${BASELINE_JIT_MODE:-0}" =~ ^(1|true|yes|on)$ ]]; then
  CONFIG="${BASELINE_CONFIG_PATH:-/app/configs/config_tp2_triton_attn.yaml}"
  JIT_ENV=(-e "BASELINE_JIT_MODE=1" -e "BASELINE_JIT_WARMUP=1")
else
  CONFIG="${BASELINE_CONFIG_PATH:-/app/configs/baseline.yaml}"
  JIT_ENV=()
fi

if [[ -n "${BASELINE_CONFIG_PROFILE:-}" ]]; then
  JIT_ENV+=(-e "BASELINE_CONFIG_PROFILE=${BASELINE_CONFIG_PROFILE}")
fi

if [[ ! -d "$MODEL_MOUNT" ]]; then
  echo "Model not found: $MODEL_MOUNT" >&2
  echo "Set MODEL_MOUNT to your gpt-oss-20b directory." >&2
  exit 1
fi

docker run --rm -it \
  --gpus all \
  -p "${HOST_PORT}:8000" \
  -v "${MODEL_MOUNT}:/models/gpt-oss-20b:ro" \
  -e "BASELINE_CONFIG_PATH=${CONFIG}" \
  -e "MODEL_PATH=/models/gpt-oss-20b" \
  "${JIT_ENV[@]}" \
  --shm-size=16g \
  --ipc=host \
  "$IMAGE"

#!/usr/bin/env bash
# Khởi động vLLM khớp air_mini_bench length-profile=realistic.
# Qwen2.5-3B: mặc định 1 GPU (tránh NCCL/P2P lỗi trên A30 VM). Dùng TP=2 chỉ khi GPU có P2P ổn định.
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
# Mặc định 1 GPU — 3B đủ nhỏ; TP=2 hay fail NCCL trên topo PHB (không PIX)
TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION}"
else
  GPU_UTIL=""
fi

_pick_emptiest_gpu() {
  python3 - <<'PY'
import subprocess
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
best = None
for line in out.strip().splitlines():
    idx, used, total = [int(x.strip()) for x in line.split(",")]
    free = total - used
    if best is None or free > best[0]:
        best = (free, idx, used, total)
print(best[1])
PY
}

_compute_safe_gpu_util() {
  local gpu_id="${1:-0}"
  local total_mib free_mib used_mib

  if ! command -v nvidia-smi &>/dev/null; then
    echo "0.85"
    return
  fi

  read -r used_mib total_mib free_mib < <(
    nvidia-smi \
      --query-gpu=memory.used,memory.total,memory.free \
      --format=csv,noheader,nounits \
      -i "${gpu_id}" | head -1 | tr ',' ' '
  )

  python3 - <<PY
used, total, free = int(${used_mib}), int(${total_mib}), int(${free_mib})
ratio = free / total * 0.92
print(f"{min(0.9, max(0.55, ratio)):.2f}")
PY
}

_warn_stale_vllm() {
  if pgrep -f 'VLLM::Worker|vllm serve' >/dev/null 2>&1; then
    echo "WARN: Còn process vLLM cũ trên GPU. Chạy: ./scripts/cleanup_vllm.sh" >&2
  fi
}

# TP=1: chỉ bind 1 GPU (tránh nhầm CUDA_VISIBLE_DEVICES=0,1 từ shell)
if [[ "${TP}" == "1" ]]; then
  if [[ "${GPUS}" == *,* ]]; then
    GPUS="${GPUS%%,*}"
    echo "TP=1: chỉ dùng GPU ${GPUS} (bỏ các GPU còn lại trong CUDA_VISIBLE_DEVICES)"
  elif [[ "${GPUS}" == "0" && -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    GPUS="$(_pick_emptiest_gpu)"
  fi
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"

if [[ -z "${GPU_UTIL}" ]]; then
  GPU_UTIL="$(_compute_safe_gpu_util "${GPUS%%,*}")"
  echo "Auto gpu-memory-utilization: ${GPU_UTIL}"
fi

_warn_stale_vllm

EXTRA_ARGS=()
if [[ "${TP}" -gt 1 ]]; then
  # VM / A30 PHB: thử tắt P2P NCCL nếu bắt buộc TP>1
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
  EXTRA_ARGS+=(--disable-custom-all-reduce)
  echo "TP=${TP}: đã set NCCL_P2P_DISABLE=1 (nếu vẫn lỗi → VLLM_TENSOR_PARALLEL_SIZE=1)"
fi

echo "Model: ${MODEL}"
echo "max-model-len: ${MAX_MODEL_LEN}"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES}  TP=${TP}  mem_util=${GPU_UTIL}  port=${PORT}"

exec vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype auto \
  "${EXTRA_ARGS[@]}"

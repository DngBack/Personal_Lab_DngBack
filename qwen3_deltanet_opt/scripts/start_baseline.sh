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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# vLLM launches EngineCore in a child process.  CUDA cannot be safely
# re-initialised after fork, so default to spawn.  The launcher is a real file,
# which keeps spawn re-imports happy.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_HOME="${HF_HOME:-/home/bachdx2/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM="${GPU_MEM_UTIL:-0.50}"
PORT="${PORT:-8000}"
MAX_SEQS="${MAX_NUM_SEQS:-32}"

# Disable the packed recurrent decode fast-path so the baseline uses the
# general fused_sigmoid_gating_delta_rule_update path (same as the competition
# server default).  Remove this line to benchmark the already-optimised path.
export VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1


echo "=== dng-opt: baseline server ==="
echo "  model   : $MODEL"
echo "  TP      : $TP"
echo "  max-len : $MAX_LEN"
echo "  port    : $PORT"
echo "  hf-cache: $HF_HOME"
echo "  mp      : $VLLM_WORKER_MULTIPROC_METHOD"
echo "  cuda    : $CUDA_VISIBLE_DEVICES"
echo ""

PYTHON="${PYTHON:-/home/bachdx2/.conda/envs/personal_lab/bin/python3}"
exec $PYTHON "$SCRIPT_DIR/_launch_baseline.py"

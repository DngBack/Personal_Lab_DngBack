#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Start a vLLM server with the dng_opt fused Triton kernel applied.
#
# The patch is injected via a tiny Python shim that monkey-patches
# QwenGatedDeltaNetAttention BEFORE the vLLM model is loaded.
#
# Environment variables (all optional — defaults shown):
#   MODEL                   Qwen/Qwen3.5-30B-A3B
#   TENSOR_PARALLEL_SIZE    1
#   MAX_MODEL_LEN           32768
#   GPU_MEM_UTIL            0.90
#   PORT                    8001          <-- different port so both can run
#   DNGOPT_MODE             fused         ("fused" | "instrument")
# ---------------------------------------------------------------------------
set -euo pipefail

# vLLM launches EngineCore in a child process.  CUDA cannot be safely
# re-initialised after fork, so default to spawn.  The real-file launcher also
# applies the monkey patch at module import time so spawned workers inherit it.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_HOME="${HF_HOME:-/home/bachdx2/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM="${GPU_MEM_UTIL:-0.50}"
PORT="${PORT:-8001}"
MAX_SEQS="${MAX_NUM_SEQS:-32}"
export DNGOPT_MODE="${DNGOPT_MODE:-fused}"

# Make sure src/ is on the Python path so dng_opt is importable
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Packed recurrent decode must be enabled for the fused kernel path to activate
# (FusedQwenGDNAttention overrides _forward_core_decode_non_spec which is only
# reached when VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1).
export VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1


echo "=== dng-opt: optimised server (mode=$DNGOPT_MODE) ==="
echo "  model   : $MODEL"
echo "  TP      : $TP"
echo "  max-len : $MAX_LEN"
echo "  port    : $PORT"
echo "  hf-cache: $HF_HOME"
echo "  mp      : $VLLM_WORKER_MULTIPROC_METHOD"
echo "  cuda    : $CUDA_VISIBLE_DEVICES"
echo ""

# Launch via a REAL launcher file (not a stdin heredoc).  vLLM V1 uses the
# 'spawn' start method once CUDA is initialised, and spawned workers re-import
# the main module by file path — a heredoc (__file__ == "<stdin>") makes the
# EngineCore worker die with FileNotFoundError.  The launcher also applies the
# monkey-patch at module top level so it propagates into every spawned worker.
PYTHON="${PYTHON:-/home/bachdx2/.conda/envs/personal_lab/bin/python3}"
exec $PYTHON "$SCRIPT_DIR/_launch_optimized.py"

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL="${MODEL:-Qwen/Qwen3.5-30B-A3B}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM="${GPU_MEM_UTIL:-0.90}"
PORT="${PORT:-8001}"
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
echo ""

# Launch via a REAL launcher file (not a stdin heredoc).  vLLM V1 uses the
# 'spawn' start method once CUDA is initialised, and spawned workers re-import
# the main module by file path — a heredoc (__file__ == "<stdin>") makes the
# EngineCore worker die with FileNotFoundError.  The launcher also applies the
# monkey-patch at module top level so it propagates into every spawned worker.
exec python "$SCRIPT_DIR/_launch_optimized.py"

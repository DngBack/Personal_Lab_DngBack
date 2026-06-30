#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the benchmark against ONE running server and save results to JSON.
#
# Usage:
#   ./scripts/run_bench.sh baseline   # hits port 8000, tag=baseline
#   ./scripts/run_bench.sh fused      # hits port 8001, tag=fused
#   ./scripts/run_bench.sh instrument # hits port 8001, tag=instrument
#
# Environment variables:
#   BASE_URL    override server URL  (default based on tag)
#   MODEL       override model name
#   OUTPUT_DIR  where to write JSON  (default: results/)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

TAG="${1:-baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"

case "$TAG" in
  baseline)
    DEFAULT_URL="http://127.0.0.1:8000"
    ;;
  fused|instrument)
    DEFAULT_URL="http://127.0.0.1:8001"
    ;;
  *)
    echo "Unknown tag '$TAG'.  Use: baseline | fused | instrument"
    exit 1
    ;;
esac

BASE_URL="${BASE_URL:-$DEFAULT_URL}"
OUTPUT_FILE="$OUTPUT_DIR/${TAG}.json"
PYTHON="${PYTHON:-/home/bachdx2/.conda/envs/personal_lab/bin/python3}"

mkdir -p "$OUTPUT_DIR"

echo "=== dng-opt bench: tag=$TAG  url=$BASE_URL ==="

$PYTHON - <<EOF
import asyncio, json, os, sys
from dng_opt.bench.config import BenchConfig
from dng_opt.bench.runner import BenchRunner
from dng_opt.bench.report import BenchReport

cfg = BenchConfig(
    base_url="$BASE_URL",
    api_key="EMPTY",
    model="$MODEL",
    batch_sizes=[1, 4, 8, 16],
    output_tokens=256,
    input_tokens=128,
    warmup_iters=3,
    bench_iters=20,
    run_tag="$TAG",
)

runner = BenchRunner(cfg)
results = asyncio.run(runner.run_all())

report = BenchReport(baseline=results, baseline_tag="$TAG")
report.save_json("$OUTPUT_FILE")
report.print()
print(f"\nResults written to $OUTPUT_FILE")
EOF

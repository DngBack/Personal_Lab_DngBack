#!/usr/bin/env bash
# Chạy tất cả priority suites (cần vLLM).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

PHASE="${1:-phase2}"
MAX="${MAX_REQUESTS:-}"
EXTRA=()
[[ -n "${MAX}" ]] && EXTRA+=(--max-requests "${MAX}")
[[ "${ALL_SUITES:-}" == "1" ]] && EXTRA+=(--all-suites)

export AIR_MINI_BENCH_BASE_URL="${AIR_MINI_BENCH_BASE_URL:-http://127.0.0.1:8000}"
export AIR_MINI_BENCH_MODEL="${AIR_MINI_BENCH_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

python -m bench.run_scenarios --phase "${PHASE}" "${EXTRA[@]}"

echo "Summary: output/${PHASE}/runs/scenario_summary.json"

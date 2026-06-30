#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pretty-print a side-by-side comparison of baseline vs optimised results.
#
# Requires both JSON files to exist (run run_bench.sh for each server first).
#
# Usage:
#   ./scripts/compare.sh
#   ./scripts/compare.sh baseline fused       # explicit tags
#   ./scripts/compare.sh baseline instrument  # compare with instrumented run
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

RESULTS_DIR="$PROJECT_ROOT/results"
TAG_A="${1:-baseline}"
TAG_B="${2:-fused}"

FILE_A="$RESULTS_DIR/${TAG_A}.json"
FILE_B="$RESULTS_DIR/${TAG_B}.json"

for f in "$FILE_A" "$FILE_B"; do
    if [[ ! -f "$f" ]]; then
        echo "Missing results file: $f"
        echo "Run:  ./scripts/run_bench.sh <tag>"
        exit 1
    fi
done

echo "=== dng-opt comparison: $TAG_A vs $TAG_B ==="

python - <<EOF
import json
from dng_opt.bench.report import BenchReport

def load(path):
    with open(path) as f:
        data = json.load(f)
    # Handle both "raw results" format and "report" format
    if "baseline" in data:
        raw = data["baseline"]
    else:
        raw = data
    return {int(bs): v for bs, v in raw.items()}

base = load("$FILE_A")
opt  = load("$FILE_B")

report = BenchReport(
    baseline=base,
    optimized=opt,
    baseline_tag="$TAG_A",
    optimized_tag="$TAG_B",
)
report.print()
report.save_csv("$RESULTS_DIR/${TAG_A}_vs_${TAG_B}.csv")
EOF

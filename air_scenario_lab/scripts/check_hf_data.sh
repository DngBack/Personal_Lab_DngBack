#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_ROOT="${HF_ROOT:-$ROOT/../air_data/data/hf}"

echo "Checking HF data at: $HF_ROOT"
PYTHONPATH="$ROOT/src" python3 -m lab.check_hf --hf-root "$HF_ROOT"

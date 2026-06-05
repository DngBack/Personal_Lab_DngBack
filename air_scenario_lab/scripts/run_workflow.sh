#!/usr/bin/env bash
# Generate + replay one scenario suite.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-configs/scenarios/admission_crunch.yaml}"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
MAX_REQUESTS="${MAX_REQUESTS:-}"

"$ROOT/scripts/check_hf_data.sh"

PYTHONPATH="$ROOT/src" python3 -m lab.generate --config "$ROOT/$CONFIG"

SUITE="$(python3 - <<PY
import yaml
from pathlib import Path
data = yaml.safe_load(Path("$ROOT/$CONFIG").read_text())
print(data["name"])
PY
)"

ARGS=(--suite "$SUITE" --base-url "$BASE_URL" --model "$MODEL")
if [[ -n "$MAX_REQUESTS" ]]; then
  ARGS+=(--max-requests "$MAX_REQUESTS")
fi

PYTHONPATH="$ROOT/src" python3 -m lab.run "${ARGS[@]}"

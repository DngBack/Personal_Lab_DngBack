#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m bench.generate --seed "${SEED:-42}" "$@"

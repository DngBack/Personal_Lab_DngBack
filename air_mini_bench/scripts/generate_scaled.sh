#!/usr/bin/env bash
# Sinh priority scenario suites: P1 (6) + P2 (6). Mỗi suite = trace + payloads + arrival riêng.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

SEED="${SEED:-42}"
PROFILE="${LENGTH_PROFILE:-heavy}"

echo "Generating priority suites (phase1 x6, phase2 x6), profile=${PROFILE}..."
python -m bench.generate --phase all --seed "${SEED}" --length-profile "${PROFILE}"

echo "Dataset analysis per suite..."
python -m bench.analyze --phase all

echo "Done: output/phase1/<suite>/, output/phase2/<suite>/, dataset_analysis.json each"

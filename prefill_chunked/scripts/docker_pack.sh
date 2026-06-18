#!/usr/bin/env bash
# Create a clean zip for submission / offline test (no logs, cache, old tarballs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$(basename "$ROOT")"
OUT="${1:-$(dirname "$ROOT")/${NAME}-docker.zip}"

cd "$(dirname "$ROOT")"
rm -f "$OUT"
zip -r "$OUT" "$NAME" \
  -x "${NAME}/.git/*" \
  -x "${NAME}/logs.txt" \
  -x "${NAME}/metrics.txt" \
  -x "${NAME}/.cursor/*" \
  -x "${NAME}/**/__pycache__/*" \
  -x "${NAME}/**/*.pyc" \
  -x "${NAME}/*.tar.gz" \
  -x "${NAME}/*.zip"

echo "Created: $OUT ($(du -h "$OUT" | awk '{print $1}'))"

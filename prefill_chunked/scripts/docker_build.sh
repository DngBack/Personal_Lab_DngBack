#!/usr/bin/env bash
# Build submission image locally.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-llm-opt-challenge:tp2-short-long}"

cd "$ROOT"
echo "Building ${IMAGE} ..."
docker build -t "$IMAGE" -f Dockerfile .

echo ""
echo "Done. Run with:"
echo "  MODEL_MOUNT=/path/to/gpt-oss-20b docker compose up"
echo "Or (default short/long flash attention):"
echo "  docker run --rm -it --gpus all -p 8000:8000 \\"
echo "    -v /path/to/gpt-oss-20b:/models/gpt-oss-20b:ro \\"
echo "    --shm-size=16g --ipc=host ${IMAGE}"
echo "Or (Triton + JIT warmup):"
echo "  BASELINE_JIT_MODE=1 MODEL_MOUNT=/path/to/gpt-oss-20b ./scripts/docker_run.sh"

#!/usr/bin/env bash
# Parse 2 trang đầu của 4 PDF nội trú bằng Chandra OCR 2.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PDF_SRC="${PDF_SRC:-/home/admin1/Downloads/noitru-pdf/noitru-pdf}"
DEVICE="${DEVICE:-cuda:1}"

# Copy PDF nếu chưa có trong data/pdfs
bash "$ROOT/scripts/copy_pdfs.sh" "$PDF_SRC"

cd "$ROOT/.."
python3 chandra_noitru/run_parse.py \
  --input-dir "$ROOT/data/pdfs" \
  --output-dir "$ROOT/results" \
  --max-pages 2 \
  --device-map "$DEVICE"

#!/usr/bin/env bash
# Copy PDF nội trú từ máy nguồn (đường dẫn mặc định) vào data/pdfs/
set -euo pipefail

SRC_DIR="${1:-/home/admin1/Downloads/noitru-pdf/noitru-pdf}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/pdfs"

mkdir -p "$DEST"

for f in \
  2300030376-6263.pdf \
  2300033911-5962.pdf \
  2500064072-3637.pdf \
  2500077856-33.pdf
do
  if [[ -f "$SRC_DIR/$f" ]]; then
    cp -v "$SRC_DIR/$f" "$DEST/"
  else
    echo "Thiếu: $SRC_DIR/$f" >&2
  fi
done

echo "PDF trong $DEST:"
ls -la "$DEST"/*.pdf 2>/dev/null || echo "(chưa có file .pdf)"

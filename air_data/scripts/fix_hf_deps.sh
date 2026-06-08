#!/usr/bin/env bash
# Fix datasets / huggingface_hub version mismatch (BucketNotFoundError import).
set -euo pipefail

echo "Python: $(which python3)"
python3 -V
echo "Before:"
python3 -m pip show datasets huggingface_hub 2>/dev/null | grep -E '^(Name|Version):' || true

python3 -m pip uninstall -y datasets huggingface_hub huggingface-hub 2>/dev/null || true

python3 -m pip install --no-cache-dir \
  "datasets==3.5.1" \
  "huggingface_hub==0.34.4" \
  "charset-normalizer>=3.0" \
  "pyarrow>=15.0.0"

echo ""
echo "After:"
python3 -m pip show datasets huggingface_hub | grep -E '^(Name|Version):'

python3 - <<'PY'
import huggingface_hub, datasets
print("hub:", huggingface_hub.__version__)
print("datasets:", datasets.__version__)
from datasets import load_dataset
print("import datasets: OK")
PY

echo ""
echo "Done. Re-run:"
echo "  python3 src/data/down_data.py L4NLP/LEval --config gsm100"

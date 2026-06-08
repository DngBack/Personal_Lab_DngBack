#!/usr/bin/env bash
# Pin HF stack compatible with BOTH:
#   - air_data down_data.py (datasets)
#   - vLLM 0.22.x (transformers < 5, huggingface_hub < 1.0)
set -euo pipefail

echo "Python: $(which python3)"
python3 -V
echo "Before:"
python3 -m pip show datasets huggingface_hub transformers 2>/dev/null | grep -E '^(Name|Version):' || true

python3 -m pip uninstall -y datasets huggingface_hub huggingface-hub transformers 2>/dev/null || true

python3 -m pip install --no-cache-dir \
  "datasets==4.4.1" \
  "huggingface_hub==0.34.4" \
  "transformers==4.57.1" \
  "charset-normalizer>=3.0" \
  "pyarrow>=15.0.0"

echo ""
echo "After:"
python3 -m pip show datasets huggingface_hub transformers | grep -E '^(Name|Version):'

python3 - <<'PY'
import huggingface_hub, datasets, transformers
from datasets import load_dataset
print("hub:", huggingface_hub.__version__)
print("datasets:", datasets.__version__)
print("transformers:", transformers.__version__)
print("import OK")
PY

echo ""
echo "Done."
echo "  vLLM:      ./scripts/start_vllm.sh"
echo "  download:  python3 src/data/down_data.py L4NLP/LEval --config gsm100"

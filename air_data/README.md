# air_data

## Setup

```bash
cd air_data
pip install -r requirements.txt
```

## CLI

Preview a dataset:

```bash
python3 src/data/down_data.py L4NLP/LEval --config gsm100 --preview
```

Download one config:

```bash
python3 src/data/down_data.py L4NLP/LEval --config gsm100
```

Download all configs in a repo:

```bash
python3 src/data/down_data.py L4NLP/LEval --all-configs
```

Download every repo listed in `data/data_hf.txt`:

```bash
python3 src/data/down_data.py --from-file
```

Download raw files from the Hub (skip `datasets.load_dataset`):

```bash
python3 src/data/down_data.py L4NLP/LEval --config gsm100 --hub-files
```

Optional flags:

```bash
python3 src/data/down_data.py <repo_id> \
  --config <config_name> \
  --split <split_name> \
  --output-dir <path> \
  --format jsonl|parquet|csv \
  --cache-dir <path>
```

## Python

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("air_data/src")))
from data.down_data import download_hf_dataset, preview_dataset

preview_dataset("L4NLP/LEval", config_name="gsm100")

exported = download_hf_dataset("L4NLP/LEval", config_name="gsm100")
for key, path in exported.items():
    print(key, path)
```

## Output

Default output directory:

```
air_data/data/hf/<repo_id>/
```

Example:

```
air_data/data/hf/L4NLP__LEval/LEval/Exam/gsm100.jsonl
```

Metadata is written to:

```
air_data/data/hf/<repo_id>/dataset_info.json
```

## Analyze (contest prep)

After downloads, summarize row counts, context lengths, and task hints:

```bash
python3 src/data/analyze_data.py
python3 src/data/analyze_data.py --json-out data/contest_report.json
```

## Inference optimization (researcher view)

Long-context means, percentiles, document/prefix reuse, and prefill vs decode budgeting:

```bash
python3 src/data/analyze_inference.py
python3 src/data/analyze_inference.py --json-out data/inference_research_report.json
```

`ShareChat` is gated on the Hub — run `huggingface-cli login` and accept the dataset license before downloading.

## Repo list

Edit `data/data_hf.txt` (one Hugging Face repo id per line):

```
tucnguyen/ShareChat
L4NLP/LEval
bigai-nlco/LooGLE
```

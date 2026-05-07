# LoRA Switcher Baseline (One LoRA Per Request)

This demo validates a simple pipeline:

- Base model: `Qwen/Qwen3-0.6B`
- Inference engine: `vLLM`
- Routing: `doc_type -> exactly one LoRA adapter`

## Install

```bash
cd LoRASwitcher
conda activate personal_lab
pip install -r requirements.txt
```

## Adapter config

Edit `adapters.json` if you want different adapters.

Current file contains 5 public adapter candidates for pipeline validation.

These adapters are selected for the Qwen3-0.6B family and include adapter files
(`adapter_config.json`, `adapter_model.safetensors`) for direct LoRA loading.

## Run single request (exactly one LoRA)

```bash
python router_demo.py \
  --doc-type tldr \
  --prompt "Summarize this income statement into 3 bullets." \
  --max-tokens 128 \
  --cache-dir ./.hf_cache \
  --gpu-memory-utilization 0.15 \
  --max-model-len 4096
```

## Run grouped benchmark

```bash
python router_demo.py \
  --doc-type function_calling \
  --prompt "Write SQL to compute weekly active users." \
  --bench-grouped \
  --cache-dir ./.hf_cache \
  --gpu-memory-utilization 0.15 \
  --max-model-len 4096
```

Grouped benchmark batches requests by `doc_type` to reduce LoRA switching overhead, while each request still uses exactly one LoRA.

## Validate all adapters in one process

This is the quickest way to verify multi-adapter inference logic and catch broken adapters:

```bash
python router_demo.py \
  --doc-type tldr \
  --prompt "warmup request" \
  --validate-all-adapters \
  --max-tokens 64 \
  --cache-dir ./.hf_cache \
  --gpu-memory-utilization 0.20 \
  --max-model-len 768
```

The script prints per-adapter metrics:
- `switch_ms`: time spent resolving/downloading/switching adapter
- `infer_ms`: generation time for that adapter request
- `ok/fail`: adapter health in current environment

## If you see cache permission errors

Use a writable cache directory (already supported by script):

```bash
python router_demo.py --doc-type tldr --prompt "test" --cache-dir /tmp/lora_cache
```

## If you see GPU memory startup errors

The message `free memory ... less than desired GPU memory utilization` means another process is occupying most VRAM.

Try lower values:

```bash
python router_demo.py \
  --doc-type tldr \
  --prompt "test prompt" \
  --max-tokens 64 \
  --cache-dir ./.hf_cache \
  --gpu-memory-utilization 0.12 \
  --max-model-len 2048
```

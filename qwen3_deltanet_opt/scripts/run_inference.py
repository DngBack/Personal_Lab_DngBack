"""
Quick inference test for Qwen3.5-2B with/without dng_opt fused kernel.

Usage
-----
# baseline (no patch):
  python scripts/run_inference.py

# fused Triton kernel:
  DNGOPT_MODE=fused python scripts/run_inference.py

# instrumented (per-stage timing):
  DNGOPT_MODE=instrument python scripts/run_inference.py
"""

from __future__ import annotations

import os
import sys
import time

# Ensure src/ is on path when run directly
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))

os.environ.setdefault("HF_HOME", "/home/bachdx2/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MODEL = os.environ.get(
    "MODEL",
    "/home/bachdx2/hf_cache/hub/models--Qwen--Qwen3.5-2B/snapshots/"
    "15852e8c16360a2fea060d615a32b45270f8a8fc",
)
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "1024"))
GPU_MEM_UTIL  = float(os.environ.get("GPU_MEM_UTIL", "0.5"))
MAX_NUM_SEQS  = int(os.environ.get("MAX_NUM_SEQS", "8"))
MODE          = os.environ.get("DNGOPT_MODE", "").lower()

# Apply patch BEFORE importing vllm model classes
if MODE in ("fused", "instrument"):
    from dng_opt.patch import apply_patch
    apply_patch(MODE)
    print(f"[dng_opt] patch applied: mode={MODE}")
else:
    print("[dng_opt] no patch (baseline)")

from vllm import LLM, SamplingParams  # noqa: E402 — after patch

PROMPTS = [
    "The capital of France is",
    "In machine learning, a neural network is",
    "The theory of relativity states that",
    "Once upon a time, in a land far away,",
]

SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=64)


def main() -> None:
    print(f"\nLoading model: {MODEL}")
    print(f"  max_model_len={MAX_MODEL_LEN}, gpu_memory_utilization={GPU_MEM_UTIL}, "
          f"max_num_seqs={MAX_NUM_SEQS}\n")

    llm = LLM(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=False,
    )

    # Warm-up
    llm.generate(PROMPTS[:1], SAMPLING_PARAMS)

    # Timed run
    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"Mode : {MODE or 'baseline'}")
    print(f"Time : {elapsed*1000:.1f} ms  ({len(PROMPTS)} prompts)")
    print(f"{'='*60}")
    for out in outputs:
        print(f"\n>> {out.prompt!r}")
        print(f"   {out.outputs[0].text!r}")


if __name__ == "__main__":
    main()

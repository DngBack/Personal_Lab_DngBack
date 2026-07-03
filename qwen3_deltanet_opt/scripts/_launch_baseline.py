"""
Baseline server launcher — mirrors _launch_optimized.py but applies NO patch.

Why use a file instead of `python -m vllm.entrypoints.openai.api_server`?
--------------------------------------------------------------------------
The `__main__` entry-point calls `uvloop.run(run_server(...))`.  On this
setup, using a real launcher file plus `asyncio.run` gives us explicit control
over environment variables before vLLM imports and keeps multiprocessing
`spawn` re-imports pointed at an actual Python file.
"""

from __future__ import annotations

import os

# CUDA cannot be safely re-initialised in a forked subprocess once the parent
# has imported CUDA-touching libraries, so use spawn unless the caller opts out.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("HF_HOME", "/home/bachdx2/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])

if __name__ == "__main__":
    import argparse
    import asyncio

    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    parser = make_arg_parser(argparse.ArgumentParser())
    args = parser.parse_args([
        "--model",                  os.environ.get("MODEL", "Qwen/Qwen3.5-2B"),
        "--tensor-parallel-size",   os.environ.get("TENSOR_PARALLEL_SIZE", "1"),
        "--max-model-len",          os.environ.get("MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEM_UTIL", "0.50"),
        "--max-num-seqs",           os.environ.get("MAX_NUM_SEQS", "32"),
        "--port",                   os.environ.get("PORT", "8000"),
        "--mamba-cache-mode",       "align",
        "--no-enable-log-requests",
    ])
    asyncio.run(run_server(args))

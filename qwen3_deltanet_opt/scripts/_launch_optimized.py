"""
Real-file launcher for the dng_opt optimised vLLM server.

Why a file instead of a `python - <<EOF` heredoc?
-------------------------------------------------
vLLM V1 forces the ``spawn`` multiprocessing start method once CUDA is
initialised.  Under ``spawn`` each worker process re-imports the *main module
by file path*.  A heredoc has no file path (``__file__`` == ``<stdin>``), so
the EngineCore worker dies with ``FileNotFoundError: .../<stdin>``.

Putting ``apply_patch()`` at module top level also makes the monkey-patch
propagate into every spawned worker: the worker re-imports this file as
``__mp_main__`` (so the top-level code runs and patches the class), while the
server-launch code below stays guarded by ``if __name__ == "__main__":`` and
is therefore skipped in workers.
"""

from __future__ import annotations

import os

# Runs in the parent AND in every spawned worker (imported as __mp_main__),
# so the EngineCore subprocess that actually builds the model is patched too.
from dng_opt.patch import apply_patch

apply_patch()


if __name__ == "__main__":
    import argparse
    import asyncio

    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    parser = make_arg_parser(argparse.ArgumentParser())
    args = parser.parse_args([
        "--model",                  os.environ.get("MODEL", "Qwen/Qwen3.5-30B-A3B"),
        "--tensor-parallel-size",   os.environ.get("TENSOR_PARALLEL_SIZE", "1"),
        "--max-model-len",          os.environ.get("MAX_MODEL_LEN", "32768"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEM_UTIL", "0.90"),
        "--port",                   os.environ.get("PORT", "8001"),
        "--mamba-cache-mode",       "align",
        "--no-enable-log-requests",
    ])
    asyncio.run(run_server(args))

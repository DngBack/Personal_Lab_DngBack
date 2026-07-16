#!/usr/bin/env python3
"""Bake a smart default for long_prefill_token_threshold directly into
vLLM 0.24.0's SchedulerConfig, so no CLI flag is needed.

Why: source-read of vllm/v1/core/sched/scheduler.py (both the RUNNING loop
around line 468 and the WAITING loop around line 797) confirmed the exact
mechanism this project has documented since 2026-07-07: when
long_prefill_token_threshold is 0 (the untouched default), a single
big-prompt request's chunk is capped only by the step's total
max_num_batched_tokens -- so the first large request in a wave can consume
the ENTIRE per-step token budget, leaving zero room for its wave-mates that
step. This is a step-budget monopolization bug, not a cache-build latency
issue.

The project's own history (contest-phase1-optimization-levers.md) singles
out a STATIC long_prefill_token_threshold as "the only lever that was
directionally non-negative in every single test (never regressed TBT/ERC)",
in contrast to dynamic/adaptive variants (DNG_INTERLEAVE) which were reverted
after a real-world regression the user hit personally. This patch
deliberately stays in the static-threshold family, not a new adaptive
scheme.

Mechanism: extend SchedulerConfig.__post_init__ so that whenever
long_prefill_token_threshold is left at 0 (its default) AND chunked prefill
is enabled, it self-derives to max_num_batched_tokens // 2 -- guaranteeing
at least two concurrently-admitted requests can share a single step's
budget instead of one monopolizing it, and scaling automatically with
whatever --max-num-batched-tokens value is actually configured (no magic
absolute number to re-tune if the budget changes). Explicitly setting
--long-prefill-token-threshold on the CLI still overrides this (only fires
when the value is exactly 0).
"""

from __future__ import annotations

import argparse
import hashlib
import py_compile
from pathlib import Path


PATCH_MARKER = "# R13_SCHED_THRESHOLD_DEFAULT_PATCH"


class PatchError(RuntimeError):
    pass


def find_target(vllm_root: Path) -> Path:
    return vllm_root / "vllm" / "config" / "scheduler.py"


def find_vllm_root(explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root)
        return root if (root / "vllm").is_dir() else root.parent
    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise PatchError("Unable to locate installed vllm package")
    return Path(spec.origin).resolve().parent.parent


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(
            f"Expected exactly one anchor for {description}, found {count}. "
            "vLLM source has likely changed -- refusing to patch blindly."
        )
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_root", nargs="?", default=None)
    args = parser.parse_args()

    vllm_root = find_vllm_root(args.vllm_root)
    target = find_target(vllm_root)
    if not target.exists():
        raise PatchError(f"Missing target file: {target}")

    text = target.read_text()
    if PATCH_MARKER in text:
        print(f"already patched: {target}")
        return 0

    anchor = '''        if self.max_num_partial_prefills > 1:
            if self.long_prefill_token_threshold == 0:
                self.long_prefill_token_threshold = int(max_model_len * 0.04)

            logger.info(
                "Concurrent partial prefills enabled with "
                "max_num_partial_prefills=%d, max_long_partial_prefills=%d, "
                "long_prefill_token_threshold=%d",
                self.max_num_partial_prefills,
                self.max_long_partial_prefills,
                self.long_prefill_token_threshold,
            )

        self.verify_max_model_len(max_model_len)'''

    replacement = f'''        if self.max_num_partial_prefills > 1:
            if self.long_prefill_token_threshold == 0:
                self.long_prefill_token_threshold = int(max_model_len * 0.04)

            logger.info(
                "Concurrent partial prefills enabled with "
                "max_num_partial_prefills=%d, max_long_partial_prefills=%d, "
                "long_prefill_token_threshold=%d",
                self.max_num_partial_prefills,
                self.max_long_partial_prefills,
                self.long_prefill_token_threshold,
            )

        {PATCH_MARKER}
        # Baked default (no CLI flag needed): with chunked prefill on and no
        # explicit --long-prefill-token-threshold, a single running/waiting
        # request's chunk is otherwise capped only by the step's total
        # max_num_batched_tokens -- the first big prompt in a wave can eat
        # the WHOLE step budget and starve its wave-mates that step. Cap any
        # one request to half the step budget by default so at least two
        # requests can always share a step. Scales with whatever
        # max_num_batched_tokens is configured; an explicit CLI flag value
        # (anything != 0) still wins.
        if (
            self.enable_chunked_prefill
            and self.long_prefill_token_threshold == 0
            and self.max_num_batched_tokens >= 2
        ):
            self.long_prefill_token_threshold = max(
                1, self.max_num_batched_tokens // 2
            )
            logger.info(
                "R13 baked long_prefill_token_threshold=%d (max_num_batched_tokens=%d // 2)",
                self.long_prefill_token_threshold,
                self.max_num_batched_tokens,
            )

        self.verify_max_model_len(max_model_len)'''

    text = replace_once(text, anchor, replacement, "__post_init__ threshold default")
    target.write_text(text)
    py_compile.compile(str(target), doraise=True)
    print(f"patched {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

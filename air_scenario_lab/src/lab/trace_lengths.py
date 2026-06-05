from __future__ import annotations

import random
from typing import Any

from .config import (
    COMPACT_INPUT_CAP,
    COMPACT_OUTPUT_CAP,
    LENGTH_PROFILES,
    REALISTIC_INPUT_CAP,
    REALISTIC_OUTPUT_CAP,
)


def length_caps(length_profile: str) -> tuple[dict[str, int], dict[str, int]] | tuple[None, None]:
    if length_profile in ("local", "compact"):
        return COMPACT_INPUT_CAP, COMPACT_OUTPUT_CAP
    if length_profile in ("realistic", "heavy"):
        return REALISTIC_INPUT_CAP, REALISTIC_OUTPUT_CAP
    return None, None


def sample_length(
    rng: random.Random,
    workload: str,
    kind: str,
    *,
    length_profile: str,
    max_context_tokens: int,
    safety_tokens: int,
    target_in_hint: int | None = None,
) -> int:
    prof = LENGTH_PROFILES[workload]
    median = prof[f"{kind}_median"]
    p95 = prof[f"{kind}_p95"]
    input_caps, output_caps = length_caps(length_profile)
    caps_in = REALISTIC_INPUT_CAP if input_caps is None else input_caps

    if (
        length_profile in ("realistic", "heavy")
        and workload == "long_context"
        and kind == "input"
        and rng.random() < (0.55 if length_profile == "heavy" else 0.35)
    ):
        lo, hi = 20_000, caps_in["long_context"]
    elif length_profile == "heavy" and workload == "tool_agent" and kind == "input":
        if rng.random() < 0.45:
            lo, hi = int(median * 1.05), caps_in["tool_agent"]
        elif rng.random() < 0.85:
            lo, hi = int(median * 0.9), int(median * 1.15)
        else:
            lo, hi = int(median * 1.1), p95
    elif length_profile == "heavy" and kind == "input":
        lo, hi = int(median * 0.85), int(median * 1.2)
    elif rng.random() < 0.95:
        lo, hi = int(median * 0.7), int(median * 1.3)
    else:
        lo, hi = int(median * 1.1), p95

    if input_caps is not None and kind == "input":
        cap = input_caps[workload]
        hi = min(hi, cap)
        lo = min(lo, cap)
    if output_caps is not None and kind == "output":
        hi = min(hi, output_caps[workload])

    hi = max(lo, hi)
    value = max(16, rng.randint(lo, hi))

    if output_caps is not None and kind == "output":
        value = min(value, output_caps[workload])
        if target_in_hint is not None:
            value = min(value, max_context_tokens - safety_tokens - target_in_hint)
        value = max(16, value)
    return value


def fit_lengths(
    workload: str,
    target_in: int,
    target_out: int,
    *,
    length_profile: str,
    max_context_tokens: int,
    safety_tokens: int,
) -> tuple[int, int]:
    input_caps, output_caps = length_caps(length_profile)
    if input_caps is None:
        return target_in, target_out
    budget = max_context_tokens - safety_tokens
    target_in = min(target_in, input_caps[workload], budget - 16)
    target_out = min(target_out, output_caps[workload], budget - target_in)
    target_in = min(target_in, budget - target_out)
    return max(16, target_in), max(16, target_out)


def validate_trace_lengths(
    rows: list[dict[str, Any]],
    max_context_tokens: int,
    safety_tokens: int,
) -> None:
    bad = [
        r
        for r in rows
        if r["input_length"] + r["output_length"] > max_context_tokens - safety_tokens
    ]
    if bad:
        sample = bad[0]
        raise RuntimeError(
            f"Trace violates context budget: {sample['request_id']} "
            f"in={sample['input_length']} out={sample['output_length']}"
        )

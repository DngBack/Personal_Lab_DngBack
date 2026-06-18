#!/usr/bin/env python3
"""Helpers for routing requests to short vs long vLLM pools."""

from __future__ import annotations

import hashlib
import json
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

INFERENCE_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})

# Fast-path margins around split_prompt_tokens (asymmetric: conservative on long).
FAST_SHORT_RATIO = 0.75
FAST_LONG_RATIO = 1.5

# SLO-aware routing thresholds (V1-B).
ROUTE_SHORT_MAX = 6144
ROUTE_AMBIGUOUS_MAX = 12288
ROUTE_LONG_MIN = 20000
PREFIX_HASH_HISTORY = 64
PREFIX_CACHE_BONUS_TOKENS = 500.0
PREFIX_LOAD_OVERRIDE_RATIO = 2.0


def path_only(raw_path: str) -> str:
    return raw_path.split("?", 1)[0]


def extract_prompt_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str):
                            parts.append(text)
        if parts:
            return "\n".join(parts)

    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(str(item) for item in prompt)
    return ""


def estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    text = extract_prompt_text(payload)
    if not text:
        return 0
    # Conservative char/token ratio for gpt-oss style tokenizers.
    return max(1, len(text) // 3)


def count_prompt_tokens_http(
    tokenize_url: str,
    model_name: str,
    payload: dict[str, Any],
    timeout_sec: float = 30.0,
) -> int | None:
    prompt = extract_prompt_text(payload)
    if not prompt:
        return 0

    bodies = [
        {"model": model_name, "messages": payload.get("messages") or [{"role": "user", "content": prompt}]},
        {"model": model_name, "prompt": prompt},
    ]
    for body in bodies:
        try:
            request = urllib.request.Request(
                tokenize_url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            for key in ("count", "num_tokens", "token_count"):
                value = parsed.get(key)
                if isinstance(value, int):
                    return value
            for key in ("tokens", "token_ids", "input_ids"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return len(value)
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
            continue
    return None


def choose_pool(prompt_tokens: int, split_prompt_tokens: int) -> str:
    if prompt_tokens >= split_prompt_tokens:
        return "long"
    return "short"


def resolve_prompt_tokens(
    payload: dict[str, Any],
    *,
    tokenize_url: str | None,
    model_name: str,
    timeout_sec: float = 30.0,
) -> int:
    """Resolve prompt length; prefers HTTP tokenize when a URL is available."""
    if tokenize_url:
        counted = count_prompt_tokens_http(
            tokenize_url, model_name, payload, timeout_sec=timeout_sec
        )
        if counted is not None:
            return counted
    return estimate_prompt_tokens(payload)


def resolve_routing(
    payload: dict[str, Any],
    *,
    split_prompt_tokens: int,
    tokenize_url: str | None,
    model_name: str,
    timeout_sec: float = 30.0,
) -> tuple[str, int, str]:
    """Pick short/long pool with fast-path to skip HTTP tokenize when obvious.

    Returns (pool, prompt_tokens, method) where method is one of:
    fast-short, fast-long, tokenize, estimate.
    """
    estimate = estimate_prompt_tokens(payload)
    short_cutoff = int(split_prompt_tokens * FAST_SHORT_RATIO)
    long_cutoff = int(split_prompt_tokens * FAST_LONG_RATIO)

    if estimate < short_cutoff:
        return "short", estimate, "fast-short"
    if estimate > long_cutoff:
        return "long", estimate, "fast-long"

    if tokenize_url:
        counted = count_prompt_tokens_http(
            tokenize_url, model_name, payload, timeout_sec=timeout_sec
        )
        if counted is not None:
            return choose_pool(counted, split_prompt_tokens), counted, "tokenize"

    pool = choose_pool(estimate, split_prompt_tokens)
    return pool, estimate, "estimate"


@dataclass
class WorkerEndpoint:
    worker_id: str
    pool: str
    host: str
    port: int


@dataclass
class WorkerState:
    inflight: int = 0
    queue_prefill_tokens: int = 0
    ewma_ttft_ms: float = 3100.0
    ewma_itl_ms: float = 7.8
    prefill_tps_ewma: float = 18000.0
    decode_tps_ewma: float = 1500.0
    recent_prefix_hashes: list[str] = field(default_factory=list)


def compute_prefix_hash(payload: dict[str, Any]) -> str:
    text = extract_prompt_text(payload)
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def rendezvous_hash(key: str, worker_ids: list[str]) -> str:
    if not worker_ids:
        raise ValueError("no workers for rendezvous hash")
    best_id = worker_ids[0]
    best_score = -1
    for worker_id in worker_ids:
        digest = hashlib.md5(f"{key}:{worker_id}".encode(), usedforsecurity=False).digest()
        score = struct.unpack(">Q", digest[:8])[0]
        if score > best_score:
            best_score = score
            best_id = worker_id
    return best_id


def record_prefix_hit(state: WorkerState, prefix_hash: str) -> None:
    if not prefix_hash:
        return
    if prefix_hash in state.recent_prefix_hashes:
        state.recent_prefix_hashes.remove(prefix_hash)
    state.recent_prefix_hashes.append(prefix_hash)
    if len(state.recent_prefix_hashes) > PREFIX_HASH_HISTORY:
        state.recent_prefix_hashes = state.recent_prefix_hashes[-PREFIX_HASH_HISTORY:]


def prefix_cache_bonus(state: WorkerState, prefix_hash: str) -> float:
    if prefix_hash and prefix_hash in state.recent_prefix_hashes:
        return PREFIX_CACHE_BONUS_TOKENS / state.prefill_tps_ewma
    return 0.0


def slo_route_score(
    state: WorkerState,
    prompt_tokens: int,
    output_tokens: int,
    prefix_hash: str,
) -> float:
    return (
        state.queue_prefill_tokens / state.prefill_tps_ewma
        + prompt_tokens / state.prefill_tps_ewma
        + output_tokens / state.decode_tps_ewma
        - prefix_cache_bonus(state, prefix_hash)
    )


def slo_candidate_workers(
    prompt_tokens: int,
    workers: list[WorkerEndpoint],
    *,
    split_prompt_tokens: int,
) -> list[WorkerEndpoint]:
    short_workers = [worker for worker in workers if worker.pool == "short"]
    long_workers = [worker for worker in workers if worker.pool == "long"]

    if prompt_tokens >= ROUTE_LONG_MIN:
        return long_workers
    if prompt_tokens >= ROUTE_AMBIGUOUS_MAX:
        return long_workers
    if prompt_tokens >= split_prompt_tokens:
        return long_workers + short_workers
    if prompt_tokens >= ROUTE_SHORT_MAX:
        return short_workers + long_workers
    return short_workers


def select_slo_worker(
    payload: dict[str, Any],
    workers: list[WorkerEndpoint],
    states: dict[str, WorkerState],
    *,
    split_prompt_tokens: int,
    tokenize_url: str | None,
    model_name: str,
    prefix_locality: bool = True,
    timeout_sec: float = 30.0,
) -> tuple[WorkerEndpoint, str, int, str, str]:
    """Pick worker using length + load + optional prefix locality.

    Returns (worker, pool, prompt_tokens, method, prefix_hash).
    """
    pool, prompt_tokens, method = resolve_routing(
        payload,
        split_prompt_tokens=split_prompt_tokens,
        tokenize_url=tokenize_url,
        model_name=model_name,
        timeout_sec=timeout_sec,
    )
    output_tokens = int(payload.get("max_tokens", 256))
    prefix_hash = compute_prefix_hash(payload)
    candidates = slo_candidate_workers(
        prompt_tokens,
        workers,
        split_prompt_tokens=split_prompt_tokens,
    )
    if not candidates:
        raise RuntimeError("no routing candidates available")

    candidate_ids = [worker.worker_id for worker in candidates]
    route_method = method

    if prefix_locality and prefix_hash and len(candidates) > 1:
        sticky_id = rendezvous_hash(prefix_hash, candidate_ids)
        sticky = next(worker for worker in candidates if worker.worker_id == sticky_id)
        sticky_state = states[sticky.worker_id]
        scores = {
            worker.worker_id: slo_route_score(
                states[worker.worker_id],
                prompt_tokens,
                output_tokens,
                prefix_hash,
            )
            for worker in candidates
        }
        min_score = min(scores.values())
        sticky_score = scores[sticky.worker_id]
        if sticky_score <= min_score * PREFIX_LOAD_OVERRIDE_RATIO:
            return sticky, pool, prompt_tokens, f"{method}+prefix", prefix_hash

    best = min(
        candidates,
        key=lambda worker: slo_route_score(
            states[worker.worker_id],
            prompt_tokens,
            output_tokens,
            prefix_hash,
        ),
    )
    return best, pool, prompt_tokens, route_method, prefix_hash


def track_request_start(state: WorkerState, prompt_tokens: int, prefix_hash: str) -> None:
    state.inflight += 1
    state.queue_prefill_tokens += prompt_tokens
    record_prefix_hit(state, prefix_hash)


def track_request_end(state: WorkerState, prompt_tokens: int) -> None:
    state.inflight = max(0, state.inflight - 1)
    state.queue_prefill_tokens = max(0, state.queue_prefill_tokens - prompt_tokens)

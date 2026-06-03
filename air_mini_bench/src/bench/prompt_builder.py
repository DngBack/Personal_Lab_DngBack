from __future__ import annotations

import random
from typing import Any

from .tokens import estimate_token_count, tokens_to_chars


def _filler_sentence(token_seed: int) -> str:
    return (
        f"Document paragraph {token_seed}: the policy states regulatory clause "
        f"{token_seed % 997} applies to section {token_seed % 113}. "
    )


def build_shared_prefix_block(session_id: str, target_prefix_tokens: int) -> str:
    """Prefix text thật — runtime vLLM cache theo token prefix, không theo hash_ids."""
    header = f"[SHARED_SYSTEM_PREFIX_{session_id}]\n"
    header_tokens = estimate_token_count(header)
    need = max(0, target_prefix_tokens - header_tokens)
    body_parts: list[str] = []
    t = 0
    while estimate_token_count("".join(body_parts)) < need:
        body_parts.append(_filler_sentence(hash(session_id) % 10000 + t))
        t += 1
    body = "".join(body_parts)
    combined = header + body
    if estimate_token_count(combined) > target_prefix_tokens:
        combined = combined[: tokens_to_chars(target_prefix_tokens)]
    return combined


def build_tool_suffix(rng: random.Random, question_idx: int) -> tuple[str, str]:
    a, b = rng.randint(10, 999), rng.randint(1, 499)
    suffix = (
        f"\n[USER_QUESTION_{question_idx:04d}]\n"
        f"Compute {a} + {b}. Reply with the integer only."
    )
    return suffix, str(a + b)


def build_tool_agent_prompt(
    *,
    session_id: str,
    prefix_tokens: int,
    total_input_tokens: int,
    rng: random.Random,
    question_idx: int,
) -> tuple[str, str, dict[str, Any]]:
    prefix = build_shared_prefix_block(session_id, prefix_tokens)
    suffix, gold = build_tool_suffix(rng, question_idx)
    prompt = prefix + suffix
    n = estimate_token_count(prompt)
    if n > total_input_tokens:
        prompt = prompt[: tokens_to_chars(total_input_tokens)]
        n = estimate_token_count(prompt)
    meta = {
        "cache_session_id": session_id,
        "prefix_tokens_target": prefix_tokens,
        "prefix_tokens_actual": estimate_token_count(prefix),
        "shared_prefix_verified": prompt.startswith(prefix[: min(64, len(prefix))]),
    }
    return prompt, gold, meta


def build_conversation_prompt(rng: random.Random, turn: int, target_tokens: int) -> str:
    vocab = ["hello", "thanks", "help", "explain", "summarize", "please", "detail"]
    parts = [f"User turn {turn}:"]
    while estimate_token_count(" ".join(parts)) < target_tokens:
        parts.append(rng.choice(vocab))
    text = " ".join(parts)
    if estimate_token_count(text) > target_tokens:
        text = text[: tokens_to_chars(target_tokens)]
    return text


def build_long_context_prompt(rng: random.Random, doc_idx: int, target_tokens: int) -> tuple[str, str]:
    header = f"[LONG_DOC_{doc_idx:05d}]\n"
    parts = [header]
    t = 0
    while estimate_token_count("".join(parts)) < target_tokens - 32:
        parts.append(_filler_sentence(doc_idx * 1000 + t))
        t += 1
    q = f"\n[QUESTION]\nSummarize key fact index {doc_idx} in one sentence."
    prompt = "".join(parts) + q
    if estimate_token_count(prompt) > target_tokens:
        prompt = prompt[: tokens_to_chars(target_tokens)]
    return prompt, f"fact {doc_idx}"

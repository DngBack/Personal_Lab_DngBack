from __future__ import annotations

import math
import zlib

CHARS_PER_TOKEN = 4.0
HASH_BLOCK_TOKENS = 512


def chars_to_tokens(chars: int) -> int:
    return max(1, int(round(chars / CHARS_PER_TOKEN)))


def tokens_to_chars(tokens: int) -> int:
    return max(1, int(round(tokens * CHARS_PER_TOKEN)))


def estimate_token_count(text: str) -> int:
    return chars_to_tokens(len(text))


def truncate_prompt_natural(text: str, max_tokens: int) -> tuple[str, int]:
    """Truncate at sentence/paragraph boundary — never pad with filler."""
    n = estimate_token_count(text)
    if n <= max_tokens:
        return text, n

    max_chars = tokens_to_chars(max_tokens)
    cut = text[:max_chars]
    for sep in ("\n\n", "\n", ". ", "? ", "! "):
        pos = cut.rfind(sep)
        if pos > max_chars // 3:
            cut = cut[: pos + len(sep.rstrip())]
            break
    cut = cut.rstrip()
    return cut, estimate_token_count(cut)


def _token_chunks(text: str, block_tokens: int = HASH_BLOCK_TOKENS) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    approx_words_per_block = max(1, int(block_tokens * 0.75))
    return [
        " ".join(words[i : i + approx_words_per_block])
        for i in range(0, len(words), approx_words_per_block)
    ]


def hash_block(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFF


def hash_block_count(input_length: int, block_tokens: int = HASH_BLOCK_TOKENS) -> int:
    """Number of hash blocks AIPerf requires: ceil(input_length / block_tokens).

    AIPerf's mooncake-trace loader validates that
    ``len(hash_ids) == ceil(input_length / block_size)`` and that the final block
    size ``input_length - (len - 1) * block_size`` is in ``(0, block_size]``.
    """
    return max(1, math.ceil(max(1, input_length) / block_tokens))


def compute_hash_ids(
    prompt: str,
    input_length: int | None = None,
    block_tokens: int = HASH_BLOCK_TOKENS,
) -> list[int]:
    """Hash-id list for the AIPerf mooncake trace.

    When ``input_length`` is given, the block COUNT is derived from it (not from a
    separate word/char estimate) so it always satisfies AIPerf's check. The prompt
    is split at fixed character boundaries so a shared prefix keeps shared hashes.
    """
    if input_length is None:
        return [hash_block(chunk) for chunk in _token_chunks(prompt, block_tokens)]

    n_blocks = hash_block_count(input_length, block_tokens)
    block_chars = tokens_to_chars(block_tokens)
    chunks = [prompt[i : i + block_chars] for i in range(0, len(prompt), block_chars)]
    return [
        hash_block(chunks[k]) if k < len(chunks) and chunks[k] else hash_block(f"__pad_{k}")
        for k in range(n_blocks)
    ]

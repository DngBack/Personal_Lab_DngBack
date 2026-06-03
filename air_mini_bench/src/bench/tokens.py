from __future__ import annotations

import zlib

CHARS_PER_TOKEN = 4.0
HASH_BLOCK_TOKENS = 512


def chars_to_tokens(chars: int) -> int:
    return max(1, int(round(chars / CHARS_PER_TOKEN)))


def tokens_to_chars(tokens: int) -> int:
    return max(1, int(round(tokens * CHARS_PER_TOKEN)))


def estimate_token_count(text: str) -> int:
    return chars_to_tokens(len(text))


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


def compute_hash_ids(prompt: str, block_tokens: int = HASH_BLOCK_TOKENS) -> list[int]:
    return [hash_block(chunk) for chunk in _token_chunks(prompt, block_tokens)]

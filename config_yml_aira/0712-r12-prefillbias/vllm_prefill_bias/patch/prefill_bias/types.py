"""Pure-data snapshots consumed by PrefillBiasController."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WaitingRequestSnapshot:
    request_id: str
    arrival_time_s: float
    prompt_tokens: int
    computed_tokens: int = 0
    cache_hit_tokens: int = 0
    has_encoder_inputs: bool = False
    is_blocked: bool = False

    @property
    def remaining_prefill_tokens(self) -> int:
        reusable = max(self.computed_tokens, self.cache_hit_tokens)
        return max(0, self.prompt_tokens - reusable)


@dataclass(frozen=True)
class DecodeSnapshot:
    request_id: str
    scheduled_decode_tokens: int = 1
    last_token_latency_ms: float = 0.0


@dataclass(frozen=True)
class CacheSnapshot:
    free_tokens: int
    token_budget: int
    prefix_cache_enabled: bool = True


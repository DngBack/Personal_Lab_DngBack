"""Pure scheduling policy for prefill bias.

The controller returns a small decision object. vLLM-specific code is expected
to enforce the decision inside Scheduler.schedule().
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import PrefillBiasConfig
from .types import CacheSnapshot, DecodeSnapshot, WaitingRequestSnapshot


@dataclass(frozen=True)
class PrefillBiasDecision:
    active: bool
    reason: str
    reserved_prefill_tokens: int = 0
    protected_decode_tokens: int = 0
    urgent_request_ids: tuple[str, ...] = field(default_factory=tuple)


class PrefillBiasController:
    def __init__(self, config: PrefillBiasConfig) -> None:
        self.config = config.normalized()

    def decide(
        self,
        *,
        now_s: float,
        waiting: list[WaitingRequestSnapshot],
        running_decodes: list[DecodeSnapshot],
        cache: CacheSnapshot,
    ) -> PrefillBiasDecision:
        if not self.config.enabled:
            return PrefillBiasDecision(False, "disabled")
        if not cache.prefix_cache_enabled:
            return PrefillBiasDecision(False, "prefix_cache_disabled")
        if cache.token_budget <= 0:
            return PrefillBiasDecision(False, "no_token_budget")
        if len(running_decodes) < self.config.min_decode_running:
            return PrefillBiasDecision(False, "not_enough_decodes")

        guarded_decodes = [
            decode
            for decode in running_decodes
            if decode.last_token_latency_ms >= self.config.tbt_guard_ms
        ]
        if guarded_decodes:
            return PrefillBiasDecision(False, "tbt_guard")

        scan = waiting[: self.config.max_waiting_scan]
        urgent = []
        for request in scan:
            if request.is_blocked or request.has_encoder_inputs:
                continue
            wait_ms = max(0.0, (now_s - request.arrival_time_s) * 1000.0)
            if wait_ms >= self.config.urgent_wait_ms:
                urgent.append(request.request_id)
            if len(urgent) >= self.config.max_urgent_prefills:
                break

        reserve = min(self.config.reserve_prefill_tokens, cache.token_budget)
        protected_decode = min(
            self.config.protected_decode_tokens * len(running_decodes),
            max(0, cache.token_budget - reserve),
        )

        if reserve <= 0 and not urgent:
            return PrefillBiasDecision(False, "nothing_to_bias")

        return PrefillBiasDecision(
            active=True,
            reason="active",
            reserved_prefill_tokens=reserve,
            protected_decode_tokens=protected_decode,
            urgent_request_ids=tuple(urgent),
        )


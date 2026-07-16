"""Lightweight metrics accumulator for prefill-bias decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .controller import PrefillBiasDecision


@dataclass
class PrefillBiasMetrics:
    decisions: int = 0
    active: int = 0
    fallback: int = 0
    urgent_prefills: int = 0
    reserved_prefill_tokens: int = 0
    protected_decode_tokens: int = 0
    tbt_guard_hits: int = 0

    def observe(self, decision: PrefillBiasDecision) -> None:
        self.decisions += 1
        if decision.active:
            self.active += 1
        else:
            self.fallback += 1
        if decision.reason == "tbt_guard":
            self.tbt_guard_hits += 1
        self.urgent_prefills += len(decision.urgent_request_ids)
        self.reserved_prefill_tokens += decision.reserved_prefill_tokens
        self.protected_decode_tokens += decision.protected_decode_tokens

    def as_log_fields(self) -> dict[str, int]:
        return {
            "decisions": self.decisions,
            "active": self.active,
            "fallback": self.fallback,
            "urgent_prefills": self.urgent_prefills,
            "reserved_prefill_tokens": self.reserved_prefill_tokens,
            "protected_decode_tokens": self.protected_decode_tokens,
            "tbt_guard_hits": self.tbt_guard_hits,
        }


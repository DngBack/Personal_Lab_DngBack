"""Config model for the prefill-bias scheduler patch.

These fields are intentionally independent from vLLM dataclasses for now.
The patch layer should translate SchedulerConfig/CLI fields into this shape.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillBiasConfig:
    enabled: bool = False
    reserve_prefill_tokens: int = 0
    urgent_wait_ms: int = 1200
    max_urgent_prefills: int = 1
    protected_decode_tokens: int = 1
    min_decode_running: int = 1
    max_waiting_scan: int = 32
    tbt_guard_ms: int = 250
    metrics_log_interval_s: float = 10.0

    def normalized(self) -> "PrefillBiasConfig":
        return PrefillBiasConfig(
            enabled=self.enabled,
            reserve_prefill_tokens=max(0, self.reserve_prefill_tokens),
            urgent_wait_ms=max(0, self.urgent_wait_ms),
            max_urgent_prefills=max(0, self.max_urgent_prefills),
            protected_decode_tokens=max(0, self.protected_decode_tokens),
            min_decode_running=max(0, self.min_decode_running),
            max_waiting_scan=max(1, self.max_waiting_scan),
            tbt_guard_ms=max(0, self.tbt_guard_ms),
            metrics_log_interval_s=max(0.0, self.metrics_log_interval_s),
        )


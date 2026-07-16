"""vLLM scheduler integration notes and adapters.

Keep this file thin. The concrete patch should:

1. Build PrefillBiasConfig from SchedulerConfig.
2. Convert vLLM Request objects into WaitingRequestSnapshot.
3. Convert running decode Request objects into DecodeSnapshot.
4. Ask PrefillBiasController.decide().
5. Apply budget reservation and urgent admission in Scheduler.schedule().
"""

from __future__ import annotations

from .types import CacheSnapshot


def build_cache_snapshot(*, token_budget: int, free_tokens: int) -> CacheSnapshot:
    return CacheSnapshot(
        free_tokens=max(0, free_tokens),
        token_budget=max(0, token_budget),
        prefix_cache_enabled=True,
    )


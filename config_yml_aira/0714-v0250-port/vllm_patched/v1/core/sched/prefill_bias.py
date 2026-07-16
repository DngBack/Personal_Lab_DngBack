# SPDX-License-Identifier: Apache-2.0
"""Experimental adaptive prefill-bias policy for vLLM V1 scheduling.

This module is intentionally pure policy code. It does not depend on
KVCacheManager, SchedulerOutput, or scheduler mutation details.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


_DEADLINE_EPSILON_S = 1e-9


@dataclass(frozen=True)
class PrefillCandidate:
    request_id: str
    waiting_age_s: float
    arrival_time: float
    total_request_tokens: int
    cached_tokens: int
    remaining_prefill_tokens: int
    remaining_bucket: int
    starved: bool
    cache_peek_supported: bool = True
    predicted_completion_s: float = 0.0
    ttft_slack_s: float = math.inf
    urgent: bool = False
    forced_ttft: bool = False
    salvageable: bool = True


@dataclass(frozen=True)
class PrefillBiasDecision:
    active: bool
    reserve_tokens: int
    candidate_request_ids: tuple[str, ...]
    reason: str
    scored_requests: int = 0
    selected_cached_tokens: int = 0
    selected_remaining_tokens: int = 0
    minimum_ttft_slack_s: float = math.inf
    predicted_completion_s: float = 0.0
    forced_ttft: bool = False
    candidate_token_caps: tuple[tuple[str, int], ...] = ()
    predicted_step_time_s: float = 0.0
    policy_mode: str = "legacy"
    slot_swap_eligible: bool = True


@dataclass(frozen=True)
class PrefillBatchWorkItem:
    request_id: str
    source: str
    arrival_time: float
    waiting_age_s: float
    remaining_prefill_tokens: int
    cached_tokens: int
    predicted_completion_s: float
    ttft_slack_s: float
    urgent: bool
    salvageable: bool
    hard_starved: bool


@dataclass(frozen=True)
class PrefillBatchBudgetDecision:
    total_prefill_budget: int
    request_token_caps: tuple[tuple[str, int], ...]
    running_prefill_order: tuple[str, ...]
    waiting_prefill_order: tuple[str, ...]
    predicted_step_time_s: float
    minimum_tbt_slack_s: float
    reason: str
    fail_closed: bool = False
    slot_swap_request_ids: tuple[str, ...] = ()


class PrefillBudgetAccountant:
    """Tracks one step's global and per-request prefill token budget."""

    def __init__(self, decision: PrefillBatchBudgetDecision) -> None:
        self.total_budget = max(0, int(decision.total_prefill_budget))
        self.remaining_budget = self.total_budget
        self._caps = {
            str(request_id): max(0, int(cap))
            for request_id, cap in decision.request_token_caps
        }
        self._committed: dict[str, int] = {}
        self.running_prefill_tokens = 0
        self.waiting_prefill_tokens = 0

    @property
    def committed_tokens(self) -> int:
        return sum(self._committed.values())

    @property
    def committed_by_request(self) -> tuple[tuple[str, int], ...]:
        return tuple(self._committed.items())

    def has_cap(self, request_id: str) -> bool:
        return request_id in self._caps

    def clamp(self, request_id: str, requested_tokens: int) -> int:
        request_id = str(request_id)
        requested = max(0, int(requested_tokens))
        cap = self._caps.get(request_id)
        if cap is None or requested <= 0 or self.remaining_budget <= 0:
            return 0
        used = self._committed.get(request_id, 0)
        return min(requested, max(0, cap - used), self.remaining_budget)

    def commit(self, request_id: str, token_count: int, *, source: str) -> None:
        request_id = str(request_id)
        tokens = max(0, int(token_count))
        if tokens <= 0:
            return
        allowed = self.clamp(request_id, tokens)
        if allowed != tokens:
            raise ValueError("prefill budget commit exceeds request or global cap")
        self._committed[request_id] = self._committed.get(request_id, 0) + tokens
        self.remaining_budget -= tokens
        if source == "running":
            self.running_prefill_tokens += tokens
        elif source == "waiting":
            self.waiting_prefill_tokens += tokens
        else:
            raise ValueError("prefill budget source must be running or waiting")

    def restore(self, request_id: str, token_count: int, *, source: str) -> int:
        request_id = str(request_id)
        requested = max(0, int(token_count))
        committed = self._committed.get(request_id, 0)
        restored = min(requested, committed)
        if restored <= 0:
            return 0
        remaining = committed - restored
        if remaining:
            self._committed[request_id] = remaining
        else:
            self._committed.pop(request_id, None)
        self.remaining_budget += restored
        if source == "running":
            self.running_prefill_tokens -= restored
        elif source == "waiting":
            self.waiting_prefill_tokens -= restored
        else:
            raise ValueError("prefill budget source must be running or waiting")
        self.assert_invariants()
        return restored

    def assert_invariants(self) -> None:
        if self.remaining_budget < 0 or self.committed_tokens > self.total_budget:
            raise AssertionError("global prefill budget exceeded")
        if self.remaining_budget + self.committed_tokens != self.total_budget:
            raise AssertionError("prefill budget accounting mismatch")
        for request_id, committed in self._committed.items():
            if committed > self._caps.get(request_id, 0):
                raise AssertionError("per-request prefill cap exceeded")
        if self.running_prefill_tokens + self.waiting_prefill_tokens != (
            self.committed_tokens
        ):
            raise AssertionError("prefill source accounting mismatch")


@dataclass(frozen=True)
class TBTGuardSnapshot:
    active_decode_count: int
    known_decode_count: int
    unknown_decode_count: int
    oldest_output_gap_s: float
    minimum_tbt_slack_s: float
    predicted_step_time_s: float
    safety_margin_s: float
    allowed: bool
    reason: str


@dataclass(frozen=True)
class DecodeTimingState:
    request_identity: int
    last_accepted_output_monotonic_s: float
    accepted_output_events: int = 1


class SlotSwapBlocker(str, Enum):
    NO_BLOCKER = "no_blocker"
    MAX_NUM_SEQS = "max_num_seqs"
    TOKEN_BUDGET = "token_budget"
    KV_CAPACITY = "kv_capacity"
    ENCODER_BUDGET = "encoder_budget"
    GRAMMAR_NOT_READY = "grammar_not_ready"
    REMOTE_KV = "remote_kv"
    ASYNC_IN_FLIGHT = "async_in_flight"
    COOLDOWN = "cooldown"
    TBT_GUARD = "tbt_guard"
    UNSUPPORTED_STATE = "unsupported_state"


class SlotSwapRejectReason(str, Enum):
    FEATURE_DISABLED = "feature_disabled"
    NO_CANDIDATE = "no_candidate"
    CANDIDATE_NOT_URGENT = "candidate_not_urgent"
    BLOCKER_NOT_MAX_NUM_SEQS = "blocker_not_max_num_seqs"
    TOKEN_BUDGET = "token_budget"
    KV_PREFLIGHT_FAILED = "kv_preflight_failed"
    TBT_GUARD = "tbt_guard"
    NO_SAFE_VICTIM = "no_safe_victim"
    VICTIM_ASYNC_IN_FLIGHT = "victim_async_in_flight"
    VICTIM_COOLDOWN = "victim_cooldown"
    VICTIM_PREEMPTION_LIMIT = "victim_preemption_limit"
    VICTIM_RECOMPUTE_TOO_LARGE = "victim_recompute_too_large"
    COMMIT_FAILED = "commit_failed"


class AdaptivePrefillState(str, Enum):
    COLD_START = "cold_start"
    BALANCED = "balanced"
    PREFILL_RECOVERY = "prefill_recovery"
    DECODE_PROTECT = "decode_protect"
    OVERLOAD = "overload"
    FAIL_SAFE = "fail_safe"


@dataclass(frozen=True)
class PrefillAdmissionSwapResult:
    performed: bool
    candidate_request_id: str | None
    victim_request_id: str | None
    blocker_reason: SlotSwapBlocker
    reject_reason: SlotSwapRejectReason | None
    predicted_ttft_slack_s: float | None
    predicted_victim_tbt_s: float | None
    candidate_remaining_tokens: int | None
    victim_recompute_tokens: int | None


@dataclass(frozen=True)
class AdaptivePrefillPolicy:
    level: int
    state: AdaptivePrefillState
    prefill_bias_reserve_tokens: int
    prefill_bias_wait_threshold_s: float
    prefill_bias_max_requests_per_step: int
    prefill_bias_score_window_k: int
    prefill_bias_tbt_safety_margin_s: float
    prefill_bias_slot_swap_enabled: bool
    prefill_bias_max_swaps_per_step: int
    prefill_bias_max_candidate_remaining_tokens: int
    reason: str = "baseline"

    def as_config(self, base_config: Any) -> "AdaptivePrefillConfigView":
        return AdaptivePrefillConfigView(base_config, self)


class AdaptivePrefillConfigView:
    """Read-only scheduler-config overlay used for one scheduler step."""

    def __init__(self, base_config: Any, policy: AdaptivePrefillPolicy) -> None:
        self._base_config = base_config
        self._policy = policy

    def __getattr__(self, name: str) -> Any:
        if hasattr(self._policy, name):
            return getattr(self._policy, name)
        return getattr(self._base_config, name)


@dataclass
class _AdaptiveRequestState:
    arrival_time: float
    first_token_wall_s: float | None = None
    last_token_monotonic_s: float | None = None
    ttft_ok: bool = False
    tbt_ok: bool = True
    has_tbt_sample: bool = False


@dataclass(frozen=True)
class AdaptivePrefillSignals:
    samples: int
    ttft_violation_ratio: float
    tbt_violation_ratio: float
    joint_attainment_ratio: float
    goodput_rps: float
    oldest_waiting_prefill_age_s: float
    waiting_prefill_count: int
    active_decode_count: int
    kv_cache_usage: float
    swap_failure_rate: float
    recompute_tokens: int


def bucket_remaining_tokens(remaining_tokens: int, edges: tuple[int, ...]) -> int:
    """Return the first bucket whose upper edge is greater than remaining.

    Edges are lower bounds for the next bucket. With edges ``(16, 64)``,
    remaining tokens 1..15 map to bucket 0, 16..63 to bucket 1, and 64+
    to bucket 2.
    """

    remaining = max(1, int(remaining_tokens))
    for bucket, edge in enumerate(edges):
        if remaining < edge:
            return bucket
    return len(edges)


def compute_remaining_prefill_tokens(
    *,
    total_prefill_tokens: int,
    request_computed_tokens: int,
    cached_prefix_tokens: int,
    minimum_required_tokens: int = 1,
) -> int:
    """Estimate decoder prefill work left using vLLM scheduler semantics.

    The value is only a scoring hint. Normal waiting admission still performs
    the authoritative cache lookup and token-budget accounting.
    """

    total = max(1, int(total_prefill_tokens))
    computed = max(0, int(request_computed_tokens))
    cached = max(0, int(cached_prefix_tokens))
    minimum = max(0, int(minimum_required_tokens))
    effective_computed = min(total, max(computed, cached))
    return max(total - effective_computed, minimum)


class PrefillStepTimeEstimator:
    def __init__(self, *, ewma_alpha: float = 0.2) -> None:
        alpha = float(ewma_alpha)
        if not math.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
            raise ValueError("ewma_alpha must satisfy 0.0 < value <= 1.0")
        self.ewma_alpha = alpha
        self.ewma_step_time_s: float | None = None
        self.num_samples = 0
        self.max_observed_step_time_s = 0.0
        self.last_observed_step_time_s = 0.0

    def observe(self, duration_s: float) -> None:
        duration = float(duration_s)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be a positive finite value")
        if self.ewma_step_time_s is None:
            self.ewma_step_time_s = duration
        else:
            self.ewma_step_time_s = (
                self.ewma_alpha * duration
                + (1.0 - self.ewma_alpha) * self.ewma_step_time_s
            )
        self.num_samples += 1
        self.last_observed_step_time_s = duration
        self.max_observed_step_time_s = max(self.max_observed_step_time_s, duration)

    def estimate(
        self,
        *,
        initial_step_time_s: float,
        headroom_factor: float,
        min_samples: int,
    ) -> float:
        initial = float(initial_step_time_s)
        headroom = float(headroom_factor)
        samples_required = max(1, int(min_samples))
        if not math.isfinite(initial) or initial <= 0.0:
            raise ValueError("initial_step_time_s must be a positive finite value")
        if not math.isfinite(headroom) or headroom < 1.0:
            raise ValueError("headroom_factor must be finite and >= 1.0")
        if self.ewma_step_time_s is None or self.num_samples < samples_required:
            return initial
        return max(initial, self.ewma_step_time_s) * headroom


@dataclass
class _PredictiveEwmaCell:
    value: float = 0.0
    samples: int = 0


class PredictivePrefillEstimator:
    """Small bounded online model for mixed prefill/decode step latency."""

    _PREFILL_BUCKETS = (128, 256, 512, 1024, 2048, 3072)
    _DECODE_BUCKETS = (0, 8, 32, 128, 512)

    def __init__(self, *, ewma_alpha: float = 0.2) -> None:
        alpha = float(ewma_alpha)
        if not math.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
            raise ValueError("ewma_alpha must satisfy 0.0 < value <= 1.0")
        self.ewma_alpha = alpha
        self._batch_cells: dict[tuple[int, int], _PredictiveEwmaCell] = {}
        self._decode_cells: dict[int, _PredictiveEwmaCell] = {}
        self._prefill_s_per_token = _PredictiveEwmaCell()
        self.observations = 0
        self.last_prediction_s = 0.0
        self.last_actual_s = 0.0

    @classmethod
    def prefill_bucket(cls, tokens: int) -> int:
        tokens = max(0, int(tokens))
        if tokens <= 0:
            return 0
        for edge in cls._PREFILL_BUCKETS:
            if tokens <= edge:
                return edge
        return cls._PREFILL_BUCKETS[-1]

    @classmethod
    def decode_bucket(cls, active_decode_count: int) -> int:
        count = max(0, int(active_decode_count))
        if count <= 0:
            return 0
        for edge in cls._DECODE_BUCKETS[1:]:
            if count <= edge:
                return edge
        return cls._DECODE_BUCKETS[-1]

    def _update(self, cell: _PredictiveEwmaCell, value: float) -> None:
        if cell.samples == 0:
            cell.value = value
        else:
            cell.value = (
                self.ewma_alpha * value
                + (1.0 - self.ewma_alpha) * cell.value
            )
        cell.samples += 1

    def observe(
        self,
        *,
        duration_s: float,
        prefill_tokens: int,
        active_decode_count: int,
    ) -> None:
        duration = float(duration_s)
        if not math.isfinite(duration) or duration <= 0.0:
            return
        prefill_tokens = max(0, int(prefill_tokens))
        decode_bucket = self.decode_bucket(active_decode_count)
        prefill_bucket = self.prefill_bucket(prefill_tokens)
        self._update(
            self._batch_cells.setdefault(
                (prefill_bucket, decode_bucket),
                _PredictiveEwmaCell(),
            ),
            duration,
        )
        if prefill_tokens <= 0:
            self._update(
                self._decode_cells.setdefault(decode_bucket, _PredictiveEwmaCell()),
                duration,
            )
        else:
            decode_cell = self._decode_cells.get(decode_bucket)
            decode_base = decode_cell.value if decode_cell and decode_cell.samples else 0.0
            incremental_s = max(0.0, duration - decode_base)
            if decode_base <= 0.0:
                incremental_s = duration
            self._update(
                self._prefill_s_per_token,
                incremental_s / prefill_tokens,
            )
        self.observations += 1
        self.last_actual_s = duration

    def predict(
        self,
        *,
        prefill_tokens: int,
        active_decode_count: int,
        initial_step_time_s: float,
        legacy_reserve_tokens: int,
        headroom_factor: float,
        min_samples: int,
    ) -> float:
        tokens = max(0, int(prefill_tokens))
        initial = max(1e-9, float(initial_step_time_s))
        headroom = max(1.0, float(headroom_factor))
        legacy_reserve = max(1, int(legacy_reserve_tokens))
        required = max(1, int(min_samples))
        decode_bucket = self.decode_bucket(active_decode_count)
        prefill_bucket = self.prefill_bucket(tokens)

        cold_start = initial * max(1.0, tokens / legacy_reserve)
        decode_cell = self._decode_cells.get(decode_bucket)
        decode_base = (
            decode_cell.value
            if decode_cell is not None and decode_cell.samples >= required
            else 0.0
        )
        linear = decode_base
        if tokens > 0 and self._prefill_s_per_token.samples >= required:
            linear += self._prefill_s_per_token.value * tokens

        bucket_value = 0.0
        batch_cell = self._batch_cells.get((prefill_bucket, decode_bucket))
        if batch_cell is not None and batch_cell.samples >= required:
            bucket_value = batch_cell.value

        predicted = max(cold_start, linear, bucket_value) * headroom
        self.last_prediction_s = predicted
        return predicted


class PrefillBatchBudgetController:
    """Plans one bounded prefill batch across running and waiting work."""

    def __init__(
        self,
        scheduler_config: Any,
        *,
        estimator: PredictivePrefillEstimator,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.estimator = estimator
        self._time_fn = time_fn
        self.decisions_total = 0
        self.fail_closed_total = 0
        self.tbt_rejections_total = 0
        self.unknown_timing_total = 0
        self.predictor_errors_total = 0
        self.last_decision: PrefillBatchBudgetDecision | None = None

    @staticmethod
    def fail_closed(reason: str) -> PrefillBatchBudgetDecision:
        return PrefillBatchBudgetDecision(
            total_prefill_budget=0,
            request_token_caps=(),
            running_prefill_order=(),
            waiting_prefill_order=(),
            predicted_step_time_s=0.0,
            minimum_tbt_slack_s=math.inf,
            reason=reason,
            fail_closed=True,
        )

    def plan(
        self,
        *,
        running_prefills: Iterable[Any],
        waiting_prefills: Iterable[Any],
        max_safe_prefill_tokens: int,
        active_decode_count: int,
        minimum_tbt_slack_s: float,
        unknown_decode_count: int,
        peek_cached_tokens: Callable[[Any], int],
        is_request_schedulable: Callable[[Any], bool] | None = None,
    ) -> PrefillBatchBudgetDecision:
        self.decisions_total += 1
        active_decode_count = max(0, int(active_decode_count))
        if active_decode_count > 0 and int(unknown_decode_count) > 0:
            self.unknown_timing_total += 1
            self.fail_closed_total += 1
            decision = self.fail_closed("batch_budget_unknown_decode_timing")
            self.last_decision = decision
            return decision

        max_budget = min(
            max(0, int(max_safe_prefill_tokens)),
            max(
                0,
                int(
                    getattr(
                        self.scheduler_config,
                        "prefill_bias_batch_max_prefill_tokens",
                        3072,
                    )
                ),
            ),
        )
        if max_budget <= 0:
            decision = PrefillBatchBudgetDecision(
                total_prefill_budget=0,
                request_token_caps=(),
                running_prefill_order=(),
                waiting_prefill_order=(),
                predicted_step_time_s=0.0,
                minimum_tbt_slack_s=minimum_tbt_slack_s,
                reason="batch_budget_no_token_budget",
            )
            self.last_decision = decision
            return decision

        try:
            items = self._build_work_items(
                running_prefills=running_prefills,
                waiting_prefills=waiting_prefills,
                per_step_cap=max_budget,
                active_decode_count=active_decode_count,
                peek_cached_tokens=peek_cached_tokens,
                is_request_schedulable=is_request_schedulable,
            )
            if not items:
                decision = PrefillBatchBudgetDecision(
                    total_prefill_budget=0,
                    request_token_caps=(),
                    running_prefill_order=(),
                    waiting_prefill_order=(),
                    predicted_step_time_s=0.0,
                    minimum_tbt_slack_s=minimum_tbt_slack_s,
                    reason="batch_budget_no_prefill_work",
                )
                self.last_decision = decision
                return decision

            upper_bound = min(
                max_budget,
                sum(item.remaining_prefill_tokens for item in items),
            )
            selected_budget = self._select_budget(
                upper_bound=upper_bound,
                active_decode_count=active_decode_count,
                minimum_tbt_slack_s=minimum_tbt_slack_s,
            )
            if selected_budget <= 0:
                self.tbt_rejections_total += 1
                self.fail_closed_total += 1
                decision = self.fail_closed("batch_budget_no_tbt_safe_budget")
                self.last_decision = decision
                return decision

            ordered = sorted(items, key=self._completion_first_key)
            max_requests = max(
                1,
                int(
                    getattr(
                        self.scheduler_config,
                        "prefill_bias_batch_max_requests_per_step",
                        4,
                    )
                ),
            )
            caps: list[tuple[str, int]] = []
            selected_items: list[PrefillBatchWorkItem] = []
            remaining_budget = selected_budget
            for item in ordered:
                if remaining_budget <= 0 or len(caps) >= max_requests:
                    break
                cap = min(item.remaining_prefill_tokens, remaining_budget)
                if cap <= 0:
                    continue
                caps.append((item.request_id, cap))
                selected_items.append(item)
                remaining_budget -= cap

            actual_budget = sum(cap for _, cap in caps)
            predicted_step = self._predict_step(
                prefill_tokens=actual_budget,
                active_decode_count=active_decode_count,
            )
            decision = PrefillBatchBudgetDecision(
                total_prefill_budget=actual_budget,
                request_token_caps=tuple(caps),
                running_prefill_order=tuple(
                    item.request_id
                    for item in selected_items
                    if item.source == "running"
                ),
                waiting_prefill_order=tuple(
                    item.request_id
                    for item in selected_items
                    if item.source == "waiting"
                ),
                predicted_step_time_s=predicted_step,
                minimum_tbt_slack_s=minimum_tbt_slack_s,
                reason="batch_budget_active",
                slot_swap_request_ids=tuple(
                    item.request_id
                    for item in selected_items
                    if item.source == "waiting"
                    and (item.hard_starved or item.ttft_slack_s <= 0.0)
                ),
            )
            self.last_decision = decision
            return decision
        except Exception:
            self.predictor_errors_total += 1
            self.fail_closed_total += 1
            decision = self.fail_closed("batch_budget_predictor_error")
            self.last_decision = decision
            return decision

    def _build_work_items(
        self,
        *,
        running_prefills: Iterable[Any],
        waiting_prefills: Iterable[Any],
        per_step_cap: int,
        active_decode_count: int,
        peek_cached_tokens: Callable[[Any], int],
        is_request_schedulable: Callable[[Any], bool] | None,
    ) -> list[PrefillBatchWorkItem]:
        now = self._time_fn()
        ttft_slo_s = max(
            1e-9,
            float(getattr(self.scheduler_config, "prefill_bias_ttft_slo_s", 0.0)),
        )
        trigger_margin_s = max(
            0.0,
            float(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_swap_slack_threshold_s",
                    0.0,
                )
            ),
        )
        hard_starvation_s = max(
            float(getattr(self.scheduler_config, "prefill_bias_starvation_s", 0.2)),
            4.0 * ttft_slo_s,
        )
        running_limit = max(
            1,
            int(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_batch_running_scan_limit",
                    32,
                )
            ),
        )
        waiting_limit = max(
            1,
            int(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_candidate_scan_limit",
                    32,
                )
            ),
        )
        min_cached_tokens = max(
            0,
            int(getattr(self.scheduler_config, "prefill_bias_min_cached_tokens", 0)),
        )

        raw_items: list[tuple[str, str, float, int, int]] = []
        running_seen = 0
        for request in running_prefills:
            if running_seen >= running_limit:
                break
            if not bool(getattr(request, "is_prefill_chunk", False)):
                continue
            running_seen += 1
            request_id = str(getattr(request, "request_id", ""))
            computed = max(0, int(getattr(request, "num_computed_tokens", 0) or 0))
            total = getattr(request, "num_tokens_with_spec", None)
            if total is None:
                total = getattr(request, "num_tokens", 0)
            total = max(1, int(total or 0)) + max(
                0,
                int(getattr(request, "num_output_placeholders", 0) or 0),
            )
            remaining = max(0, total - computed)
            if remaining <= 0:
                continue
            raw_items.append(
                (
                    request_id,
                    "running",
                    float(getattr(request, "arrival_time", 0.0) or 0.0),
                    remaining,
                    0,
                )
            )

        waiting_seen = 0
        for request in waiting_prefills:
            if waiting_seen >= waiting_limit:
                break
            if is_request_schedulable is not None and not is_request_schedulable(request):
                continue
            status = getattr(request, "status", None)
            if str(getattr(status, "name", status)) != "WAITING":
                continue
            if bool(getattr(request, "is_prefill_chunk", False)) or bool(
                getattr(request, "abort_immediately", False)
            ):
                continue
            waiting_seen += 1
            cached_tokens = max(0, int(peek_cached_tokens(request)))
            effective_cached = cached_tokens if cached_tokens >= min_cached_tokens else 0
            total_value = getattr(request, "num_prompt_tokens", None)
            if total_value is None:
                total_value = getattr(request, "num_tokens", 0)
            total = max(1, int(total_value or 0))
            computed = max(0, int(getattr(request, "num_computed_tokens", 0) or 0))
            remaining = compute_remaining_prefill_tokens(
                total_prefill_tokens=total,
                request_computed_tokens=computed,
                cached_prefix_tokens=effective_cached,
                minimum_required_tokens=1,
            )
            raw_items.append(
                (
                    str(getattr(request, "request_id", "")),
                    "waiting",
                    float(getattr(request, "arrival_time", 0.0) or 0.0),
                    remaining,
                    cached_tokens,
                )
            )

        items: list[PrefillBatchWorkItem] = []
        for request_id, source, arrival_time, remaining, cached_tokens in raw_items:
            age_s = max(0.0, now - arrival_time)
            completion_s = self._predict_completion(
                remaining_tokens=remaining,
                per_step_cap=per_step_cap,
                active_decode_count=active_decode_count,
            )
            slack_s = ttft_slo_s - age_s - completion_s
            hard_starved = age_s >= hard_starvation_s
            salvageable = slack_s >= 0.0
            urgent = hard_starved or slack_s <= trigger_margin_s or math.isclose(
                slack_s,
                trigger_margin_s,
                rel_tol=0.0,
                abs_tol=_DEADLINE_EPSILON_S,
            )
            items.append(
                PrefillBatchWorkItem(
                    request_id=request_id,
                    source=source,
                    arrival_time=arrival_time,
                    waiting_age_s=age_s,
                    remaining_prefill_tokens=remaining,
                    cached_tokens=cached_tokens,
                    predicted_completion_s=completion_s,
                    ttft_slack_s=slack_s,
                    urgent=urgent,
                    salvageable=salvageable,
                    hard_starved=hard_starved,
                )
            )
        return items

    def _select_budget(
        self,
        *,
        upper_bound: int,
        active_decode_count: int,
        minimum_tbt_slack_s: float,
    ) -> int:
        upper = max(0, int(upper_bound))
        if upper <= 0:
            return 0
        min_tokens = max(
            1,
            int(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_batch_min_prefill_tokens",
                    128,
                )
            ),
        )
        budgets = {
            budget
            for budget in PredictivePrefillEstimator._PREFILL_BUCKETS
            if min_tokens <= budget <= upper
        }
        if upper < min_tokens:
            budgets.add(upper)
        else:
            budgets.add(upper)
        if active_decode_count <= 0:
            return max(budgets)
        if not math.isfinite(minimum_tbt_slack_s):
            return 0

        tbt_slo_s = max(
            1e-9,
            float(getattr(self.scheduler_config, "prefill_bias_tbt_slo_s", 0.0)),
        )
        margin_s = max(
            0.0,
            float(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_tbt_safety_margin_s",
                    0.0,
                )
            ),
        )
        decode_next_s = self._predict_step(
            prefill_tokens=0,
            active_decode_count=active_decode_count,
        )
        best = 0
        for budget in sorted(budgets):
            current_s = self._predict_step(
                prefill_tokens=budget,
                active_decode_count=active_decode_count,
            )
            if (
                current_s <= minimum_tbt_slack_s - margin_s
                and current_s + decode_next_s
                <= minimum_tbt_slack_s + tbt_slo_s - margin_s
            ):
                best = budget
        return best

    def _predict_completion(
        self,
        *,
        remaining_tokens: int,
        per_step_cap: int,
        active_decode_count: int,
    ) -> float:
        remaining = max(1, int(remaining_tokens))
        cap = max(1, int(per_step_cap))
        full_steps, tail = divmod(remaining, cap)
        total = full_steps * self._predict_step(
            prefill_tokens=cap,
            active_decode_count=active_decode_count,
        )
        if tail:
            total += self._predict_step(
                prefill_tokens=tail,
                active_decode_count=active_decode_count,
            )
        return total

    def _predict_step(self, *, prefill_tokens: int, active_decode_count: int) -> float:
        return self.estimator.predict(
            prefill_tokens=prefill_tokens,
            active_decode_count=active_decode_count,
            initial_step_time_s=float(
                getattr(self.scheduler_config, "prefill_bias_initial_step_time_s", 0.01)
            ),
            legacy_reserve_tokens=max(
                1,
                int(getattr(self.scheduler_config, "prefill_bias_reserve_tokens", 64)),
            ),
            headroom_factor=float(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_step_time_headroom_factor",
                    1.25,
                )
            ),
            min_samples=int(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_step_observation_min_samples",
                    3,
                )
            ),
        )

    @staticmethod
    def _completion_first_key(item: PrefillBatchWorkItem) -> tuple[Any, ...]:
        if item.hard_starved:
            tier = 0
        elif item.urgent and item.salvageable:
            tier = 1
        elif item.urgent:
            tier = 2
        else:
            tier = 3
        return (
            tier,
            item.ttft_slack_s if tier == 1 else 0.0,
            item.remaining_prefill_tokens,
            0 if item.source == "running" else 1,
            item.arrival_time,
            item.request_id,
        )


class AdaptivePrefillController:
    """Bounded SLO/goodput controller for Phase 4.

    The controller is deliberately conservative. It only emits immutable policy
    snapshots inside configured bounds and never treats missing observations as
    successful SLO attainment.
    """

    def __init__(
        self,
        scheduler_config: Any,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.scheduler_config = scheduler_config
        self._monotonic_clock = monotonic_clock
        self.enabled = bool(
            getattr(scheduler_config, "adaptive_prefill_controller_enabled", False)
        )
        self.state = AdaptivePrefillState.COLD_START
        self.level = 0
        self.last_update_monotonic_s = 0.0
        self._state_epochs = 0
        self._cooldown_until_s = 0.0
        self._requests: dict[str, _AdaptiveRequestState] = {}
        self._completed: deque[tuple[float, bool, bool, bool]] = deque()
        self._pressure: AdaptivePrefillSignals | None = None
        self._updates_total = 0
        self._fail_safe_total = 0
        self._overload_epochs_total = 0
        self._last_reason = "cold_start"
        self._policy = self._baseline_policy(AdaptivePrefillState.COLD_START)

    @property
    def updates_total(self) -> int:
        return self._updates_total

    @property
    def fail_safe_total(self) -> int:
        return self._fail_safe_total

    @property
    def overload_epochs_total(self) -> int:
        return self._overload_epochs_total

    @property
    def last_reason(self) -> str:
        return self._last_reason

    @property
    def observation_samples(self) -> int:
        return len(self._completed)

    def current_policy(self) -> AdaptivePrefillPolicy:
        return self._policy

    def reset_request(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def reset(self) -> None:
        self.state = AdaptivePrefillState.COLD_START
        self.level = 0
        self.last_update_monotonic_s = 0.0
        self._state_epochs = 0
        self._cooldown_until_s = 0.0
        self._requests.clear()
        self._completed.clear()
        self._pressure = None
        self._last_reason = "reset"
        self._policy = self._baseline_policy(AdaptivePrefillState.COLD_START)

    def observe_accepted_tokens(
        self,
        *,
        request_id: str,
        arrival_time: float,
        num_new_tokens: int,
        now_wall: float,
        now_monotonic: float,
        ttft_slo_s: float,
        tbt_slo_s: float,
    ) -> None:
        if not self.enabled or num_new_tokens <= 0:
            return
        if not self._valid_positive(ttft_slo_s) or not self._valid_positive(tbt_slo_s):
            self._enter_fail_safe("invalid_slo")
            return
        if (
            not math.isfinite(arrival_time)
            or not math.isfinite(now_wall)
            or not math.isfinite(now_monotonic)
        ):
            self._enter_fail_safe("invalid_timestamp")
            return
        state = self._requests.setdefault(
            request_id,
            _AdaptiveRequestState(arrival_time=arrival_time),
        )
        if state.first_token_wall_s is None:
            state.first_token_wall_s = now_wall
            state.ttft_ok = (now_wall - state.arrival_time) <= ttft_slo_s
        elif state.last_token_monotonic_s is not None:
            interval_s = now_monotonic - state.last_token_monotonic_s
            if not math.isfinite(interval_s) or interval_s < 0.0:
                self._enter_fail_safe("invalid_tbt_interval")
                return
            state.has_tbt_sample = True
            state.tbt_ok = state.tbt_ok and interval_s <= tbt_slo_s
        state.last_token_monotonic_s = now_monotonic

    def observe_request_finished(
        self,
        *,
        request_id: str,
        finished_ok: bool,
        now_monotonic: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        finished_at = (
            self._monotonic_clock() if now_monotonic is None else now_monotonic
        )
        if not math.isfinite(finished_at):
            self._enter_fail_safe("invalid_finish_timestamp")
            return
        state = self._requests.pop(request_id, None)
        if state is None or state.first_token_wall_s is None:
            ttft_ok = False
            tbt_ok = False
        else:
            ttft_ok = state.ttft_ok
            tbt_ok = state.tbt_ok
        joint_ok = bool(finished_ok and ttft_ok and tbt_ok)
        self._completed.append((finished_at, bool(ttft_ok), bool(tbt_ok), joint_ok))
        self._trim_window(finished_at)

    def observe_pressure(
        self,
        *,
        now_monotonic: float,
        waiting_prefill_count: int,
        oldest_waiting_prefill_age_s: float,
        active_decode_count: int,
        running_count: int,
        max_running_count: int,
        token_budget: int,
        max_token_budget: int,
        kv_cache_usage: float,
        swap_attempts: int = 0,
        swap_failures: int = 0,
        recompute_tokens: int = 0,
    ) -> None:
        if not self.enabled:
            return
        if (
            not math.isfinite(now_monotonic)
            or not math.isfinite(oldest_waiting_prefill_age_s)
            or not math.isfinite(kv_cache_usage)
        ):
            self._enter_fail_safe("invalid_pressure")
            return
        swap_failure_rate = (
            min(1.0, max(0.0, swap_failures / swap_attempts))
            if swap_attempts > 0
            else 0.0
        )
        samples, ttft_bad, tbt_bad, joint, goodput = self._window_rates(now_monotonic)
        self._pressure = AdaptivePrefillSignals(
            samples=samples,
            ttft_violation_ratio=ttft_bad,
            tbt_violation_ratio=tbt_bad,
            joint_attainment_ratio=joint,
            goodput_rps=goodput,
            oldest_waiting_prefill_age_s=max(0.0, oldest_waiting_prefill_age_s),
            waiting_prefill_count=max(0, int(waiting_prefill_count)),
            active_decode_count=max(0, int(active_decode_count)),
            kv_cache_usage=min(1.0, max(0.0, kv_cache_usage)),
            swap_failure_rate=swap_failure_rate,
            recompute_tokens=max(0, int(recompute_tokens)),
        )

    def maybe_update(self, now_monotonic: float | None = None) -> AdaptivePrefillPolicy:
        if not self.enabled:
            return self._policy
        now = self._monotonic_clock() if now_monotonic is None else now_monotonic
        if not math.isfinite(now):
            self._enter_fail_safe("invalid_update_time")
            return self._policy
        interval_s = float(
            getattr(self.scheduler_config, "adaptive_prefill_control_interval_s", 0.25)
        )
        if (
            self.last_update_monotonic_s
            and now - self.last_update_monotonic_s < interval_s
        ):
            return self._policy
        self._trim_window(now)
        signals = self._pressure or AdaptivePrefillSignals(
            samples=0,
            ttft_violation_ratio=0.0,
            tbt_violation_ratio=0.0,
            joint_attainment_ratio=0.0,
            goodput_rps=0.0,
            oldest_waiting_prefill_age_s=0.0,
            waiting_prefill_count=0,
            active_decode_count=0,
            kv_cache_usage=0.0,
            swap_failure_rate=0.0,
            recompute_tokens=0,
        )
        next_state, next_level, reason = self._decide_next(now, signals)
        if next_state != self.state:
            self._state_epochs = 0
        else:
            self._state_epochs += 1
        self.state = next_state
        self.level = next_level
        self.last_update_monotonic_s = now
        self._updates_total += 1
        if self.state == AdaptivePrefillState.OVERLOAD:
            self._overload_epochs_total += 1
        self._last_reason = reason
        self._policy = self._policy_for_level(self.level, self.state, reason)
        return self._policy

    def _decide_next(
        self,
        now: float,
        signals: AdaptivePrefillSignals,
    ) -> tuple[AdaptivePrefillState, int, str]:
        if self.state == AdaptivePrefillState.FAIL_SAFE:
            return AdaptivePrefillState.FAIL_SAFE, -2, "fail_safe"
        max_swap_failures = int(
            getattr(
                self.scheduler_config,
                "adaptive_prefill_max_swap_failures_per_window",
                3,
            )
        )
        max_recompute = int(
            getattr(
                self.scheduler_config,
                "adaptive_prefill_max_recompute_tokens_per_window",
                4096,
            )
        )
        if (
            signals.swap_failure_rate >= 1.0 and max_swap_failures <= 0
        ) or signals.recompute_tokens > max_recompute:
            self._enter_fail_safe("swap_instability")
            return AdaptivePrefillState.FAIL_SAFE, -2, "swap_instability"

        min_samples = int(
            getattr(self.scheduler_config, "adaptive_prefill_min_samples", 8)
        )
        if signals.samples < min_samples:
            return self._cold_start_decision(signals)

        target_ttft_attain = float(
            getattr(
                self.scheduler_config, "adaptive_prefill_target_ttft_attainment", 0.95
            )
        )
        target_tbt_attain = float(
            getattr(
                self.scheduler_config, "adaptive_prefill_target_tbt_attainment", 0.99
            )
        )
        ttft_bad_high = max(0.0, 1.0 - target_ttft_attain)
        tbt_bad_high = max(0.0, 1.0 - target_tbt_attain)
        tbt_emergency = float(
            getattr(self.scheduler_config, "adaptive_prefill_tbt_emergency_ratio", 0.02)
        )
        max_step = max(
            1,
            int(getattr(self.scheduler_config, "adaptive_prefill_max_level_step", 1)),
        )

        if signals.tbt_violation_ratio > max(tbt_bad_high, tbt_emergency):
            return (
                AdaptivePrefillState.DECODE_PROTECT,
                max(-2, self.level - max_step),
                "tbt_violation",
            )
        if (
            signals.ttft_violation_ratio > ttft_bad_high
            and signals.tbt_violation_ratio > tbt_bad_high
        ):
            return AdaptivePrefillState.OVERLOAD, min(0, self.level), "both_slos_bad"
        if signals.ttft_violation_ratio > ttft_bad_high:
            if now < self._cooldown_until_s:
                return self.state, self.level, "cooldown"
            self._cooldown_until_s = now + float(
                getattr(self.scheduler_config, "adaptive_prefill_cooldown_s", 0.25)
            )
            return (
                AdaptivePrefillState.PREFILL_RECOVERY,
                min(2, self.level + 1),
                "ttft_recovery",
            )
        if self.level > 0:
            return AdaptivePrefillState.BALANCED, self.level - 1, "healthy_decay"
        if self.level < 0 and signals.tbt_violation_ratio <= tbt_bad_high * 0.5:
            return AdaptivePrefillState.BALANCED, self.level + 1, "decode_recovery"
        return AdaptivePrefillState.BALANCED, self.level, "healthy"

    def _cold_start_decision(
        self,
        signals: AdaptivePrefillSignals,
    ) -> tuple[AdaptivePrefillState, int, str]:
        ttft_slo = float(
            getattr(self.scheduler_config, "adaptive_prefill_ttft_slo_s", 0.0)
        )
        if (
            self.level < 1
            and ttft_slo > 0.0
            and signals.waiting_prefill_count > 0
            and signals.oldest_waiting_prefill_age_s >= ttft_slo * 0.5
            and signals.active_decode_count == 0
        ):
            return AdaptivePrefillState.COLD_START, 1, "cold_start_pressure"
        return AdaptivePrefillState.COLD_START, self.level, "insufficient_samples"

    def _baseline_policy(self, state: AdaptivePrefillState) -> AdaptivePrefillPolicy:
        return AdaptivePrefillPolicy(
            level=0,
            state=state,
            prefill_bias_reserve_tokens=int(
                getattr(self.scheduler_config, "prefill_bias_reserve_tokens", 0)
            ),
            prefill_bias_wait_threshold_s=float(
                getattr(self.scheduler_config, "prefill_bias_wait_threshold_s", 0.03)
            ),
            prefill_bias_max_requests_per_step=int(
                getattr(self.scheduler_config, "prefill_bias_max_requests_per_step", 1)
            ),
            prefill_bias_score_window_k=int(
                getattr(self.scheduler_config, "prefill_bias_score_window_k", 16)
            ),
            prefill_bias_tbt_safety_margin_s=float(
                getattr(
                    self.scheduler_config, "prefill_bias_tbt_safety_margin_s", 0.005
                )
            ),
            prefill_bias_slot_swap_enabled=bool(
                getattr(self.scheduler_config, "prefill_bias_slot_swap_enabled", False)
            ),
            prefill_bias_max_swaps_per_step=int(
                getattr(self.scheduler_config, "prefill_bias_max_swaps_per_step", 1)
            ),
            prefill_bias_max_candidate_remaining_tokens=int(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_max_candidate_remaining_tokens",
                    256,
                )
            ),
        )

    def _policy_for_level(
        self,
        level: int,
        state: AdaptivePrefillState,
        reason: str,
    ) -> AdaptivePrefillPolicy:
        baseline = self._baseline_policy(state)
        min_reserve = int(
            getattr(self.scheduler_config, "adaptive_prefill_min_reserve_tokens", 0)
        )
        max_reserve = int(
            getattr(
                self.scheduler_config,
                "adaptive_prefill_max_reserve_tokens",
                max(baseline.prefill_bias_reserve_tokens, 128),
            )
        )
        min_chunk = int(
            getattr(self.scheduler_config, "adaptive_prefill_min_chunk_tokens", 16)
        )
        max_chunk = int(
            getattr(
                self.scheduler_config,
                "adaptive_prefill_max_chunk_tokens",
                max(baseline.prefill_bias_max_candidate_remaining_tokens, 256),
            )
        )
        max_swaps = int(
            getattr(self.scheduler_config, "adaptive_prefill_max_swaps_per_epoch", 1)
        )
        reserve_step = max(1, max(16, baseline.prefill_bias_reserve_tokens // 2 or 16))
        reserve = self._clamp_int(
            baseline.prefill_bias_reserve_tokens + level * reserve_step,
            min_reserve,
            max_reserve,
        )
        wait_threshold = max(
            0.0,
            baseline.prefill_bias_wait_threshold_s * (1.0 - 0.25 * level),
        )
        safety_margin = max(
            0.0,
            baseline.prefill_bias_tbt_safety_margin_s * (1.0 - 0.2 * level),
        )
        if level < 0:
            safety_margin = baseline.prefill_bias_tbt_safety_margin_s * (1.0 - level)
        slot_swap_static = bool(
            getattr(self.scheduler_config, "prefill_bias_slot_swap_enabled", False)
        )
        slot_swap_enabled = (
            slot_swap_static
            and level >= 1
            and state != (AdaptivePrefillState.DECODE_PROTECT)
        )
        if state in (AdaptivePrefillState.OVERLOAD, AdaptivePrefillState.FAIL_SAFE):
            slot_swap_enabled = False
        return replace(
            baseline,
            level=level,
            state=state,
            prefill_bias_reserve_tokens=reserve,
            prefill_bias_wait_threshold_s=wait_threshold,
            prefill_bias_max_requests_per_step=1 if level <= 0 else 2,
            prefill_bias_score_window_k=self._clamp_int(
                baseline.prefill_bias_score_window_k + max(0, level) * 4,
                1,
                32,
            ),
            prefill_bias_tbt_safety_margin_s=safety_margin,
            prefill_bias_slot_swap_enabled=slot_swap_enabled,
            prefill_bias_max_swaps_per_step=(
                min(max_swaps, baseline.prefill_bias_max_swaps_per_step)
                if slot_swap_enabled
                else 1
            ),
            prefill_bias_max_candidate_remaining_tokens=self._clamp_int(
                baseline.prefill_bias_max_candidate_remaining_tokens + level * 64,
                min_chunk,
                max_chunk,
            ),
            reason=reason,
        )

    def _window_rates(self, now: float) -> tuple[int, float, float, float, float]:
        self._trim_window(now)
        samples = len(self._completed)
        if samples == 0:
            return 0, 0.0, 0.0, 0.0, 0.0
        ttft_ok = sum(1 for _, ok, _, _ in self._completed if ok)
        tbt_ok = sum(1 for _, _, ok, _ in self._completed if ok)
        joint_ok = sum(1 for _, _, _, ok in self._completed if ok)
        span = max(
            1e-6,
            self._completed[-1][0] - self._completed[0][0],
        )
        return (
            samples,
            1.0 - ttft_ok / samples,
            1.0 - tbt_ok / samples,
            joint_ok / samples,
            joint_ok / span,
        )

    def _trim_window(self, now: float) -> None:
        window_s = float(
            getattr(self.scheduler_config, "adaptive_prefill_window_s", 2.0)
        )
        max_samples = max(
            16,
            int(getattr(self.scheduler_config, "adaptive_prefill_min_samples", 8)) * 8,
        )
        while self._completed and now - self._completed[0][0] > window_s:
            self._completed.popleft()
        while len(self._completed) > max_samples:
            self._completed.popleft()

    def _enter_fail_safe(self, reason: str) -> None:
        self.state = AdaptivePrefillState.FAIL_SAFE
        self.level = -2
        self._fail_safe_total += 1
        self._last_reason = reason
        self._policy = self._policy_for_level(
            -2, AdaptivePrefillState.FAIL_SAFE, reason
        )

    @staticmethod
    def _valid_positive(value: float) -> bool:
        return math.isfinite(float(value)) and float(value) > 0.0

    @staticmethod
    def _clamp_int(value: int, lower: int, upper: int) -> int:
        return max(lower, min(upper, int(value)))


class PrefillBiasController:
    def __init__(
        self,
        scheduler_config: Any,
        *,
        time_fn: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.scheduler_config = scheduler_config
        self._time_fn = time_fn
        self._monotonic_clock = monotonic_clock

    def candidate_ttft_slack(
        self,
        *,
        now_wall: float,
        arrival_time: float,
        ttft_slo_s: float,
        predicted_prefill_s: float,
    ) -> float:
        now = float(now_wall)
        arrival = float(arrival_time)
        slo = float(ttft_slo_s)
        predicted = float(predicted_prefill_s)
        if (
            not math.isfinite(now)
            or not math.isfinite(arrival)
            or not math.isfinite(slo)
            or slo <= 0.0
            or not math.isfinite(predicted)
            or predicted <= 0.0
        ):
            raise ValueError("candidate TTFT inputs must be finite and valid")
        # Arrival time is wall-clock in vLLM v0.24.0. Do not mix it with
        # monotonic timestamps used for TBT/cooldown accounting.
        waiting_age_s = max(0.0, now - arrival)
        return slo - waiting_age_s - predicted

    def should_try_slot_swap(
        self,
        *,
        now_wall: float,
        arrival_time: float,
        ttft_slo_s: float,
        predicted_prefill_s: float,
        slack_threshold_s: float,
    ) -> tuple[bool, float]:
        slack = self.candidate_ttft_slack(
            now_wall=now_wall,
            arrival_time=arrival_time,
            ttft_slo_s=ttft_slo_s,
            predicted_prefill_s=predicted_prefill_s,
        )
        return slack <= float(slack_threshold_s), slack

    @staticmethod
    def is_safe_projected_tbt(*, projected_tbt_s: float, tbt_limit_s: float) -> bool:
        projected = float(projected_tbt_s)
        limit = float(tbt_limit_s)
        return (
            math.isfinite(projected)
            and math.isfinite(limit)
            and projected >= 0.0
            and projected < limit
        )

    @staticmethod
    def forced_victim_sort_key(
        *,
        tbt_slack_s: float,
        recompute_tokens: int,
        preemption_count: int,
        position: int,
        request_id: str,
    ) -> tuple[float, int, int, int, str]:
        slack = float(tbt_slack_s)
        if not math.isfinite(slack):
            raise ValueError("tbt_slack_s must be finite")
        return (
            -slack,
            max(0, int(recompute_tokens)),
            max(0, int(preemption_count)),
            max(0, int(position)),
            str(request_id),
        )

    @staticmethod
    def make_swap_result(
        *,
        performed: bool = False,
        candidate_request_id: str | None = None,
        victim_request_id: str | None = None,
        blocker_reason: SlotSwapBlocker = SlotSwapBlocker.NO_BLOCKER,
        reject_reason: SlotSwapRejectReason | None = None,
        predicted_ttft_slack_s: float | None = None,
        predicted_victim_tbt_s: float | None = None,
        candidate_remaining_tokens: int | None = None,
        victim_recompute_tokens: int | None = None,
    ) -> PrefillAdmissionSwapResult:
        return PrefillAdmissionSwapResult(
            performed=performed,
            candidate_request_id=candidate_request_id,
            victim_request_id=victim_request_id,
            blocker_reason=blocker_reason,
            reject_reason=reject_reason,
            predicted_ttft_slack_s=predicted_ttft_slack_s,
            predicted_victim_tbt_s=predicted_victim_tbt_s,
            candidate_remaining_tokens=candidate_remaining_tokens,
            victim_recompute_tokens=victim_recompute_tokens,
        )

    def evaluate_tbt_guard(
        self,
        *,
        now_monotonic: float,
        active_decode_request_ids: Iterable[str],
        last_output_ts: Mapping[str, float],
        tbt_slo_s: float,
        predicted_step_time_s: float,
        safety_margin_s: float,
        guard_unknown_decode: bool,
    ) -> TBTGuardSnapshot:
        now = float(now_monotonic)
        slo = float(tbt_slo_s)
        predicted = float(predicted_step_time_s)
        margin = float(safety_margin_s)
        if (
            not math.isfinite(now)
            or not math.isfinite(slo)
            or slo <= 0.0
            or not math.isfinite(predicted)
            or predicted <= 0.0
            or not math.isfinite(margin)
            or margin < 0.0
        ):
            return TBTGuardSnapshot(
                active_decode_count=0,
                known_decode_count=0,
                unknown_decode_count=0,
                oldest_output_gap_s=0.0,
                minimum_tbt_slack_s=0.0,
                predicted_step_time_s=0.0
                if not math.isfinite(predicted)
                else predicted,
                safety_margin_s=0.0 if not math.isfinite(margin) else margin,
                allowed=False,
                reason="invalid_tbt_guard_input",
            )

        request_ids = tuple(str(req_id) for req_id in active_decode_request_ids)
        active_count = len(request_ids)
        if active_count == 0:
            return TBTGuardSnapshot(
                active_decode_count=0,
                known_decode_count=0,
                unknown_decode_count=0,
                oldest_output_gap_s=0.0,
                minimum_tbt_slack_s=slo,
                predicted_step_time_s=predicted,
                safety_margin_s=margin,
                allowed=True,
                reason="no_active_decode",
            )

        gaps: list[float] = []
        slacks: list[float] = []
        unknown = 0
        for request_id in request_ids:
            ts = last_output_ts.get(request_id)
            if ts is None:
                unknown += 1
                continue
            ts_value = float(ts)
            if not math.isfinite(ts_value) or ts_value > now:
                return TBTGuardSnapshot(
                    active_decode_count=active_count,
                    known_decode_count=len(slacks),
                    unknown_decode_count=unknown,
                    oldest_output_gap_s=max(gaps) if gaps else 0.0,
                    minimum_tbt_slack_s=min(slacks) if slacks else 0.0,
                    predicted_step_time_s=predicted,
                    safety_margin_s=margin,
                    allowed=False,
                    reason="invalid_decode_clock",
                )
            gap_s = now - ts_value
            slack_s = slo - gap_s
            gaps.append(gap_s)
            slacks.append(slack_s)

        if unknown and guard_unknown_decode:
            return TBTGuardSnapshot(
                active_decode_count=active_count,
                known_decode_count=len(slacks),
                unknown_decode_count=unknown,
                oldest_output_gap_s=max(gaps) if gaps else 0.0,
                minimum_tbt_slack_s=min(slacks) if slacks else 0.0,
                predicted_step_time_s=predicted,
                safety_margin_s=margin,
                allowed=False,
                reason="unknown_decode_state",
            )

        if not slacks:
            return TBTGuardSnapshot(
                active_decode_count=active_count,
                known_decode_count=0,
                unknown_decode_count=unknown,
                oldest_output_gap_s=0.0,
                minimum_tbt_slack_s=slo,
                predicted_step_time_s=predicted,
                safety_margin_s=margin,
                allowed=True,
                reason="unknown_decode_ignored",
            )

        minimum_slack = min(slacks)
        oldest_gap = max(gaps)
        if minimum_slack < 0.0:
            reason = "tbt_already_late"
            allowed = False
        elif minimum_slack > predicted + margin:
            reason = "allowed"
            allowed = True
        else:
            reason = "insufficient_tbt_slack"
            allowed = False

        return TBTGuardSnapshot(
            active_decode_count=active_count,
            known_decode_count=len(slacks),
            unknown_decode_count=unknown,
            oldest_output_gap_s=oldest_gap,
            minimum_tbt_slack_s=minimum_slack,
            predicted_step_time_s=predicted,
            safety_margin_s=margin,
            allowed=allowed,
            reason=reason,
        )

    def decide(
        self,
        *,
        waiting: Iterable[Any],
        policy: Any,
        paused: bool,
        throttle_prefills: bool,
        max_safe_reserve: int,
        is_request_schedulable: Callable[[Any], bool] | None = None,
        peek_cached_tokens: Callable[[Any], int] | None = None,
        predicted_step_time_s: float | None = None,
    ) -> PrefillBiasDecision:
        if bool(
            getattr(
                self.scheduler_config,
                "prefill_bias_ttft_deadline_enabled",
                False,
            )
        ):
            return self._deadline_decision(
                waiting=waiting,
                policy=policy,
                paused=paused,
                throttle_prefills=throttle_prefills,
                max_safe_reserve=max_safe_reserve,
                is_request_schedulable=is_request_schedulable,
                peek_cached_tokens=peek_cached_tokens,
                predicted_step_time_s=predicted_step_time_s,
            )

        base = self._base_decision(
            waiting=waiting,
            policy=policy,
            paused=paused,
            throttle_prefills=throttle_prefills,
            max_safe_reserve=max_safe_reserve,
            is_request_schedulable=is_request_schedulable,
        )
        if not base.active:
            return base
        if not bool(getattr(self.scheduler_config, "prefill_bias_cache_aware", False)):
            return base
        # Phase 1 scores after the RUNNING loop because cache residency can
        # change during RUNNING allocation/preemption. Until then, hold budget
        # but defer final candidate ordering.
        return PrefillBiasDecision(
            active=True,
            reserve_tokens=base.reserve_tokens,
            candidate_request_ids=(),
            reason="active_cache_aware_pending",
        )

    def _deadline_decision(
        self,
        *,
        waiting: Iterable[Any],
        policy: Any,
        paused: bool,
        throttle_prefills: bool,
        max_safe_reserve: int,
        is_request_schedulable: Callable[[Any], bool] | None,
        peek_cached_tokens: Callable[[Any], int] | None,
        predicted_step_time_s: float | None,
    ) -> PrefillBiasDecision:
        if not bool(getattr(self.scheduler_config, "prefill_bias_enabled", False)):
            return self._inactive("disabled")
        policy_value = str(getattr(policy, "value", policy)).lower()
        if policy_value != "fcfs":
            return self._inactive("unsupported_policy")
        if paused:
            return self._inactive("paused")
        if throttle_prefills:
            return self._inactive("prefill_throttled")
        if peek_cached_tokens is None:
            return self._inactive("deadline_cache_probe_unavailable")

        step_time_s = float(predicted_step_time_s or 0.0)
        ttft_slo_s = float(
            getattr(self.scheduler_config, "prefill_bias_ttft_slo_s", 0.0) or 0.0
        )
        trigger_margin_s = float(
            getattr(
                self.scheduler_config,
                "prefill_bias_swap_slack_threshold_s",
                0.0,
            )
        )
        if (
            not math.isfinite(step_time_s)
            or step_time_s <= 0.0
            or not math.isfinite(ttft_slo_s)
            or ttft_slo_s <= 0.0
            or not math.isfinite(trigger_margin_s)
        ):
            return self._inactive("invalid_deadline_input")

        reserve_tokens = min(
            int(getattr(self.scheduler_config, "prefill_bias_reserve_tokens", 0)),
            max(0, int(max_safe_reserve)),
        )
        if reserve_tokens <= 0:
            return self._inactive("no_safe_budget")

        now = self._time_fn()
        min_cached_tokens = max(
            0,
            int(getattr(self.scheduler_config, "prefill_bias_min_cached_tokens", 0)),
        )
        edges = tuple(
            int(edge)
            for edge in getattr(
                self.scheduler_config,
                "prefill_bias_remaining_token_buckets",
                (16, 64, 256, 1024),
            )
        )
        starvation_s = max(
            0.0,
            float(getattr(self.scheduler_config, "prefill_bias_starvation_s", 0.2)),
        )
        force_enabled = bool(
            getattr(
                self.scheduler_config,
                "prefill_bias_ttft_force_preempt_enabled",
                False,
            )
        )

        candidates: list[PrefillCandidate] = []
        for request in waiting:
            if not self._is_normal_waiting_prefill(
                request,
                is_request_schedulable=is_request_schedulable,
            ):
                continue
            candidate = self._build_deadline_candidate(
                request,
                now=now,
                ttft_slo_s=ttft_slo_s,
                trigger_margin_s=trigger_margin_s,
                reserve_tokens=reserve_tokens,
                predicted_step_time_s=step_time_s,
                starvation_s=starvation_s,
                min_cached_tokens=min_cached_tokens,
                edges=edges,
                force_enabled=force_enabled,
                peek_cached_tokens=peek_cached_tokens,
            )
            candidates.append(candidate)

        if not candidates:
            return self._inactive("no_waiting_prefill")

        minimum_slack_s = min(candidate.ttft_slack_s for candidate in candidates)
        urgent = [candidate for candidate in candidates if candidate.urgent]
        if not urgent:
            return PrefillBiasDecision(
                active=False,
                reserve_tokens=0,
                candidate_request_ids=(),
                reason="ttft_not_urgent",
                scored_requests=len(candidates),
                minimum_ttft_slack_s=minimum_slack_s,
            )

        urgent.sort(key=self._deadline_candidate_sort_key)
        selected = urgent[0]
        return PrefillBiasDecision(
            active=True,
            reserve_tokens=reserve_tokens,
            candidate_request_ids=(selected.request_id,),
            reason=(
                "active_forced_ttft" if selected.forced_ttft else "active_ttft_deadline"
            ),
            scored_requests=len(candidates),
            selected_cached_tokens=selected.cached_tokens,
            selected_remaining_tokens=selected.remaining_prefill_tokens,
            minimum_ttft_slack_s=minimum_slack_s,
            predicted_completion_s=selected.predicted_completion_s,
            forced_ttft=selected.forced_ttft,
        )

    def _build_deadline_candidate(
        self,
        request: Any,
        *,
        now: float,
        ttft_slo_s: float,
        trigger_margin_s: float,
        reserve_tokens: int,
        predicted_step_time_s: float,
        starvation_s: float,
        min_cached_tokens: int,
        edges: tuple[int, ...],
        force_enabled: bool,
        peek_cached_tokens: Callable[[Any], int],
    ) -> PrefillCandidate:
        arrival_time = float(getattr(request, "arrival_time", 0.0) or 0.0)
        waiting_age_s = max(0.0, now - arrival_time)
        cached_tokens = max(0, int(peek_cached_tokens(request)))
        effective_cached_tokens = (
            cached_tokens if cached_tokens >= min_cached_tokens else 0
        )
        total_prefill_value = getattr(request, "num_prompt_tokens", None)
        if total_prefill_value is None:
            total_prefill_value = getattr(request, "num_tokens", 0)
        total_prefill_tokens = max(1, int(total_prefill_value or 0))
        request_computed_tokens = max(
            0,
            int(getattr(request, "num_computed_tokens", 0) or 0),
        )
        remaining_prefill_tokens = compute_remaining_prefill_tokens(
            total_prefill_tokens=total_prefill_tokens,
            request_computed_tokens=request_computed_tokens,
            cached_prefix_tokens=effective_cached_tokens,
            minimum_required_tokens=1,
        )
        num_steps = max(
            1,
            math.ceil(remaining_prefill_tokens / max(1, reserve_tokens)),
        )
        predicted_completion_s = num_steps * predicted_step_time_s
        ttft_slack_s = ttft_slo_s - waiting_age_s - predicted_completion_s
        urgent = ttft_slack_s <= trigger_margin_s or math.isclose(
            ttft_slack_s,
            trigger_margin_s,
            rel_tol=0.0,
            abs_tol=_DEADLINE_EPSILON_S,
        )
        forced_ttft = force_enabled and (
            ttft_slack_s <= 0.0
            or math.isclose(
                ttft_slack_s,
                0.0,
                rel_tol=0.0,
                abs_tol=_DEADLINE_EPSILON_S,
            )
        )
        return PrefillCandidate(
            request_id=str(getattr(request, "request_id", "")),
            waiting_age_s=waiting_age_s,
            arrival_time=arrival_time,
            total_request_tokens=total_prefill_tokens,
            cached_tokens=cached_tokens,
            remaining_prefill_tokens=remaining_prefill_tokens,
            remaining_bucket=bucket_remaining_tokens(remaining_prefill_tokens, edges),
            starved=waiting_age_s >= starvation_s,
            predicted_completion_s=predicted_completion_s,
            ttft_slack_s=ttft_slack_s,
            urgent=urgent,
            forced_ttft=forced_ttft,
        )

    @staticmethod
    def _deadline_candidate_sort_key(candidate: PrefillCandidate) -> tuple[Any, ...]:
        return (
            round(candidate.ttft_slack_s, 9),
            candidate.predicted_completion_s,
            candidate.remaining_prefill_tokens,
            candidate.arrival_time,
            candidate.request_id,
        )

    def score_candidates(
        self,
        *,
        waiting: Iterable[Any],
        policy: Any,
        reserve_tokens: int,
        peek_cached_tokens: Callable[[Any], int],
        is_request_schedulable: Callable[[Any], bool] | None = None,
    ) -> PrefillBiasDecision:
        if not bool(getattr(self.scheduler_config, "prefill_bias_cache_aware", False)):
            return PrefillBiasDecision(
                active=True,
                reserve_tokens=reserve_tokens,
                candidate_request_ids=(),
                reason="cache_aware_disabled",
            )

        policy_value = str(getattr(policy, "value", policy)).lower()
        if policy_value != "fcfs":
            return PrefillBiasDecision(False, 0, (), "unsupported_policy")

        now = self._time_fn()
        threshold_s = max(
            0.0,
            float(getattr(self.scheduler_config, "prefill_bias_wait_threshold_s", 0.0)),
        )
        max_requests = max(
            1,
            int(
                getattr(self.scheduler_config, "prefill_bias_max_requests_per_step", 1)
            ),
        )
        score_window_k = max(
            1, int(getattr(self.scheduler_config, "prefill_bias_score_window_k", 16))
        )
        candidate_scan_limit = max(
            max_requests,
            int(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_candidate_scan_limit",
                    score_window_k,
                )
            ),
        )
        min_cached_tokens = max(
            0, int(getattr(self.scheduler_config, "prefill_bias_min_cached_tokens", 0))
        )
        starvation_s = max(
            0.0, float(getattr(self.scheduler_config, "prefill_bias_starvation_s", 0.2))
        )
        edges = tuple(
            int(edge)
            for edge in getattr(
                self.scheduler_config,
                "prefill_bias_remaining_token_buckets",
                (16, 64, 256, 1024),
            )
        )

        candidates: list[PrefillCandidate] = []
        inspected = 0
        for request in waiting:
            if inspected >= candidate_scan_limit or len(candidates) >= score_window_k:
                break
            if not self._is_normal_waiting_prefill(
                request, is_request_schedulable=is_request_schedulable
            ):
                continue
            inspected += 1
            candidate = self._build_candidate(
                request,
                now=now,
                threshold_s=threshold_s,
                starvation_s=starvation_s,
                min_cached_tokens=min_cached_tokens,
                edges=edges,
                peek_cached_tokens=peek_cached_tokens,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return PrefillBiasDecision(False, 0, (), "below_wait_threshold")

        candidates.sort(key=self._candidate_sort_key)
        selected = candidates[:max_requests]
        return PrefillBiasDecision(
            active=True,
            reserve_tokens=reserve_tokens,
            candidate_request_ids=tuple(candidate.request_id for candidate in selected),
            reason="active_cache_aware",
            scored_requests=len(candidates),
            selected_cached_tokens=sum(
                candidate.cached_tokens for candidate in selected
            ),
            selected_remaining_tokens=sum(
                candidate.remaining_prefill_tokens for candidate in selected
            ),
        )

    def _base_decision(
        self,
        *,
        waiting: Iterable[Any],
        policy: Any,
        paused: bool,
        throttle_prefills: bool,
        max_safe_reserve: int,
        is_request_schedulable: Callable[[Any], bool] | None = None,
    ) -> PrefillBiasDecision:
        if not bool(getattr(self.scheduler_config, "prefill_bias_enabled", False)):
            return self._inactive("disabled")

        policy_value = str(getattr(policy, "value", policy)).lower()
        if policy_value != "fcfs":
            return self._inactive("unsupported_policy")

        if paused:
            return self._inactive("paused")

        if throttle_prefills:
            return self._inactive("prefill_throttled")

        now = self._time_fn()
        threshold_s = max(
            0.0,
            float(getattr(self.scheduler_config, "prefill_bias_wait_threshold_s", 0.0)),
        )
        max_requests = max(
            1,
            int(
                getattr(self.scheduler_config, "prefill_bias_max_requests_per_step", 1)
            ),
        )

        candidates: list[tuple[float, float, str, Any]] = []
        saw_waiting_prefill = False
        for request in waiting:
            if not self._is_normal_waiting_prefill(
                request, is_request_schedulable=is_request_schedulable
            ):
                continue
            saw_waiting_prefill = True
            arrival_time = float(getattr(request, "arrival_time", 0.0) or 0.0)
            age_s = max(0.0, now - arrival_time)
            if age_s >= threshold_s:
                request_id = str(getattr(request, "request_id", ""))
                candidates.append((-age_s, arrival_time, request_id, request))

        if not saw_waiting_prefill:
            return self._inactive("no_waiting_prefill")
        if not candidates:
            return self._inactive("below_wait_threshold")

        reserve_tokens = min(
            int(getattr(self.scheduler_config, "prefill_bias_reserve_tokens", 0)),
            max(0, int(max_safe_reserve)),
        )
        if reserve_tokens <= 0:
            return self._inactive("no_safe_budget")

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        request_ids = tuple(item[2] for item in candidates[:max_requests])
        return PrefillBiasDecision(
            active=True,
            reserve_tokens=reserve_tokens,
            candidate_request_ids=request_ids,
            reason="active",
        )

    def _build_candidate(
        self,
        request: Any,
        *,
        now: float,
        threshold_s: float,
        starvation_s: float,
        min_cached_tokens: int,
        edges: tuple[int, ...],
        peek_cached_tokens: Callable[[Any], int],
    ) -> PrefillCandidate | None:
        arrival_time = float(getattr(request, "arrival_time", 0.0) or 0.0)
        waiting_age_s = max(0.0, now - arrival_time)
        if waiting_age_s < threshold_s:
            return None

        cached_tokens = max(0, int(peek_cached_tokens(request)))
        effective_cached_tokens = (
            cached_tokens if cached_tokens >= min_cached_tokens else 0
        )
        total_prefill_value = getattr(request, "num_prompt_tokens", None)
        if total_prefill_value is None:
            total_prefill_value = getattr(request, "num_tokens", 0)
        total_prefill_tokens = max(1, int(total_prefill_value or 0))
        request_computed_tokens = max(
            0,
            int(getattr(request, "num_computed_tokens", 0) or 0),
        )
        remaining_prefill_tokens = compute_remaining_prefill_tokens(
            total_prefill_tokens=total_prefill_tokens,
            request_computed_tokens=request_computed_tokens,
            cached_prefix_tokens=effective_cached_tokens,
            minimum_required_tokens=1,
        )
        return PrefillCandidate(
            request_id=str(getattr(request, "request_id", "")),
            waiting_age_s=waiting_age_s,
            arrival_time=arrival_time,
            total_request_tokens=total_prefill_tokens,
            cached_tokens=cached_tokens,
            remaining_prefill_tokens=remaining_prefill_tokens,
            remaining_bucket=bucket_remaining_tokens(remaining_prefill_tokens, edges),
            starved=waiting_age_s >= starvation_s,
            cache_peek_supported=True,
        )

    @staticmethod
    def _candidate_sort_key(candidate: PrefillCandidate) -> tuple[Any, ...]:
        if candidate.starved:
            return (0, candidate.arrival_time, candidate.request_id)
        return (
            1,
            candidate.remaining_bucket,
            candidate.remaining_prefill_tokens,
            candidate.arrival_time,
            candidate.request_id,
        )

    @staticmethod
    def _is_normal_waiting_prefill(
        request: Any,
        *,
        is_request_schedulable: Callable[[Any], bool] | None,
    ) -> bool:
        if is_request_schedulable is not None and not is_request_schedulable(request):
            return False
        status = getattr(request, "status", None)
        if str(getattr(status, "name", status)) != "WAITING":
            return False
        if bool(getattr(request, "is_prefill_chunk", False)):
            return False
        if bool(getattr(request, "abort_immediately", False)):
            return False
        return True

    @staticmethod
    def _inactive(reason: str) -> PrefillBiasDecision:
        return PrefillBiasDecision(
            active=False,
            reserve_tokens=0,
            candidate_request_ids=(),
            reason=reason,
        )


class PredictivePrefillBiasController(PrefillBiasController):
    """Deadline policy with bounded online latency prediction."""

    _MAX_SCAN = 32

    def __init__(
        self,
        scheduler_config: Any,
        *,
        time_fn: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        estimator: PredictivePrefillEstimator | None = None,
    ) -> None:
        super().__init__(
            scheduler_config,
            time_fn=time_fn,
            monotonic_clock=monotonic_clock,
        )
        self.estimator = estimator or PredictivePrefillEstimator(
            ewma_alpha=float(
                getattr(scheduler_config, "prefill_bias_step_time_ewma_alpha", 0.2)
            )
        )

    def observe_batch(
        self,
        *,
        duration_s: float,
        prefill_tokens: int,
        active_decode_count: int,
    ) -> None:
        self.estimator.observe(
            duration_s=duration_s,
            prefill_tokens=prefill_tokens,
            active_decode_count=active_decode_count,
        )

    def decide(
        self,
        *,
        waiting: Iterable[Any],
        policy: Any,
        paused: bool,
        throttle_prefills: bool,
        max_safe_reserve: int,
        is_request_schedulable: Callable[[Any], bool] | None = None,
        peek_cached_tokens: Callable[[Any], int] | None = None,
        predicted_step_time_s: float | None = None,
        active_decode_count: int = 0,
        minimum_tbt_slack_s: float = math.inf,
        unknown_decode_count: int = 0,
    ) -> PrefillBiasDecision:
        del predicted_step_time_s
        if not bool(getattr(self.scheduler_config, "prefill_bias_enabled", False)):
            return self._inactive("disabled")
        policy_value = str(getattr(policy, "value", policy)).lower()
        if policy_value != "fcfs":
            return self._inactive("unsupported_policy")
        if paused:
            return self._inactive("paused")
        if throttle_prefills:
            return self._inactive("prefill_throttled")
        if peek_cached_tokens is None:
            return self._inactive("predictive_cache_probe_unavailable")

        active_decode_count = max(0, int(active_decode_count))
        if active_decode_count > 0 and int(unknown_decode_count) > 0:
            return self._inactive("predictive_unknown_decode_state")

        max_reserve = min(
            max(0, int(max_safe_reserve)),
            max(
                0,
                int(
                    getattr(
                        self.scheduler_config,
                        "prefill_bias_predictive_max_reserve_tokens",
                        3072,
                    )
                ),
            ),
        )
        if max_reserve <= 0:
            return self._inactive("no_safe_budget")

        candidates = self._predictive_candidates(
            waiting=waiting,
            max_reserve=max_reserve,
            active_decode_count=active_decode_count,
            peek_cached_tokens=peek_cached_tokens,
            is_request_schedulable=is_request_schedulable,
        )
        if not candidates:
            return self._inactive("no_waiting_prefill")

        urgent = [candidate for candidate in candidates if candidate.urgent]
        minimum_slack = min(candidate.ttft_slack_s for candidate in candidates)
        if not urgent:
            return PrefillBiasDecision(
                active=False,
                reserve_tokens=0,
                candidate_request_ids=(),
                reason="predictive_ttft_not_urgent",
                scored_requests=len(candidates),
                minimum_ttft_slack_s=minimum_slack,
                policy_mode="predictive",
            )

        urgent.sort(key=self._predictive_candidate_sort_key)
        total_remaining = sum(candidate.remaining_prefill_tokens for candidate in urgent)
        selected_budget = self._select_predictive_budget(
            upper_bound=min(max_reserve, total_remaining),
            urgent=urgent,
            active_decode_count=active_decode_count,
            minimum_tbt_slack_s=minimum_tbt_slack_s,
        )
        if selected_budget <= 0:
            return PrefillBiasDecision(
                active=False,
                reserve_tokens=0,
                candidate_request_ids=(),
                reason="predictive_no_tbt_safe_chunk",
                scored_requests=len(candidates),
                minimum_ttft_slack_s=minimum_slack,
                policy_mode="predictive",
            )

        max_requests = max(
            1,
            int(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_predictive_max_requests_per_step",
                    4,
                )
            ),
        )
        caps: list[tuple[str, int]] = []
        remaining_budget = selected_budget
        selected_candidates: list[PrefillCandidate] = []
        for candidate in urgent:
            if remaining_budget <= 0 or len(caps) >= max_requests:
                break
            cap = min(candidate.remaining_prefill_tokens, remaining_budget)
            if cap <= 0:
                continue
            caps.append((candidate.request_id, cap))
            selected_candidates.append(candidate)
            remaining_budget -= cap

        reserve_tokens = sum(cap for _, cap in caps)
        if reserve_tokens <= 0:
            return self._inactive("predictive_no_allocation")

        predicted_step = self._predict_step(
            prefill_tokens=reserve_tokens,
            active_decode_count=active_decode_count,
        )
        selected = selected_candidates[0]
        force_enabled = bool(
            getattr(
                self.scheduler_config,
                "prefill_bias_ttft_force_preempt_enabled",
                False,
            )
        )
        forced = force_enabled and (
            selected.ttft_slack_s <= 0.0 or selected.starved
        )
        return PrefillBiasDecision(
            active=True,
            reserve_tokens=reserve_tokens,
            candidate_request_ids=tuple(request_id for request_id, _ in caps),
            reason=(
                "predictive_forced_ttft" if forced else "predictive_ttft_deadline"
            ),
            scored_requests=len(candidates),
            selected_cached_tokens=sum(item.cached_tokens for item in selected_candidates),
            selected_remaining_tokens=sum(
                item.remaining_prefill_tokens for item in selected_candidates
            ),
            minimum_ttft_slack_s=minimum_slack,
            predicted_completion_s=selected.predicted_completion_s,
            forced_ttft=forced,
            candidate_token_caps=tuple(caps),
            predicted_step_time_s=predicted_step,
            policy_mode="predictive",
            slot_swap_eligible=(selected.ttft_slack_s <= 0.0 or selected.starved),
        )

    def _predictive_candidates(
        self,
        *,
        waiting: Iterable[Any],
        max_reserve: int,
        active_decode_count: int,
        peek_cached_tokens: Callable[[Any], int],
        is_request_schedulable: Callable[[Any], bool] | None,
    ) -> list[PrefillCandidate]:
        now = self._time_fn()
        ttft_slo_s = max(
            1e-9,
            float(getattr(self.scheduler_config, "prefill_bias_ttft_slo_s", 0.0)),
        )
        trigger_margin_s = max(
            0.0,
            float(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_swap_slack_threshold_s",
                    0.0,
                )
            ),
        )
        starvation_multiplier = max(
            1.0,
            float(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_predictive_starvation_multiplier",
                    4.0,
                )
            ),
        )
        hard_starvation_s = max(
            float(getattr(self.scheduler_config, "prefill_bias_starvation_s", 0.2)),
            starvation_multiplier * ttft_slo_s,
        )
        scan_limit = min(
            self._MAX_SCAN,
            max(
                1,
                int(
                    getattr(
                        self.scheduler_config,
                        "prefill_bias_candidate_scan_limit",
                        self._MAX_SCAN,
                    )
                ),
            ),
        )
        min_cached_tokens = max(
            0,
            int(getattr(self.scheduler_config, "prefill_bias_min_cached_tokens", 0)),
        )
        edges = tuple(
            int(edge)
            for edge in getattr(
                self.scheduler_config,
                "prefill_bias_remaining_token_buckets",
                (16, 64, 256, 1024),
            )
        )

        candidates: list[PrefillCandidate] = []
        inspected = 0
        for request in waiting:
            if inspected >= scan_limit:
                break
            if not self._is_normal_waiting_prefill(
                request,
                is_request_schedulable=is_request_schedulable,
            ):
                continue
            inspected += 1
            arrival_time = float(getattr(request, "arrival_time", 0.0) or 0.0)
            waiting_age_s = max(0.0, now - arrival_time)
            cached_tokens = max(0, int(peek_cached_tokens(request)))
            effective_cached = cached_tokens if cached_tokens >= min_cached_tokens else 0
            total_value = getattr(request, "num_prompt_tokens", None)
            if total_value is None:
                total_value = getattr(request, "num_tokens", 0)
            total_tokens = max(1, int(total_value or 0))
            computed_tokens = max(
                0,
                int(getattr(request, "num_computed_tokens", 0) or 0),
            )
            remaining_tokens = compute_remaining_prefill_tokens(
                total_prefill_tokens=total_tokens,
                request_computed_tokens=computed_tokens,
                cached_prefix_tokens=effective_cached,
                minimum_required_tokens=1,
            )
            predicted_completion = self._predict_completion(
                remaining_tokens=remaining_tokens,
                per_step_cap=max_reserve,
                active_decode_count=active_decode_count,
            )
            slack = ttft_slo_s - waiting_age_s - predicted_completion
            hard_starved = waiting_age_s >= hard_starvation_s
            salvageable = slack >= 0.0
            urgent = hard_starved or slack <= trigger_margin_s or math.isclose(
                slack,
                trigger_margin_s,
                rel_tol=0.0,
                abs_tol=_DEADLINE_EPSILON_S,
            )
            candidates.append(
                PrefillCandidate(
                    request_id=str(getattr(request, "request_id", "")),
                    waiting_age_s=waiting_age_s,
                    arrival_time=arrival_time,
                    total_request_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    remaining_prefill_tokens=remaining_tokens,
                    remaining_bucket=bucket_remaining_tokens(remaining_tokens, edges),
                    starved=hard_starved,
                    predicted_completion_s=predicted_completion,
                    ttft_slack_s=slack,
                    urgent=urgent,
                    salvageable=salvageable,
                )
            )
        return candidates

    def _predict_completion(
        self,
        *,
        remaining_tokens: int,
        per_step_cap: int,
        active_decode_count: int,
    ) -> float:
        remaining = max(1, int(remaining_tokens))
        cap = max(1, int(per_step_cap))
        full_steps, tail = divmod(remaining, cap)
        full_step_s = self._predict_step(
            prefill_tokens=cap,
            active_decode_count=active_decode_count,
        )
        total = full_steps * full_step_s
        if tail:
            total += self._predict_step(
                prefill_tokens=tail,
                active_decode_count=active_decode_count,
            )
        return total

    def _predict_step(self, *, prefill_tokens: int, active_decode_count: int) -> float:
        return self.estimator.predict(
            prefill_tokens=prefill_tokens,
            active_decode_count=active_decode_count,
            initial_step_time_s=float(
                getattr(self.scheduler_config, "prefill_bias_initial_step_time_s", 0.01)
            ),
            legacy_reserve_tokens=max(
                1,
                int(getattr(self.scheduler_config, "prefill_bias_reserve_tokens", 64)),
            ),
            headroom_factor=float(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_step_time_headroom_factor",
                    1.25,
                )
            ),
            min_samples=int(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_step_observation_min_samples",
                    3,
                )
            ),
        )

    def _select_predictive_budget(
        self,
        *,
        upper_bound: int,
        urgent: list[PrefillCandidate],
        active_decode_count: int,
        minimum_tbt_slack_s: float,
    ) -> int:
        upper = max(0, int(upper_bound))
        if upper <= 0:
            return 0
        min_chunk = max(
            1,
            int(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_predictive_min_chunk_tokens",
                    128,
                )
            ),
        )
        budgets = {
            budget
            for budget in PredictivePrefillEstimator._PREFILL_BUCKETS
            if min_chunk <= budget <= upper
        }
        if upper < min_chunk and any(
            candidate.remaining_prefill_tokens <= upper for candidate in urgent
        ):
            budgets.add(upper)
        elif upper >= min_chunk:
            budgets.add(upper)
        if not budgets:
            return 0
        if active_decode_count <= 0:
            return max(budgets)
        if not math.isfinite(minimum_tbt_slack_s):
            return 0

        tbt_slo_s = max(
            1e-9,
            float(getattr(self.scheduler_config, "prefill_bias_tbt_slo_s", 0.0)),
        )
        margin_s = max(
            0.0,
            float(
                getattr(
                    self.scheduler_config,
                    "prefill_bias_tbt_safety_margin_s",
                    0.0,
                )
            ),
        )
        next_decode_s = self._predict_step(
            prefill_tokens=0,
            active_decode_count=active_decode_count,
        )
        best = 0
        for budget in sorted(budgets):
            current_s = self._predict_step(
                prefill_tokens=budget,
                active_decode_count=active_decode_count,
            )
            current_ok = current_s <= minimum_tbt_slack_s - margin_s
            next_ok = (
                current_s + next_decode_s
                <= minimum_tbt_slack_s + tbt_slo_s - margin_s
            )
            if current_ok and next_ok:
                best = budget
        return best

    @staticmethod
    def _predictive_candidate_sort_key(candidate: PrefillCandidate) -> tuple[Any, ...]:
        return (
            0 if candidate.starved else 1,
            0 if candidate.salvageable else 1,
            candidate.ttft_slack_s if candidate.salvageable else 0.0,
            candidate.remaining_prefill_tokens,
            candidate.predicted_completion_s,
            candidate.arrival_time,
            candidate.request_id,
        )


class PrefillBiasPolicyRouter:
    """Runtime-selectable legacy, shadow, and predictive policy facade."""

    VALID_MODES = {
        "legacy",
        "predictive-shadow",
        "predictive",
        "batch-budget-shadow",
        "batch-budget",
    }

    def __init__(
        self,
        scheduler_config: Any,
        *,
        time_fn: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._scheduler_config = scheduler_config
        self.legacy = PrefillBiasController(
            scheduler_config,
            time_fn=time_fn,
            monotonic_clock=monotonic_clock,
        )
        self.predictive = PredictivePrefillBiasController(
            scheduler_config,
            time_fn=time_fn,
            monotonic_clock=monotonic_clock,
        )
        self.batch_budget = PrefillBatchBudgetController(
            scheduler_config,
            estimator=self.predictive.estimator,
            time_fn=time_fn,
        )
        self.last_legacy_decision: PrefillBiasDecision | None = None
        self.last_predictive_decision: PrefillBiasDecision | None = None
        self.last_applied_mode = "legacy"
        self.predictive_decisions_total = 0
        self.predictive_would_activate_total = 0
        self.predictive_fallback_to_legacy_total = 0
        self.shadow_candidate_overlap_total = 0

    @property
    def scheduler_config(self) -> Any:
        return self._scheduler_config

    @scheduler_config.setter
    def scheduler_config(self, value: Any) -> None:
        self._scheduler_config = value
        self.legacy.scheduler_config = value
        self.predictive.scheduler_config = value
        self.batch_budget.scheduler_config = value

    def _mode(self) -> str:
        mode = str(
            getattr(self.scheduler_config, "prefill_bias_policy_mode", "legacy")
        ).lower()
        return mode if mode in self.VALID_MODES else "legacy"

    def decide(self, **kwargs: Any) -> PrefillBiasDecision:
        legacy_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "waiting",
                "policy",
                "paused",
                "throttle_prefills",
                "max_safe_reserve",
                "is_request_schedulable",
                "peek_cached_tokens",
                "predicted_step_time_s",
            }
        }
        legacy_decision = self.legacy.decide(**legacy_kwargs)
        self.last_legacy_decision = legacy_decision
        mode = self._mode()
        self.last_applied_mode = mode
        if mode == "legacy":
            return legacy_decision
        if mode in {"batch-budget-shadow", "batch-budget"}:
            return legacy_decision

        self.predictive_decisions_total += 1
        try:
            predictive_decision = self.predictive.decide(**kwargs)
        except Exception:
            self.predictive_fallback_to_legacy_total += 1
            self.last_applied_mode = "legacy-fallback"
            return legacy_decision
        self.last_predictive_decision = predictive_decision
        if predictive_decision.active:
            self.predictive_would_activate_total += 1
        self.shadow_candidate_overlap_total += len(
            set(legacy_decision.candidate_request_ids)
            & set(predictive_decision.candidate_request_ids)
        )
        if mode == "predictive-shadow":
            return legacy_decision
        return predictive_decision

    def plan_batch(self, **kwargs: Any) -> PrefillBatchBudgetDecision:
        return self.batch_budget.plan(**kwargs)

    def score_candidates(self, **kwargs: Any) -> PrefillBiasDecision:
        return self.legacy.score_candidates(**kwargs)

    def observe_batch(
        self,
        *,
        duration_s: float,
        prefill_tokens: int,
        active_decode_count: int,
    ) -> None:
        self.predictive.observe_batch(
            duration_s=duration_s,
            prefill_tokens=prefill_tokens,
            active_decode_count=active_decode_count,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.legacy, name)

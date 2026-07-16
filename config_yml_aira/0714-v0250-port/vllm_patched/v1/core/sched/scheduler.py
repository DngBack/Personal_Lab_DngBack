# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import math
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
# VLLM_PREFILL_BIAS_PATCH: imports
from vllm.v1.core.sched.prefill_bias import (
    AdaptivePrefillController,
    DecodeTimingState,
    PrefillAdmissionSwapResult,
    PrefillBiasController,
    PrefillBatchBudgetDecision,
    PrefillBiasDecision,
    PrefillBiasPolicyRouter,
    PrefillBudgetAccountant,
    PrefillStepTimeEstimator,
    SlotSwapBlocker,
    SlotSwapRejectReason,
    TBTGuardSnapshot,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


@dataclass
class ScheduledBatchTiming:
    started_at: float
    prefill_tokens: int
    total_tokens: int
    active_decode_count: int


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        # Track requests scheduled in prior step (MRV1-only).
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = (
            1 if not vllm_config.model_config.is_diffusion else 0
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        self.defer_block_free = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                self.defer_block_free = True

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # IDs of requests preempted since the last call to schedule().
        self.reset_preempted_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_lookahead_tokens = 0
        self.dynamic_sd_lookup: list[int] | None = None
        if speculative_config is not None:
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.use_dflash():
                # DFlash requires an extra lookahead slot since it uses in-fill-style
                # decoding instead of standard next-token sampling, so it has a query
                # for the last sampled token plus queries for each draft token.
                self.num_lookahead_tokens = self.num_spec_tokens + 1
            if speculative_config.use_dspark():
                # DSpark drafts a block of num_spec_tokens query tokens in which the
                # anchor itself is the first prediction position (no separate bonus
                # query), so it needs exactly num_spec_tokens lookahead slots.
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            scheduler_block_size=self.block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
            watermark=self.scheduler_config.watermark,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        self.current_step = 0
        # DP prefill balancing: Flag to track whether the last cadence-aligned
        # prefill batch fully drained the waiting queue. Prefill throttling
        # is disabled in this case.
        self.prefill_capacity_bound = False
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )

        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        self.enable_return_routed_experts = (
            vllm_config.model_config.enable_return_routed_experts
        )

        if self.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_mgr = RoutedExpertsManager(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            self._re_block_ids: dict[str, list[int]] = {}

        self._pause_state: PauseState = PauseState.UNPAUSED

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

        # VLLM_PREFILL_BIAS_PATCH: controller and phase-0 counters
        self._prefill_bias_monotonic_clock = time.monotonic
        # VLLM_PREFILL_BIAS_PHASE8_PATCH: runtime policy router.
        self.prefill_bias_controller = PrefillBiasPolicyRouter(
            self.scheduler_config,
            monotonic_clock=self._prefill_bias_monotonic_clock,
        )
        self._prefill_blocked_token_budget = 0
        self._prefill_blocked_max_num_seqs = 0
        self._prefill_blocked_no_kv = 0
        self._prefill_bias_activations = 0
        self._prefill_bias_reserved_tokens = 0
        self._prefill_bias_admitted_requests = 0
        self._prefill_bias_scored_requests = 0
        self._prefill_bias_cache_aware_activations = 0
        self._prefill_bias_selected_cached_tokens = 0
        self._prefill_bias_selected_remaining_tokens = 0
        self._prefill_bias_score_time_ns = 0
        self._prefill_bias_starvation_overrides = 0
        # VLLM_PREFILL_BIAS_PHASE6_PATCH: accepted-output timing is keyed by
        # request id and guarded by Request object identity to prevent stale async
        # output or request-id reuse from reusing an old decode timestamp.
        self._decode_timing: dict[str, DecodeTimingState] = {}
        self._last_accepted_output_ts: dict[str, float] = {}
        self._scheduled_batch_timing: dict[int, ScheduledBatchTiming] = {}
        self._prefill_bias_step_time_estimator = PrefillStepTimeEstimator(
            ewma_alpha=self.scheduler_config.prefill_bias_step_time_ewma_alpha,
        )
        self._prefill_bias_tbt_guard_checks = 0
        self._prefill_bias_tbt_guard_allowed = 0
        self._prefill_bias_tbt_guard_blocked = 0
        self._prefill_bias_tbt_guard_unknown_decode = 0
        self._prefill_bias_tbt_already_late = 0
        self._prefill_bias_step_time_samples = 0
        self._prefill_bias_step_time_ewma_s = 0.0
        self._prefill_bias_predicted_step_time_s = 0.0
        self._prefill_bias_min_tbt_slack_s = 0.0
        self._prefill_bias_batch_timing_missing = 0
        self._prefill_bias_batch_timing_entries = 0
        self._prefill_bias_last_tbt_allowed: bool | None = None
        self._prefill_bias_last_tbt_snapshot = TBTGuardSnapshot(
            active_decode_count=0,
            known_decode_count=0,
            unknown_decode_count=0,
            oldest_output_gap_s=0.0,
            minimum_tbt_slack_s=0.0,
            predicted_step_time_s=0.0,
            safety_margin_s=0.0,
            allowed=True,
            reason="guard_disabled",
        )
        # VLLM_PREFILL_BIAS_PHASE3_PATCH: slot-swap state and counters.
        self._prefill_bias_phase3_preemption_count: dict[str, int] = {}
        self._prefill_bias_phase3_last_preempted_ts: dict[str, float] = {}
        self._prefill_bias_phase3_last_resumed_ts: dict[str, float] = {}
        self._prefill_bias_phase3_candidate_backoff_until: dict[str, float] = {}
        self._prefill_slot_swap_attempts_total = 0
        self._prefill_slot_swaps_total = 0
        self._prefill_slot_swap_commit_failures_total = 0
        self._prefill_slot_swap_candidates_rejected_total: dict[str, int] = {}
        self._prefill_slot_swap_victims_rejected_total: dict[str, int] = {}
        self._prefill_slot_swap_preemptions_total = 0
        self._prefill_slot_swap_candidate_admissions_total = 0
        self._prefill_slot_swap_last_result = PrefillAdmissionSwapResult(
            performed=False,
            candidate_request_id=None,
            victim_request_id=None,
            blocker_reason=SlotSwapBlocker.NO_BLOCKER,
            reject_reason=SlotSwapRejectReason.FEATURE_DISABLED,
            predicted_ttft_slack_s=None,
            predicted_victim_tbt_s=None,
            candidate_remaining_tokens=None,
            victim_recompute_tokens=None,
        )
        # VLLM_PREFILL_BIAS_PHASE4_PATCH: bounded adaptive-controller state.
        self.adaptive_prefill_controller = AdaptivePrefillController(
            self.scheduler_config,
            monotonic_clock=self._prefill_bias_monotonic_clock,
        )
        self._adaptive_prefill_policy = (
            self.adaptive_prefill_controller.current_policy()
        )
        self._adaptive_prefill_policy_updates = 0
        self._adaptive_prefill_current_level = 0
        self._adaptive_prefill_last_reason = "cold_start"
        # VLLM_PREFILL_BIAS_PHASE7_PATCH: bounded TTFT-deadline metrics.
        self._prefill_bias_ttft_deadline_checks = 0
        self._prefill_bias_ttft_urgent_activations = 0
        self._prefill_bias_ttft_min_slack_s = float("inf")
        self._prefill_bias_ttft_predicted_completion_s = 0.0
        self._prefill_bias_ttft_candidate_scan_time_ns = 0
        self._prefill_bias_safe_swaps_total = 0
        self._prefill_bias_forced_tbt_overrides_total = 0
        self._prefill_bias_forced_preemptions_total = 0
        self._prefill_bias_forced_unknown_timing_blocked_total = 0
        self._prefill_bias_forced_kv_preflight_blocked_total = 0
        self._prefill_bias_forced_preemption_limit_blocked_total = 0
        self._prefill_bias_predictive_decisions_total = 0
        self._prefill_bias_predictive_would_activate_total = 0
        self._prefill_bias_predictive_fallback_to_legacy_total = 0
        self._prefill_bias_predictive_shadow_overlap_total = 0
        self._prefill_bias_predictive_chosen_reserve_tokens = 0
        self._prefill_bias_predictive_chosen_request_count = 0
        self._prefill_bias_predictive_predicted_step_s = 0.0
        self._prefill_bias_predictive_last_batch_predicted_step_s = 0.0
        self._prefill_bias_predictive_actual_to_predicted_ratio = 0.0
        self._prefill_bias_predictive_underprediction_total = 0
        self._prefill_bias_last_policy_mode = "legacy"
        # VLLM_PREFILL_BIAS_PHASE9_PATCH: global prefill batch-budget metrics.
        self._prefill_batch_budget_steps_total = 0
        self._prefill_batch_budget_planned_tokens_total = 0
        self._prefill_batch_budget_actual_tokens_total = 0
        self._prefill_batch_budget_running_tokens_total = 0
        self._prefill_batch_budget_waiting_tokens_total = 0
        self._prefill_batch_budget_shadow_token_diff_total = 0
        self._prefill_batch_budget_shadow_request_diff_total = 0
        self._prefill_batch_budget_last_utilization = 0.0
        self._prefill_batch_budget_last_actual_tokens = 0
        self._prefill_batch_budget_last_running_tokens = 0
        self._prefill_batch_budget_last_waiting_tokens = 0
        self._prefill_batch_budget_last_selected_ids: tuple[str, ...] = ()
        self._prefill_batch_budget_last_caps: tuple[tuple[str, int], ...] = ()
        self._prefill_batch_budget_last_decision = PrefillBatchBudgetDecision(
            total_prefill_budget=0,
            request_token_caps=(),
            running_prefill_order=(),
            waiting_prefill_order=(),
            predicted_step_time_s=0.0,
            minimum_tbt_slack_s=math.inf,
            reason="batch_budget_inactive",
        )
        self._prefill_bias_last_decision = PrefillBiasDecision(
            active=False,
            reserve_tokens=0,
            candidate_request_ids=(),
            reason="disabled",
        )

    def _adaptive_prefill_enabled(self) -> bool:
        return bool(
            getattr(self.scheduler_config, "adaptive_prefill_controller_enabled", False)
        )

    def _adaptive_prefill_effective_config(self):
        if not self._adaptive_prefill_enabled():
            return self.scheduler_config
        return self._adaptive_prefill_policy.as_config(self.scheduler_config)

    def _adaptive_prefill_refresh_policy(self) -> None:
        if not self._adaptive_prefill_enabled():
            self.prefill_bias_controller.scheduler_config = self.scheduler_config
            return
        now = self._prefill_bias_monotonic_clock()
        policy = self.adaptive_prefill_controller.maybe_update(now)
        self._adaptive_prefill_policy = policy
        self._adaptive_prefill_policy_updates = (
            self.adaptive_prefill_controller.updates_total
        )
        self._adaptive_prefill_current_level = policy.level
        self._adaptive_prefill_last_reason = policy.reason
        self.prefill_bias_controller.scheduler_config = (
            self._adaptive_prefill_effective_config()
        )

    def _adaptive_prefill_observe_pressure(self, token_budget: int) -> None:
        if not self._adaptive_prefill_enabled():
            return
        now_wall = time.time()
        waiting_prefill_count = 0
        oldest_waiting_prefill_age_s = 0.0
        for request in self.waiting:
            if (
                request.status == RequestStatus.WAITING
                and not request.is_prefill_chunk
                and not request.abort_immediately
            ):
                waiting_prefill_count += 1
                oldest_waiting_prefill_age_s = max(
                    oldest_waiting_prefill_age_s,
                    max(0.0, now_wall - float(request.arrival_time or 0.0)),
                )
        recompute_tokens = (
            self._prefill_slot_swap_last_result.victim_recompute_tokens or 0
        )
        swap_failures = (
            self._prefill_slot_swap_commit_failures_total
            + sum(self._prefill_slot_swap_candidates_rejected_total.values())
        )
        self.adaptive_prefill_controller.observe_pressure(
            now_monotonic=self._prefill_bias_monotonic_clock(),
            waiting_prefill_count=waiting_prefill_count,
            oldest_waiting_prefill_age_s=oldest_waiting_prefill_age_s,
            active_decode_count=len(self._prefill_bias_active_decode_request_ids()),
            running_count=len(self.running),
            max_running_count=self.max_num_running_reqs,
            token_budget=token_budget,
            max_token_budget=self.max_num_scheduled_tokens,
            kv_cache_usage=float(self.kv_cache_manager.usage),
            swap_attempts=self._prefill_slot_swap_attempts_total,
            swap_failures=swap_failures,
            recompute_tokens=recompute_tokens,
        )

    def _adaptive_prefill_observe_accepted_tokens(
        self,
        request: Request,
        num_new_tokens: int,
        *,
        now_monotonic: float,
    ) -> None:
        if not self._adaptive_prefill_enabled() or num_new_tokens <= 0:
            return
        self.adaptive_prefill_controller.observe_accepted_tokens(
            request_id=request.request_id,
            arrival_time=float(request.arrival_time or 0.0),
            num_new_tokens=num_new_tokens,
            now_wall=time.time(),
            now_monotonic=now_monotonic,
            ttft_slo_s=self.scheduler_config.adaptive_prefill_ttft_slo_s,
            tbt_slo_s=self.scheduler_config.adaptive_prefill_tbt_slo_s,
        )

    def _adaptive_prefill_observe_request_finished(self, request: Request) -> None:
        if not self._adaptive_prefill_enabled():
            return
        finished_ok = request.status in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_REPETITION,
        )
        self.adaptive_prefill_controller.observe_request_finished(
            request_id=request.request_id,
            finished_ok=finished_ok,
            now_monotonic=self._prefill_bias_monotonic_clock(),
        )

    def _prefill_bias_request_is_schedulable(self, request: Request) -> bool:
        return not self._is_blocked_waiting_status(request.status)

    def _prefill_bias_decode_floor(self, token_budget: int) -> int:
        active_decode_requests = [
            req for req in self.running if not req.is_prefill_chunk
        ]
        tokens_per_decode = 1 + self.num_spec_tokens if self.num_spec_tokens > 0 else 1
        return min(token_budget, len(active_decode_requests) * tokens_per_decode)

    # VLLM_PREFILL_BIAS_PHASE6_PATCH: accepted-output timing helpers.
    def _clear_decode_timing(self, request_id: str) -> None:
        self._decode_timing.pop(request_id, None)
        self._last_accepted_output_ts.pop(request_id, None)

    def _record_accepted_decode_output(
        self,
        request: Request,
        accepted_token_ids: list[int],
        now_monotonic_s: float | None = None,
    ) -> None:
        if not accepted_token_ids:
            return
        request_id = request.request_id
        if self.requests.get(request_id) is not request or request.is_finished():
            return

        timestamp_s = (
            self._prefill_bias_monotonic_clock()
            if now_monotonic_s is None
            else now_monotonic_s
        )
        previous = self._decode_timing.get(request_id)
        accepted_events = 1
        if previous is not None and previous.request_identity == id(request):
            accepted_events = previous.accepted_output_events + 1
        self._decode_timing[request_id] = DecodeTimingState(
            request_identity=id(request),
            last_accepted_output_monotonic_s=timestamp_s,
            accepted_output_events=accepted_events,
        )
        self._last_accepted_output_ts[request_id] = timestamp_s

    def _prefill_bias_last_output_timestamps(self) -> dict[str, float]:
        timestamps: dict[str, float] = {}
        stale_request_ids: list[str] = []
        for request_id, state in self._decode_timing.items():
            request = self.requests.get(request_id)
            if request is None or id(request) != state.request_identity:
                stale_request_ids.append(request_id)
                continue
            timestamps[request_id] = state.last_accepted_output_monotonic_s
        for request_id in stale_request_ids:
            self._clear_decode_timing(request_id)
        return timestamps

    def _prefill_bias_last_output_timestamp(
        self,
        request: Request,
    ) -> float | None:
        state = self._decode_timing.get(request.request_id)
        if state is None or state.request_identity != id(request):
            return None
        return state.last_accepted_output_monotonic_s

    def _is_active_decode_request(self, request: Request) -> bool:
        return (
            request.status == RequestStatus.RUNNING
            and not request.is_finished()
            and request.num_output_tokens > 0
            and not request.is_prefill_chunk
        )

    def _prefill_bias_active_decode_request_ids(self) -> tuple[str, ...]:
        return tuple(
            request.request_id
            for request in self.running
            if self._is_active_decode_request(request)
        )

    def _prefill_bias_estimated_step_time(self) -> float:
        config = self._adaptive_prefill_effective_config()
        return self._prefill_bias_step_time_estimator.estimate(
            initial_step_time_s=config.prefill_bias_initial_step_time_s,
            headroom_factor=config.prefill_bias_step_time_headroom_factor,
            min_samples=config.prefill_bias_step_observation_min_samples,
        )

    def _prefill_bias_apply_tbt_guard(
        self,
        decision: PrefillBiasDecision,
    ) -> PrefillBiasDecision:
        config = self._adaptive_prefill_effective_config()
        if not decision.active or not config.prefill_bias_tbt_guard_enabled:
            return decision

        snapshot = self.prefill_bias_controller.evaluate_tbt_guard(
            now_monotonic=self._prefill_bias_monotonic_clock(),
            active_decode_request_ids=self._prefill_bias_active_decode_request_ids(),
            last_output_ts=self._prefill_bias_last_output_timestamps(),
            tbt_slo_s=config.prefill_bias_tbt_slo_s,
            predicted_step_time_s=(
                decision.predicted_step_time_s
                if decision.predicted_step_time_s > 0.0
                else self._prefill_bias_estimated_step_time()
            ),
            safety_margin_s=config.prefill_bias_tbt_safety_margin_s,
            guard_unknown_decode=config.prefill_bias_guard_unknown_decode,
        )
        self._prefill_bias_last_tbt_snapshot = snapshot
        self._prefill_bias_tbt_guard_checks += 1
        self._prefill_bias_predicted_step_time_s = snapshot.predicted_step_time_s
        self._prefill_bias_min_tbt_slack_s = snapshot.minimum_tbt_slack_s
        if snapshot.unknown_decode_count:
            self._prefill_bias_tbt_guard_unknown_decode += 1
        if snapshot.reason == "tbt_already_late":
            self._prefill_bias_tbt_already_late += 1
        if snapshot.allowed:
            self._prefill_bias_tbt_guard_allowed += 1
        else:
            self._prefill_bias_tbt_guard_blocked += 1

        if self._prefill_bias_last_tbt_allowed is not snapshot.allowed:
            logger.debug(
                "Prefill bias TBT guard state changed: allowed=%s reason=%s "
                "active_decodes=%d known_decodes=%d unknown_decodes=%d "
                "min_slack_s=%.6f predicted_step_s=%.6f margin_s=%.6f",
                snapshot.allowed,
                snapshot.reason,
                snapshot.active_decode_count,
                snapshot.known_decode_count,
                snapshot.unknown_decode_count,
                snapshot.minimum_tbt_slack_s,
                snapshot.predicted_step_time_s,
                snapshot.safety_margin_s,
            )
        self._prefill_bias_last_tbt_allowed = snapshot.allowed

        if snapshot.allowed:
            return decision

        if (
            config.prefill_bias_ttft_deadline_enabled
            and config.prefill_bias_ttft_force_preempt_enabled
            and decision.forced_ttft
        ):
            self._prefill_bias_forced_tbt_overrides_total += 1
            logger.debug(
                "TTFT deadline overrides TBT bias guard: request_ids=%s "
                "ttft_slack_s=%.6f guard_reason=%s",
                decision.candidate_request_ids,
                decision.minimum_ttft_slack_s,
                snapshot.reason,
            )
            return decision

        logger.debug(
            "Prefill bias blocked by TBT guard: reason=%s active_decodes=%d "
            "min_slack_s=%.6f predicted_step_s=%.6f margin_s=%.6f",
            snapshot.reason,
            snapshot.active_decode_count,
            snapshot.minimum_tbt_slack_s,
            snapshot.predicted_step_time_s,
            snapshot.safety_margin_s,
        )
        return PrefillBiasDecision(
            active=False,
            reserve_tokens=0,
            candidate_request_ids=(),
            reason=f"tbt_guard_{snapshot.reason}",
        )

    def _prefill_bias_track_scheduled_batch(
        self,
        scheduler_output: SchedulerOutput,
        prefill_tokens: int,
        active_decode_count: int,
    ) -> None:
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return
        self._scheduled_batch_timing[id(scheduler_output)] = ScheduledBatchTiming(
            started_at=self._prefill_bias_monotonic_clock(),
            prefill_tokens=prefill_tokens,
            total_tokens=scheduler_output.total_num_scheduled_tokens,
            active_decode_count=active_decode_count,
        )
        self._prefill_bias_batch_timing_entries = len(self._scheduled_batch_timing)

    def _prefill_bias_observe_batch_timing(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return
        timing = self._scheduled_batch_timing.pop(id(scheduler_output), None)
        self._prefill_bias_batch_timing_entries = len(self._scheduled_batch_timing)
        if timing is None:
            self._prefill_bias_batch_timing_missing += 1
            logger.debug(
                "Prefill bias batch timing metadata missing for scheduled output"
            )
            return
        duration_s = self._prefill_bias_monotonic_clock() - timing.started_at
        if timing.prefill_tokens > 0:
            self._prefill_bias_step_time_estimator.observe(duration_s)
            self._prefill_bias_step_time_samples = (
                self._prefill_bias_step_time_estimator.num_samples
            )
            self._prefill_bias_step_time_ewma_s = (
                self._prefill_bias_step_time_estimator.ewma_step_time_s or 0.0
            )
        prediction_s = self.prefill_bias_controller.predictive._predict_step(
            prefill_tokens=timing.prefill_tokens,
            active_decode_count=timing.active_decode_count,
        )
        self._prefill_bias_predictive_last_batch_predicted_step_s = prediction_s
        self.prefill_bias_controller.observe_batch(
            duration_s=duration_s,
            prefill_tokens=timing.prefill_tokens,
            active_decode_count=timing.active_decode_count,
        )
        if prediction_s > 0.0:
            ratio = duration_s / prediction_s
            self._prefill_bias_predictive_actual_to_predicted_ratio = ratio
            if ratio > 1.0:
                self._prefill_bias_predictive_underprediction_total += 1

    def _prefill_slot_swap_reject_candidate(
        self,
        reason: SlotSwapRejectReason,
    ) -> None:
        key = reason.value
        self._prefill_slot_swap_candidates_rejected_total[key] = (
            self._prefill_slot_swap_candidates_rejected_total.get(key, 0) + 1
        )

    def _prefill_slot_swap_reject_victim(
        self,
        reason: SlotSwapRejectReason,
    ) -> None:
        key = reason.value
        self._prefill_slot_swap_victims_rejected_total[key] = (
            self._prefill_slot_swap_victims_rejected_total.get(key, 0) + 1
        )

    def _prefill_slot_swap_result(
        self,
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
        result = PrefillAdmissionSwapResult(
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
        self._prefill_slot_swap_last_result = result
        return result

    def _prefill_slot_swap_candidate_from_decision(
        self,
        decision: PrefillBiasDecision,
    ) -> Request | None:
        if not decision.active or not decision.candidate_request_ids:
            return None
        selected = set(decision.candidate_request_ids)
        for request in self.waiting:
            if request.request_id in selected:
                return request
        return None

    def _prefill_slot_swap_candidate_remaining(
        self,
        candidate: Request,
    ) -> tuple[int, int]:
        config = self._adaptive_prefill_effective_config()
        cached_tokens = max(0, self.kv_cache_manager.peek_num_computed_tokens(candidate))
        effective_cached_tokens = cached_tokens
        if cached_tokens < config.prefill_bias_min_cached_tokens:
            effective_cached_tokens = 0
        effective_computed = max(
            int(candidate.num_computed_tokens),
            effective_cached_tokens,
        )
        remaining_tokens = max(
            int(candidate.num_prompt_tokens) - effective_computed,
            1,
        )
        return remaining_tokens, cached_tokens

    def _prefill_slot_swap_candidate_blocker(
        self,
        candidate: Request,
        *,
        token_budget: int,
    ) -> SlotSwapBlocker:
        if token_budget <= 0:
            return SlotSwapBlocker.TOKEN_BUDGET
        if candidate.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            return SlotSwapBlocker.REMOTE_KV
        if candidate.status != RequestStatus.WAITING or candidate.is_finished():
            return SlotSwapBlocker.UNSUPPORTED_STATE
        if candidate.abort_immediately:
            return SlotSwapBlocker.UNSUPPORTED_STATE
        if candidate.num_output_placeholders > 0:
            return SlotSwapBlocker.ASYNC_IN_FLIGHT
        if candidate.has_encoder_inputs:
            # Safe subset: encoder budget is not preflighted transactionally.
            return SlotSwapBlocker.ENCODER_BUDGET
        if len(self.running) < self.max_num_running_reqs:
            return SlotSwapBlocker.NO_BLOCKER
        return SlotSwapBlocker.MAX_NUM_SEQS

    def _prefill_slot_swap_force_candidate_first(self, candidate_id: str) -> None:
        selected = None
        rest = []
        for request in list(self.waiting):
            if request.request_id == candidate_id and selected is None:
                selected = request
            else:
                rest.append(request)
        if selected is None:
            return
        self.waiting.clear()
        self.waiting.extend([selected, *rest])

    def _prefill_slot_swap_count_blocks(self, request: Request) -> int | None:
        try:
            block_ids = self.kv_cache_manager.get_blocks(
                request.request_id
            ).get_block_ids(allow_none=True)
        except Exception:
            return None
        if block_ids is None:
            return 0
        # Conservative narrow support: only trust single-group accounting. Hybrid
        # groups can use different physical pools, so Phase 3 fails closed there.
        if len(block_ids) != 1:
            return None
        return len([block_id for block_id in block_ids[0] if block_id is not None])

    def _prefill_slot_swap_kv_preflight(
        self,
        *,
        candidate: Request,
        victim: Request,
        candidate_remaining_tokens: int,
    ) -> bool:
        try:
            block_size = int(self.cache_config.block_size)
            free_blocks = int(self.kv_cache_manager.block_pool.get_num_free_blocks())
            watermark_blocks = int(self.kv_cache_manager.watermark_blocks)
        except Exception:
            return False
        victim_blocks = self._prefill_slot_swap_count_blocks(victim)
        if victim_blocks is None or block_size <= 0:
            return False
        required_blocks = (
            candidate_remaining_tokens + self.num_lookahead_tokens + block_size - 1
        ) // block_size
        # Waiting admissions apply the watermark when other requests are already
        # scheduled/running. Keep the same headroom in this read-only preflight.
        return required_blocks + watermark_blocks <= free_blocks + victim_blocks

    def _prefill_slot_swap_select_victim(
        self,
        *,
        candidate_id: str,
        predicted_step_time_s: float,
    ) -> tuple[Request | None, float | None, int | None]:
        config = self._adaptive_prefill_effective_config()
        now = self._prefill_bias_monotonic_clock()
        tbt_slo_s = config.prefill_bias_tbt_slo_s
        safety_margin_s = config.prefill_bias_tbt_safety_margin_s
        margin_s = config.prefill_bias_victim_tbt_margin_s
        max_recompute = config.prefill_bias_max_victim_recompute_tokens
        best: tuple[tuple[float, float, int, int, str], Request, float, int] | None = None
        for position, request in enumerate(self.running):
            if request.request_id == candidate_id:
                continue
            if not self._is_active_decode_request(request):
                continue
            if request.num_output_placeholders > 0:
                self._prefill_slot_swap_reject_victim(
                    SlotSwapRejectReason.VICTIM_ASYNC_IN_FLIGHT
                )
                continue
            if request.has_encoder_inputs:
                continue
            last_output_ts = self._prefill_bias_last_output_timestamp(request)
            if last_output_ts is None:
                continue
            elapsed_s = now - last_output_ts
            if elapsed_s < 0.0 or elapsed_s >= tbt_slo_s:
                continue
            cooldown_until = self._prefill_bias_phase3_last_preempted_ts.get(
                request.request_id,
                -1.0,
            ) + config.prefill_bias_swap_cooldown_s
            resume_cooldown_until = self._prefill_bias_phase3_last_resumed_ts.get(
                request.request_id,
                -1.0,
            ) + config.prefill_bias_swap_cooldown_s
            if now < max(cooldown_until, resume_cooldown_until):
                self._prefill_slot_swap_reject_victim(
                    SlotSwapRejectReason.VICTIM_COOLDOWN
                )
                continue
            preemption_count = self._prefill_bias_phase3_preemption_count.get(
                request.request_id,
                0,
            )
            if (
                preemption_count
                >= config.prefill_bias_max_preemptions_per_request
            ):
                self._prefill_slot_swap_reject_victim(
                    SlotSwapRejectReason.VICTIM_PREEMPTION_LIMIT
                )
                continue
            recompute_tokens = request.num_tokens
            if max_recompute is not None and recompute_tokens > max_recompute:
                self._prefill_slot_swap_reject_victim(
                    SlotSwapRejectReason.VICTIM_RECOMPUTE_TOO_LARGE
                )
                continue
            victim_resume_cost_s = (
                predicted_step_time_s
                + config.prefill_bias_initial_step_time_s
                + safety_margin_s
            )
            projected_tbt_s = elapsed_s + victim_resume_cost_s
            tbt_limit = tbt_slo_s - margin_s
            if not self.prefill_bias_controller.is_safe_projected_tbt(
                projected_tbt_s=projected_tbt_s,
                tbt_limit_s=tbt_limit,
            ):
                continue
            projected_slack_s = tbt_limit - projected_tbt_s
            score = (
                float(recompute_tokens),
                -projected_slack_s,
                preemption_count,
                position,
                request.request_id,
            )
            if best is None or score < best[0]:
                best = (score, request, projected_tbt_s, recompute_tokens)
        if best is None:
            return None, None, None
        _, victim, projected_tbt_s, recompute_tokens = best
        return victim, projected_tbt_s, recompute_tokens

    def _prefill_slot_swap_select_forced_victim(
        self,
        *,
        candidate_id: str,
        predicted_step_time_s: float,
    ) -> tuple[Request | None, float | None, int | None]:
        config = self._adaptive_prefill_effective_config()
        now = self._prefill_bias_monotonic_clock()
        best: tuple[tuple[float, int, int, int, str], Request, float, int] | None = None
        for position, request in enumerate(self.running):
            if request.request_id == candidate_id:
                continue
            if not self._is_active_decode_request(request):
                continue
            if request.num_output_placeholders > 0 or request.has_encoder_inputs:
                continue
            last_output_ts = self._prefill_bias_last_output_timestamp(request)
            if last_output_ts is None:
                self._prefill_bias_forced_unknown_timing_blocked_total += 1
                continue
            elapsed_s = now - last_output_ts
            if elapsed_s < 0.0:
                self._prefill_bias_forced_unknown_timing_blocked_total += 1
                continue
            cooldown_until = self._prefill_bias_phase3_last_preempted_ts.get(
                request.request_id,
                -1.0,
            ) + config.prefill_bias_swap_cooldown_s
            resume_cooldown_until = self._prefill_bias_phase3_last_resumed_ts.get(
                request.request_id,
                -1.0,
            ) + config.prefill_bias_swap_cooldown_s
            if now < max(cooldown_until, resume_cooldown_until):
                continue
            preemption_count = self._prefill_bias_phase3_preemption_count.get(
                request.request_id,
                0,
            )
            if preemption_count >= config.prefill_bias_max_preemptions_per_request:
                self._prefill_bias_forced_preemption_limit_blocked_total += 1
                continue
            recompute_tokens = int(request.num_tokens)
            max_recompute = config.prefill_bias_max_victim_recompute_tokens
            if max_recompute is not None and recompute_tokens > max_recompute:
                continue
            victim_resume_cost_s = (
                predicted_step_time_s
                + config.prefill_bias_initial_step_time_s
                + config.prefill_bias_tbt_safety_margin_s
            )
            projected_tbt_s = elapsed_s + victim_resume_cost_s
            tbt_slack_s = config.prefill_bias_tbt_slo_s - elapsed_s
            score = self.prefill_bias_controller.forced_victim_sort_key(
                tbt_slack_s=tbt_slack_s,
                recompute_tokens=recompute_tokens,
                preemption_count=preemption_count,
                position=position,
                request_id=request.request_id,
            )
            if best is None or score < best[0]:
                best = (score, request, projected_tbt_s, recompute_tokens)
        if best is None:
            return None, None, None
        _, victim, projected_tbt_s, recompute_tokens = best
        return victim, projected_tbt_s, recompute_tokens

    def _try_prefill_admission_swap(
        self,
        *,
        decision: PrefillBiasDecision,
        token_budget: int,
        scheduled_timestamp: float,
        scheduled_running_reqs: list[Request],
        preempted_reqs: list[Request],
        req_to_new_blocks: dict[str, KVCacheBlocks],
        num_scheduled_tokens: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]],
        scheduled_encoder_inputs: dict[str, list[int]],
        swaps_this_step: int,
        prefill_budget_accountant: PrefillBudgetAccountant | None = None,
    ) -> tuple[PrefillAdmissionSwapResult, int]:
        config = self._adaptive_prefill_effective_config()
        if not config.prefill_bias_slot_swap_enabled:
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.FEATURE_DISABLED,
            ), token_budget
        self._prefill_slot_swap_attempts_total += 1
        if self.policy != SchedulingPolicy.FCFS or self._pause_state != PauseState.UNPAUSED:
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.FEATURE_DISABLED,
            ), token_budget
        if swaps_this_step >= config.prefill_bias_max_swaps_per_step:
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.FEATURE_DISABLED,
            ), token_budget
        if preempted_reqs:
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.BLOCKER_NOT_MAX_NUM_SEQS,
            ), token_budget

        candidate = self._prefill_slot_swap_candidate_from_decision(decision)
        if (
            decision.policy_mode in {"predictive", "batch-budget"}
            and not decision.slot_swap_eligible
        ):
            self._prefill_slot_swap_reject_candidate(
                SlotSwapRejectReason.CANDIDATE_NOT_URGENT
            )
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.CANDIDATE_NOT_URGENT,
            ), token_budget
        if candidate is None:
            self._prefill_slot_swap_reject_candidate(SlotSwapRejectReason.NO_CANDIDATE)
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.NO_CANDIDATE,
            ), token_budget
        candidate_id = candidate.request_id
        blocker = self._prefill_slot_swap_candidate_blocker(
            candidate,
            token_budget=token_budget,
        )
        if blocker != SlotSwapBlocker.MAX_NUM_SEQS:
            reject = (
                SlotSwapRejectReason.TOKEN_BUDGET
                if blocker == SlotSwapBlocker.TOKEN_BUDGET
                else SlotSwapRejectReason.BLOCKER_NOT_MAX_NUM_SEQS
            )
            self._prefill_slot_swap_reject_candidate(reject)
            return self._prefill_slot_swap_result(
                candidate_request_id=candidate_id,
                blocker_reason=blocker,
                reject_reason=reject,
            ), token_budget

        now_monotonic = self._prefill_bias_monotonic_clock()
        if now_monotonic < self._prefill_bias_phase3_candidate_backoff_until.get(
            candidate_id,
            -1.0,
        ):
            self._prefill_slot_swap_reject_candidate(
                SlotSwapRejectReason.CANDIDATE_NOT_URGENT
            )
            return self._prefill_slot_swap_result(
                candidate_request_id=candidate_id,
                blocker_reason=SlotSwapBlocker.COOLDOWN,
                reject_reason=SlotSwapRejectReason.CANDIDATE_NOT_URGENT,
            ), token_budget

        try:
            candidate_remaining_tokens, cached_tokens = (
                self._prefill_slot_swap_candidate_remaining(candidate)
            )
            predicted_step_time_s = (
                decision.predicted_step_time_s
                if decision.predicted_step_time_s > 0.0
                else self._prefill_bias_estimated_step_time()
            )
            if config.prefill_bias_ttft_deadline_enabled:
                should_swap = decision.active
                ttft_slack_s = decision.minimum_ttft_slack_s
            else:
                should_swap, ttft_slack_s = (
                    self.prefill_bias_controller.should_try_slot_swap(
                        now_wall=time.time(),
                        arrival_time=candidate.arrival_time,
                        ttft_slo_s=config.prefill_bias_ttft_slo_s,
                        predicted_prefill_s=predicted_step_time_s,
                        slack_threshold_s=config.prefill_bias_swap_slack_threshold_s,
                    )
                )
        except Exception:
            self._prefill_slot_swap_reject_candidate(
                SlotSwapRejectReason.CANDIDATE_NOT_URGENT
            )
            return self._prefill_slot_swap_result(
                candidate_request_id=candidate_id,
                blocker_reason=blocker,
                reject_reason=SlotSwapRejectReason.CANDIDATE_NOT_URGENT,
            ), token_budget

        if not should_swap:
            self._prefill_slot_swap_reject_candidate(
                SlotSwapRejectReason.CANDIDATE_NOT_URGENT
            )
            return self._prefill_slot_swap_result(
                candidate_request_id=candidate_id,
                blocker_reason=blocker,
                reject_reason=SlotSwapRejectReason.CANDIDATE_NOT_URGENT,
                predicted_ttft_slack_s=ttft_slack_s,
                candidate_remaining_tokens=candidate_remaining_tokens,
            ), token_budget
        if (
            candidate_remaining_tokens
            > config.prefill_bias_max_candidate_remaining_tokens
        ):
            self._prefill_slot_swap_reject_candidate(
                SlotSwapRejectReason.KV_PREFLIGHT_FAILED
            )
            return self._prefill_slot_swap_result(
                candidate_request_id=candidate_id,
                blocker_reason=blocker,
                reject_reason=SlotSwapRejectReason.KV_PREFLIGHT_FAILED,
                predicted_ttft_slack_s=ttft_slack_s,
                candidate_remaining_tokens=candidate_remaining_tokens,
            ), token_budget
        if config.prefill_bias_require_cache_residency and cached_tokens <= 0:
            self._prefill_slot_swap_reject_candidate(
                SlotSwapRejectReason.KV_PREFLIGHT_FAILED
            )
            return self._prefill_slot_swap_result(
                candidate_request_id=candidate_id,
                blocker_reason=blocker,
                reject_reason=SlotSwapRejectReason.KV_PREFLIGHT_FAILED,
                predicted_ttft_slack_s=ttft_slack_s,
                candidate_remaining_tokens=candidate_remaining_tokens,
            ), token_budget

        guard = self.prefill_bias_controller.evaluate_tbt_guard(
            now_monotonic=now_monotonic,
            active_decode_request_ids=self._prefill_bias_active_decode_request_ids(),
            last_output_ts=self._prefill_bias_last_output_timestamps(),
            tbt_slo_s=config.prefill_bias_tbt_slo_s,
            predicted_step_time_s=predicted_step_time_s,
            safety_margin_s=config.prefill_bias_tbt_safety_margin_s,
            guard_unknown_decode=config.prefill_bias_guard_unknown_decode,
        )
        force_override = (
            config.prefill_bias_ttft_force_preempt_enabled
            and decision.forced_ttft
        )
        if not guard.allowed and not force_override:
            self._prefill_slot_swap_reject_candidate(SlotSwapRejectReason.TBT_GUARD)
            return self._prefill_slot_swap_result(
                candidate_request_id=candidate_id,
                blocker_reason=SlotSwapBlocker.TBT_GUARD,
                reject_reason=SlotSwapRejectReason.TBT_GUARD,
                predicted_ttft_slack_s=ttft_slack_s,
                candidate_remaining_tokens=candidate_remaining_tokens,
            ), token_budget

        forced_swap = False
        victim, projected_tbt_s, recompute_tokens = self._prefill_slot_swap_select_victim(
            candidate_id=candidate_id,
            predicted_step_time_s=predicted_step_time_s,
        )
        if (
            victim is None
            and force_override
            and decision.forced_ttft
        ):
            victim, projected_tbt_s, recompute_tokens = (
                self._prefill_slot_swap_select_forced_victim(
                    candidate_id=candidate_id,
                    predicted_step_time_s=predicted_step_time_s,
                )
            )
            forced_swap = victim is not None
        if victim is None or projected_tbt_s is None or recompute_tokens is None:
            self._prefill_slot_swap_reject_candidate(SlotSwapRejectReason.NO_SAFE_VICTIM)
            return self._prefill_slot_swap_result(
                candidate_request_id=candidate_id,
                blocker_reason=blocker,
                reject_reason=SlotSwapRejectReason.NO_SAFE_VICTIM,
                predicted_ttft_slack_s=ttft_slack_s,
                candidate_remaining_tokens=candidate_remaining_tokens,
            ), token_budget
        if not self._prefill_slot_swap_kv_preflight(
            candidate=candidate,
            victim=victim,
            candidate_remaining_tokens=candidate_remaining_tokens,
        ):
            if forced_swap:
                self._prefill_bias_forced_kv_preflight_blocked_total += 1
            self._prefill_slot_swap_reject_candidate(
                SlotSwapRejectReason.KV_PREFLIGHT_FAILED
            )
            return self._prefill_slot_swap_result(
                candidate_request_id=candidate_id,
                victim_request_id=victim.request_id,
                blocker_reason=blocker,
                reject_reason=SlotSwapRejectReason.KV_PREFLIGHT_FAILED,
                predicted_ttft_slack_s=ttft_slack_s,
                predicted_victim_tbt_s=projected_tbt_s,
                candidate_remaining_tokens=candidate_remaining_tokens,
                victim_recompute_tokens=recompute_tokens,
            ), token_budget

        victim_id = victim.request_id
        if victim in scheduled_running_reqs:
            scheduled_running_reqs.remove(victim)
            restored_tokens = num_scheduled_tokens.pop(victim_id, 0)
            token_budget += restored_tokens
            if (
                prefill_budget_accountant is not None
                and prefill_budget_accountant.has_cap(victim_id)
            ):
                prefill_budget_accountant.restore(
                    victim_id,
                    restored_tokens,
                    source="running",
                )
            req_to_new_blocks.pop(victim_id, None)
            scheduled_spec_decode_tokens.pop(victim_id, None)
            scheduled_encoder_inputs.pop(victim_id, None)
        self.running.remove(victim)
        self._preempt_request(victim, scheduled_timestamp)
        preempted_reqs.append(victim)
        self._prefill_bias_phase3_preemption_count[victim_id] = (
            self._prefill_bias_phase3_preemption_count.get(victim_id, 0) + 1
        )
        self._prefill_bias_phase3_last_preempted_ts[victim_id] = now_monotonic
        self._prefill_slot_swap_preemptions_total += 1
        self._prefill_slot_swaps_total += 1
        if forced_swap:
            self._prefill_bias_forced_preemptions_total += 1
        else:
            self._prefill_bias_safe_swaps_total += 1
        self._prefill_slot_swap_force_candidate_first(candidate_id)
        logger.debug(
            "Prefill slot swap committed: candidate=%s victim=%s "
            "ttft_slack_s=%.6f projected_victim_tbt_s=%.6f remaining_tokens=%d "
            "victim_recompute_tokens=%d",
            candidate_id,
            victim_id,
            ttft_slack_s,
            projected_tbt_s,
            candidate_remaining_tokens,
            recompute_tokens,
        )
        return self._prefill_slot_swap_result(
            performed=True,
            candidate_request_id=candidate_id,
            victim_request_id=victim_id,
            blocker_reason=blocker,
            predicted_ttft_slack_s=ttft_slack_s,
            predicted_victim_tbt_s=projected_tbt_s,
            candidate_remaining_tokens=candidate_remaining_tokens,
            victim_recompute_tokens=recompute_tokens,
        ), token_budget

    def _prefill_bias_predictive_context(self) -> tuple[int, float, int]:
        active_ids = self._prefill_bias_active_decode_request_ids()
        if not active_ids:
            return 0, math.inf, 0
        now = self._prefill_bias_monotonic_clock()
        timestamps = self._prefill_bias_last_output_timestamps()
        slacks: list[float] = []
        unknown = 0
        tbt_slo_s = self._adaptive_prefill_effective_config().prefill_bias_tbt_slo_s
        for request_id in active_ids:
            timestamp = timestamps.get(request_id)
            if timestamp is None or not math.isfinite(timestamp) or timestamp > now:
                unknown += 1
                continue
            slacks.append(tbt_slo_s - (now - timestamp))
        return len(active_ids), min(slacks) if slacks else math.inf, unknown

    def _prefill_bias_sync_predictive_metrics(
        self,
        decision: PrefillBiasDecision,
    ) -> None:
        router = self.prefill_bias_controller
        self._prefill_bias_last_policy_mode = router.last_applied_mode
        self._prefill_bias_predictive_decisions_total = (
            router.predictive_decisions_total
        )
        self._prefill_bias_predictive_would_activate_total = (
            router.predictive_would_activate_total
        )
        self._prefill_bias_predictive_fallback_to_legacy_total = (
            router.predictive_fallback_to_legacy_total
        )
        self._prefill_bias_predictive_shadow_overlap_total = (
            router.shadow_candidate_overlap_total
        )
        predictive = router.last_predictive_decision
        if predictive is not None:
            self._prefill_bias_predictive_chosen_reserve_tokens = (
                predictive.reserve_tokens
            )
            self._prefill_bias_predictive_chosen_request_count = len(
                predictive.candidate_request_ids
            )
            self._prefill_bias_predictive_predicted_step_s = (
                predictive.predicted_step_time_s
            )
        elif decision.policy_mode == "predictive":
            self._prefill_bias_predictive_chosen_reserve_tokens = decision.reserve_tokens
            self._prefill_bias_predictive_chosen_request_count = len(
                decision.candidate_request_ids
            )

    def get_prefill_bias_predictive_metrics(self) -> dict[str, int | float | str]:
        return {
            "mode": self._prefill_bias_last_policy_mode,
            "decisions_total": self._prefill_bias_predictive_decisions_total,
            "would_activate_total": (
                self._prefill_bias_predictive_would_activate_total
            ),
            "fallback_to_legacy_total": (
                self._prefill_bias_predictive_fallback_to_legacy_total
            ),
            "shadow_candidate_overlap_total": (
                self._prefill_bias_predictive_shadow_overlap_total
            ),
            "chosen_reserve_tokens": (
                self._prefill_bias_predictive_chosen_reserve_tokens
            ),
            "chosen_request_count": (
                self._prefill_bias_predictive_chosen_request_count
            ),
            "predicted_step_s": self._prefill_bias_predictive_predicted_step_s,
            "last_batch_predicted_step_s": (
                self._prefill_bias_predictive_last_batch_predicted_step_s
            ),
            "actual_to_predicted_ratio": (
                self._prefill_bias_predictive_actual_to_predicted_ratio
            ),
            "underprediction_total": (
                self._prefill_bias_predictive_underprediction_total
            ),
        }

    def _prefill_batch_budget_mode(self) -> str | None:
        mode = str(
            getattr(self.scheduler_config, "prefill_bias_policy_mode", "legacy")
        ).lower()
        if mode in {"batch-budget-shadow", "batch-budget"}:
            return mode
        return None

    def _prefill_batch_budget_prepare(
        self,
        *,
        token_budget: int,
    ) -> PrefillBatchBudgetDecision:
        active_decode_count, minimum_tbt_slack_s, unknown_decode_count = (
            self._prefill_bias_predictive_context()
        )
        decode_floor = self._prefill_bias_decode_floor(token_budget)
        decision = self.prefill_bias_controller.plan_batch(
            running_prefills=self.running,
            waiting_prefills=self.waiting,
            max_safe_prefill_tokens=max(0, token_budget - decode_floor),
            active_decode_count=active_decode_count,
            minimum_tbt_slack_s=minimum_tbt_slack_s,
            unknown_decode_count=unknown_decode_count,
            peek_cached_tokens=self.kv_cache_manager.peek_num_computed_tokens,
            is_request_schedulable=self._prefill_bias_request_is_schedulable,
        )
        self._prefill_batch_budget_steps_total += 1
        self._prefill_batch_budget_planned_tokens_total += (
            decision.total_prefill_budget
        )
        self._prefill_batch_budget_last_selected_ids = (
            *decision.running_prefill_order,
            *decision.waiting_prefill_order,
        )
        self._prefill_batch_budget_last_caps = decision.request_token_caps
        self._prefill_batch_budget_last_decision = decision
        return decision

    def _prefill_batch_budget_record_actual(
        self,
        *,
        decision: PrefillBatchBudgetDecision,
        accountant: PrefillBudgetAccountant | None,
        actual_prefill_tokens: int,
        actual_running_prefill_tokens: int,
        actual_waiting_prefill_tokens: int,
        actual_prefill_request_ids: set[str],
    ) -> None:
        self._prefill_batch_budget_actual_tokens_total += actual_prefill_tokens
        self._prefill_batch_budget_running_tokens_total += (
            actual_running_prefill_tokens
        )
        self._prefill_batch_budget_waiting_tokens_total += (
            actual_waiting_prefill_tokens
        )
        self._prefill_batch_budget_last_actual_tokens = actual_prefill_tokens
        self._prefill_batch_budget_last_running_tokens = (
            actual_running_prefill_tokens
        )
        self._prefill_batch_budget_last_waiting_tokens = (
            actual_waiting_prefill_tokens
        )
        if decision.total_prefill_budget > 0:
            self._prefill_batch_budget_last_utilization = (
                actual_prefill_tokens / decision.total_prefill_budget
            )
        else:
            self._prefill_batch_budget_last_utilization = 0.0

        mode = self._prefill_batch_budget_mode()
        if mode == "batch-budget-shadow":
            planned_ids = {
                request_id for request_id, _ in decision.request_token_caps
            }
            self._prefill_batch_budget_shadow_token_diff_total += abs(
                decision.total_prefill_budget - actual_prefill_tokens
            )
            self._prefill_batch_budget_shadow_request_diff_total += len(
                planned_ids.symmetric_difference(actual_prefill_request_ids)
            )

        interval = max(
            1,
            int(self.scheduler_config.prefill_bias_batch_metrics_log_interval),
        )
        if self._prefill_batch_budget_steps_total % interval == 0:
            logger.info(
                "prefill_batch_budget_metrics=%s",
                self.get_prefill_batch_budget_metrics(),
            )

    def get_prefill_batch_budget_metrics(
        self,
    ) -> dict[str, int | float | str | tuple]:
        controller = self.prefill_bias_controller.batch_budget
        decision = self._prefill_batch_budget_last_decision
        return {
            "mode": self._prefill_batch_budget_mode() or "inactive",
            "steps_total": self._prefill_batch_budget_steps_total,
            "planned_tokens_total": self._prefill_batch_budget_planned_tokens_total,
            "actual_tokens_total": self._prefill_batch_budget_actual_tokens_total,
            "running_tokens_total": self._prefill_batch_budget_running_tokens_total,
            "waiting_tokens_total": self._prefill_batch_budget_waiting_tokens_total,
            "last_planned_tokens": decision.total_prefill_budget,
            "last_actual_tokens": self._prefill_batch_budget_last_actual_tokens,
            "last_running_tokens": self._prefill_batch_budget_last_running_tokens,
            "last_waiting_tokens": self._prefill_batch_budget_last_waiting_tokens,
            "last_utilization": self._prefill_batch_budget_last_utilization,
            "selected_ids": self._prefill_batch_budget_last_selected_ids,
            "caps": self._prefill_batch_budget_last_caps,
            "predicted_step_s": decision.predicted_step_time_s,
            "actual_to_predicted_ratio": (
                self._prefill_bias_predictive_actual_to_predicted_ratio
            ),
            "tbt_bound_rejections": controller.tbt_rejections_total,
            "unknown_timing_decode_only": controller.unknown_timing_total,
            "predictor_errors": controller.predictor_errors_total,
            "fail_closed_total": controller.fail_closed_total,
            "shadow_token_diff_total": (
                self._prefill_batch_budget_shadow_token_diff_total
            ),
            "shadow_request_diff_total": (
                self._prefill_batch_budget_shadow_request_diff_total
            ),
            "reason": decision.reason,
        }

    def _prefill_bias_prepare(
        self,
        *,
        token_budget: int,
        defer_prefills: bool,
    ) -> tuple[PrefillBiasDecision, int]:
        decode_floor = self._prefill_bias_decode_floor(token_budget)
        max_safe_reserve = max(0, token_budget - decode_floor)
        config = self._adaptive_prefill_effective_config()
        active_decode_count, minimum_tbt_slack_s, unknown_decode_count = (
            self._prefill_bias_predictive_context()
        )
        deadline_start_ns = time.perf_counter_ns()
        decision = self.prefill_bias_controller.decide(
            waiting=self.waiting,
            policy=self.policy,
            paused=self._pause_state != PauseState.UNPAUSED,
            throttle_prefills=defer_prefills,
            max_safe_reserve=max_safe_reserve,
            is_request_schedulable=self._prefill_bias_request_is_schedulable,
            peek_cached_tokens=self.kv_cache_manager.peek_num_computed_tokens,
            predicted_step_time_s=self._prefill_bias_estimated_step_time(),
            active_decode_count=active_decode_count,
            minimum_tbt_slack_s=minimum_tbt_slack_s,
            unknown_decode_count=unknown_decode_count,
        )
        self._prefill_bias_sync_predictive_metrics(decision)
        if config.prefill_bias_ttft_deadline_enabled:
            self._prefill_bias_ttft_deadline_checks += 1
            self._prefill_bias_ttft_candidate_scan_time_ns += (
                time.perf_counter_ns() - deadline_start_ns
            )
            self._prefill_bias_ttft_min_slack_s = decision.minimum_ttft_slack_s
            self._prefill_bias_ttft_predicted_completion_s = (
                decision.predicted_completion_s
            )
            if decision.active:
                self._prefill_bias_ttft_urgent_activations += 1
        decision = self._prefill_bias_apply_tbt_guard(decision)
        self._prefill_bias_last_decision = decision
        if not decision.active:
            if decision.reason == "no_safe_budget":
                self._prefill_blocked_token_budget += 1
            if decision.reason in ("no_safe_budget", "prefill_throttled", "paused"):
                logger.debug(
                    "Prefill bias inactive: reason=%s decode_floor=%d "
                    "max_safe_reserve=%d waiting=%d",
                    decision.reason,
                    decode_floor,
                    max_safe_reserve,
                    len(self.waiting),
                )
            return decision, decode_floor

        self._prefill_bias_activations += 1
        self._prefill_bias_reserved_tokens += decision.reserve_tokens
        logger.debug(
            "Prefill bias active: request_ids=%s reserve_tokens=%d "
            "decode_floor=%d",
            decision.candidate_request_ids,
            decision.reserve_tokens,
            decode_floor,
        )
        return decision, decode_floor

    def _prefill_bias_score_after_running(
        self,
        decision: PrefillBiasDecision,
    ) -> PrefillBiasDecision:
        config = self._adaptive_prefill_effective_config()
        if config.prefill_bias_ttft_deadline_enabled:
            return decision
        if not decision.active or not config.prefill_bias_cache_aware:
            return decision
        score_start_ns = time.perf_counter_ns()
        scored = self.prefill_bias_controller.score_candidates(
            waiting=self.waiting,
            policy=self.policy,
            reserve_tokens=decision.reserve_tokens,
            peek_cached_tokens=self.kv_cache_manager.peek_num_computed_tokens,
            is_request_schedulable=self._prefill_bias_request_is_schedulable,
        )
        elapsed_ns = time.perf_counter_ns() - score_start_ns
        self._prefill_bias_score_time_ns += elapsed_ns
        self._prefill_bias_last_decision = scored
        if not scored.active:
            return scored

        self._prefill_bias_scored_requests += scored.scored_requests
        self._prefill_bias_cache_aware_activations += 1
        self._prefill_bias_selected_cached_tokens += scored.selected_cached_tokens
        self._prefill_bias_selected_remaining_tokens += (
            scored.selected_remaining_tokens
        )
        scored_ids = set(scored.candidate_request_ids)
        for request in self.waiting:
            if request.request_id in scored_ids:
                age_s = max(0.0, time.time() - float(request.arrival_time or 0.0))
                if age_s >= config.prefill_bias_starvation_s:
                    self._prefill_bias_starvation_overrides += 1
        logger.debug(
            "Prefill bias cache-aware active: request_ids=%s "
            "scored_requests=%d selected_cached_tokens=%d "
            "selected_remaining_tokens=%d score_time_us=%.3f",
            scored.candidate_request_ids,
            scored.scored_requests,
            scored.selected_cached_tokens,
            scored.selected_remaining_tokens,
            elapsed_ns / 1000.0,
        )
        return scored

    def _prefill_bias_promote_waiting(
        self, request_ids: tuple[str, ...]
    ) -> None:
        if not request_ids or self.policy != SchedulingPolicy.FCFS:
            return
        selected_by_id = {request_id: None for request_id in request_ids}
        for request in list(self.waiting):
            if request.request_id in selected_by_id:
                selected_by_id[request.request_id] = request
        selected = [
            selected_by_id[request_id]
            for request_id in request_ids
            if selected_by_id[request_id] is not None
        ]
        if not selected:
            return
        original = list(self.waiting)
        selected_set = set(selected)
        sticky_prefix = []
        remainder_start = 0
        for request in original:
            if request.status == RequestStatus.PREEMPTED:
                sticky_prefix.append(request)
                remainder_start += 1
                continue
            break
        remainder = [
            request
            for request in original[remainder_start:]
            if request not in selected_set
        ]
        reordered = [*sticky_prefix, *selected, *remainder]
        self.waiting.clear()
        self.waiting.extend(reordered)

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        num_uncached_common_prefix_tokens: int = 0,
    ) -> int:
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # As an exception, if `num_new_tokens` is less than `block_size`, the
            # state is simply not cached, requiring no special handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass

            # Marconi cache admission optimization:
            # cache common prefixes by scheduling num_new_tokens = common prefix length
            if (
                num_uncached_common_prefix_tokens >= block_size
                and num_new_tokens > num_uncached_common_prefix_tokens
            ):
                num_new_tokens = num_uncached_common_prefix_tokens
                # keep alignment to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
        return num_new_tokens

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Whether the running batch contains any prefill requests.
        prefill_scheduled = False

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        self._adaptive_prefill_observe_pressure(token_budget)
        self._adaptive_prefill_refresh_policy()

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.
        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # VLLM_PREFILL_BIAS_PATCH: reserve a safe slice for urgent waiting prefills.
        prefill_bias_held_tokens = 0
        prefill_bias_candidate_ids: set[str] = set()
        prefill_bias_candidate_token_caps: dict[str, int] = {}
        prefill_bias_swap_performed = False
        prefill_bias_swap_candidate_id: str | None = None
        prefill_bias_swaps_this_step = 0
        batch_budget_mode = self._prefill_batch_budget_mode()
        batch_budget_enforced = batch_budget_mode == "batch-budget"
        batch_budget_accountant: PrefillBudgetAccountant | None = None
        batch_budget_decision = self._prefill_batch_budget_last_decision
        if batch_budget_mode is not None:
            batch_budget_decision = self._prefill_batch_budget_prepare(
                token_budget=token_budget,
            )

        if batch_budget_enforced:
            batch_budget_accountant = PrefillBudgetAccountant(
                batch_budget_decision
            )
            prefill_bias_candidate_ids = set(
                batch_budget_decision.waiting_prefill_order
            )
            prefill_bias_candidate_token_caps = dict(
                batch_budget_decision.request_token_caps
            )
            first_waiting_id = (
                batch_budget_decision.waiting_prefill_order[0]
                if batch_budget_decision.waiting_prefill_order
                else None
            )
            slot_swap_eligible = (
                first_waiting_id is not None
                and first_waiting_id
                in set(batch_budget_decision.slot_swap_request_ids)
            )
            prefill_bias_decision = PrefillBiasDecision(
                active=first_waiting_id is not None,
                reserve_tokens=0,
                candidate_request_ids=(
                    batch_budget_decision.waiting_prefill_order
                ),
                candidate_token_caps=batch_budget_decision.request_token_caps,
                reason=batch_budget_decision.reason,
                minimum_ttft_slack_s=(
                    -1.0 if slot_swap_eligible else math.inf
                ),
                predicted_step_time_s=(
                    batch_budget_decision.predicted_step_time_s
                ),
                policy_mode="batch-budget",
                slot_swap_eligible=slot_swap_eligible,
            )
            self._prefill_bias_last_decision = prefill_bias_decision
            self._prefill_bias_promote_waiting(
                batch_budget_decision.waiting_prefill_order
            )
        else:
            prefill_bias_decision, _ = self._prefill_bias_prepare(
                token_budget=token_budget,
                defer_prefills=defer_prefills,
            )
            if prefill_bias_decision.active:
                prefill_bias_held_tokens = prefill_bias_decision.reserve_tokens
                prefill_bias_candidate_ids = set(
                    prefill_bias_decision.candidate_request_ids
                )
                prefill_bias_candidate_token_caps = dict(
                    prefill_bias_decision.candidate_token_caps
                )
                token_budget -= prefill_bias_held_tokens

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue

            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue
            if (
                batch_budget_enforced
                and request.is_prefill_chunk
                and not batch_budget_accountant.has_cap(request.request_id)
            ):
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            if batch_budget_enforced and request.is_prefill_chunk:
                num_new_tokens = batch_budget_accountant.clamp(
                    request.request_id,
                    num_new_tokens,
                )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            restored_tokens = num_scheduled_tokens.pop(
                                preempted_req_id
                            )
                            token_budget += restored_tokens
                            if (
                                batch_budget_accountant is not None
                                and batch_budget_accountant.has_cap(preempted_req_id)
                            ):
                                batch_budget_accountant.restore(
                                    preempted_req_id,
                                    restored_tokens,
                                    source="running",
                                )
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            if batch_budget_enforced and request.is_prefill_chunk:
                batch_budget_accountant.commit(
                    request_id,
                    num_new_tokens,
                    source="running",
                )
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # VLLM_PREFILL_BIAS_PATCH: restore held tokens exactly once before WAITING.
        if prefill_bias_held_tokens:
            token_budget += prefill_bias_held_tokens
            prefill_bias_held_tokens = 0
            prefill_bias_decision = self._prefill_bias_score_after_running(
                prefill_bias_decision
            )
            prefill_bias_candidate_ids = set(
                prefill_bias_decision.candidate_request_ids
            )
            prefill_bias_candidate_token_caps = dict(
                prefill_bias_decision.candidate_token_caps
            )
            self._prefill_bias_promote_waiting(
                prefill_bias_decision.candidate_request_ids
            )

        prefill_bias_swap_result, token_budget = self._try_prefill_admission_swap(
            decision=prefill_bias_decision,
            token_budget=token_budget,
            scheduled_timestamp=scheduled_timestamp,
            scheduled_running_reqs=scheduled_running_reqs,
            preempted_reqs=preempted_reqs,
            req_to_new_blocks=req_to_new_blocks,
            num_scheduled_tokens=num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            swaps_this_step=prefill_bias_swaps_this_step,
            prefill_budget_accountant=batch_budget_accountant,
        )
        if prefill_bias_swap_result.performed:
            prefill_bias_swap_performed = True
            prefill_bias_swaps_this_step += 1
            prefill_bias_swap_candidate_id = (
                prefill_bias_swap_result.candidate_request_id
            )

        # Next, schedule the WAITING requests.
        if (
            (not preempted_reqs or prefill_bias_swap_performed)
            and self._pause_state == PauseState.UNPAUSED
        ):
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    if prefill_bias_candidate_ids:
                        self._prefill_blocked_max_num_seqs += 1
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
                num_uncached_common_prefix_tokens = 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    if (
                        self.connector is not None
                        and self.has_mamba_layers
                        and isinstance(
                            self.kv_cache_manager.coordinator,
                            HybridKVCacheCoordinator,
                        )
                    ):
                        computed, per_group_hits = (
                            self.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
                                request.block_hashes,
                                request.num_tokens - 1,
                            )
                        )
                        new_computed_blocks = (
                            self.kv_cache_manager.create_kv_cache_blocks(computed)
                        )
                        # NOTE(ZhanqiuHu): For Mamba hybrid models,
                        # num_new_local_computed_tokens should be the FA hit
                        # length. This value is passed to the connector's
                        # get_num_new_matched_tokens which computes:
                        # external = total - local_computed.
                        # Using the FA hit skips re-transferring FA blocks
                        # already cached on D-side. The Mamba state (always
                        # the last block) is transferred unconditionally by
                        # _apply_prefix_caching in nixl/worker.py.
                        num_new_local_computed_tokens = max(per_group_hits)
                        if self.kv_cache_manager.log_stats:
                            assert self.kv_cache_manager.prefix_cache_stats is not None
                            self.kv_cache_manager.prefix_cache_stats.record(
                                num_tokens=request.num_tokens,
                                num_hits=num_new_local_computed_tokens,
                                preempted=request.num_preemptions > 0,
                            )
                    else:
                        new_computed_blocks, num_new_local_computed_tokens = (
                            self.kv_cache_manager.get_computed_blocks(request)
                        )

                    # In case of hybrid models, obtain hint for Marconi-style APC logic
                    if self.has_mamba_layers:
                        num_uncached_common_prefix_tokens = getattr(
                            self.kv_cache_manager.coordinator,
                            "num_uncached_common_prefix_tokens",
                            0,
                        )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens

                    # Skip request with pending mm encoding prefetches
                    if (
                        self.ec_connector is not None
                        and request.mm_features
                        and not self.ec_connector.ensure_cache_available(
                            request, num_computed_tokens
                        )
                    ):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats is not None:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget
                pad_spec_decode = False

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                elif (
                    batch_budget_enforced
                    and not batch_budget_accountant.has_cap(request_id)
                ):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue
                elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                    # DP prefill balancing: defer this step's local prefill
                    # compute to a cadence-aligned step.
                    break
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens

                    # Pad new decode requests to uniform spec decoding size to
                    # preserve full cudagraph for this step.
                    if (
                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                        and num_new_tokens == 1
                        and (scheduled_running_reqs and not prefill_scheduled)
                    ):
                        num_new_tokens = 1 + self.num_spec_tokens
                        if (
                            num_new_tokens > token_budget
                            or num_computed_tokens + num_new_tokens > self.max_model_len
                        ):
                            # Prefer to not schedule than schedule un-padded here.
                            break
                        pad_spec_decode = True

                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    candidate_cap = prefill_bias_candidate_token_caps.get(request_id)
                    if candidate_cap is not None:
                        num_new_tokens = min(num_new_tokens, candidate_cap)
                    if batch_budget_enforced:
                        num_new_tokens = batch_budget_accountant.clamp(
                            request_id,
                            num_new_tokens,
                        )
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Skip block alignment when setting up async receive (no local work).
                if self.need_mamba_block_aligned_split and not load_kv_async:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                        num_uncached_common_prefix_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # During async KV load, no forward pass is run yet.
                # Allocate speculative lookahead slots later to avoid
                # mismatching local and remote block counts.
                limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
                effective_lookahead_tokens = (
                    0 if limit_lookahead_tokens else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations), to
                    # avoid deadlock and predictable preemptions.
                    reserved_blocks = self._inflight_prefill_reserved_blocks()

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    reserved_blocks=reserved_blocks,
                    has_scheduled_reqs=bool(self.running),
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    if request_id in prefill_bias_candidate_ids:
                        self._prefill_blocked_no_kv += 1
                    if request_id == prefill_bias_swap_candidate_id:
                        self._prefill_slot_swap_commit_failures_total += 1
                        self._prefill_bias_phase3_candidate_backoff_until[
                            request_id
                        ] = (
                            self._prefill_bias_monotonic_clock()
                            + self._adaptive_prefill_effective_config().prefill_bias_swap_failure_backoff_s
                        )

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    self._inflight_prefills.add(request)
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                    if request_id in prefill_bias_candidate_ids:
                        self._prefill_bias_admitted_requests += 1
                    if request_id == prefill_bias_swap_candidate_id:
                        self._prefill_slot_swap_candidate_admissions_total += 1
                elif request.status == RequestStatus.PREEMPTED:
                    self._prefill_bias_phase3_last_resumed_ts[request_id] = (
                        self._prefill_bias_monotonic_clock()
                    )
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                if batch_budget_enforced:
                    batch_budget_accountant.commit(
                        request_id,
                        num_new_tokens,
                        source="waiting",
                    )
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if pad_spec_decode:
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

            # DP prefill balancing: on a step that admitted prefills (release),
            # record whether it was capacity-bound.
            if not defer_prefills:
                self.prefill_capacity_bound = bool(self.waiting)

        # VLLM_PREFILL_BIAS_PHASE2_PATCH: classify scheduled prefill work before
        # _update_after_schedule mutates request.num_computed_tokens/is_prefill_chunk.
        prefill_tokens_in_batch = sum(
            num_tokens
            for request_id, num_tokens in num_scheduled_tokens.items()
            if self.requests[request_id].num_output_tokens == 0
        )
        actual_prefill_request_ids = {
            request_id
            for request_id in num_scheduled_tokens
            if self.requests[request_id].num_output_tokens == 0
        }
        actual_running_prefill_tokens = sum(
            num_scheduled_tokens.get(request.request_id, 0)
            for request in scheduled_running_reqs
            if request.request_id in actual_prefill_request_ids
        )
        actual_waiting_prefill_tokens = (
            prefill_tokens_in_batch - actual_running_prefill_tokens
        )
        if batch_budget_enforced:
            batch_budget_accountant.assert_invariants()
            prefill_tokens_in_batch = batch_budget_accountant.committed_tokens
            actual_prefill_request_ids = {
                request_id
                for request_id, _ in batch_budget_accountant.committed_by_request
            }
            actual_running_prefill_tokens = (
                batch_budget_accountant.running_prefill_tokens
            )
            actual_waiting_prefill_tokens = (
                batch_budget_accountant.waiting_prefill_tokens
            )
            assert (
                prefill_tokens_in_batch
                <= batch_budget_decision.total_prefill_budget
            )
        decode_requests_in_batch = sum(
            1
            for request_id in num_scheduled_tokens
            if request_id not in actual_prefill_request_ids
        )
        if batch_budget_mode is not None:
            self._prefill_batch_budget_record_actual(
                decision=batch_budget_decision,
                accountant=batch_budget_accountant,
                actual_prefill_tokens=prefill_tokens_in_batch,
                actual_running_prefill_tokens=actual_running_prefill_tokens,
                actual_waiting_prefill_tokens=actual_waiting_prefill_tokens,
                actual_prefill_request_ids=actual_prefill_request_ids,
            )

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step (MRV1-only).
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        self._prefill_bias_track_scheduled_batch(
            scheduler_output,
            prefill_tokens_in_batch,
            decode_requests_in_batch,
        )

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self._free_request_blocks(request)
        self.encoder_cache_manager.free(request)
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        if self.enable_return_routed_experts:
            gid = self.routed_experts_mgr.attn_gid
            self._re_block_ids.update(
                {
                    rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
                    for rid in num_scheduled_tokens
                }
            )

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            if not self.use_v2_model_runner:  # noqa: SIM102
                if req_id not in self.prev_step_scheduled_req_ids:
                    all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0

        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
        )
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0

        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        self._prefill_bias_observe_batch_timing(scheduler_output)

        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None
        routing_offsets: dict[str, int] = {}
        if model_runner_output.routed_experts is not None:
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:
                routing_offsets[rid] = offset
                offset += num_scheduled_tokens[rid]

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            # Skip a stale frame still pending discard (async_tokens_to_discard
            # > 0): its pre-reset rejection count would underflow the counters.
            if (
                scheduled_spec_token_ids
                and (generated_token_ids or self.num_sampled_tokens_per_step == 0)
                and request.async_tokens_to_discard == 0
            ):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status
            num_output_tokens_before = len(request._output_token_ids)

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
                if new_token_ids:
                    accepted_ts = self._prefill_bias_monotonic_clock()
                    self._record_accepted_decode_output(
                        request,
                        new_token_ids,
                        accepted_ts,
                    )
                    self._adaptive_prefill_observe_accepted_tokens(
                        request,
                        len(new_token_ids),
                        now_monotonic=accepted_ts,
                    )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar
                assert grammar is not None
                # new_token_ids can be a mixed block of reasoning content, then
                # the reasoning end marker, then the start of the grammar content.
                # Trim the reasoning content so the grammar only sees grammar content.
                advance_token_ids = (
                    self.structured_output_manager.trim_reasoning_for_advance(
                        request, new_token_ids
                    )
                )
                if advance_token_ids and not grammar.accept_tokens(
                    req_id, advance_token_ids
                ):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        advance_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            routed_experts = None
            if (
                self.enable_return_routed_experts
                and routing_data is not None
                and new_token_ids
            ):
                req_offset = routing_offsets[req_id]
                end = req_offset + num_tokens_scheduled
                block_ids = self._re_block_ids.pop(req_id, [])
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).
                    if (
                        request.sampling_params is not None
                        and request.sampling_params.routed_experts_prompt_start
                        is not None
                    ):
                        prompt_start = (
                            request.sampling_params.routed_experts_prompt_start
                        )
                        assert prompt_start < request.num_prompt_tokens
                    else:
                        prompt_start = 0
                    routed_experts = self.routed_experts_mgr.get(
                        block_ids,
                        request.num_prompt_tokens,
                        token_start=prompt_start,
                    )
                else:
                    if scheduled_spec_token_ids:
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        routed_experts = routing_data[
                            req_offset : req_offset + len(new_token_ids)
                        ]
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        routed_experts = routing_data[end - len(new_token_ids) : end]

            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=request.take_prefill_stats(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if (
                scheduler_kv_connector_stats is not None
                and not scheduler_kv_connector_stats.is_empty()
            ):
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Defer the free by the drafter's look-ahead so an entry stays
        # referenced until the drafter's +1 read has also passed it, mirroring
        # the shift the encoder scheduling path applies.
        spec_lookahead = 1 if self.use_eagle else 0

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif (
                start_pos + num_tokens + spec_lookahead
                <= request.num_computed_tokens - request.num_output_placeholders
            ):
                # Processed, stored in the decoder KV cache, and far enough past
                # the placeholder range (plus the drafter's look-ahead) that no
                # rejection or drafter gather can reference it.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            self._clear_decode_timing(request.request_id)
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.connector is not None:
                self.connector.on_new_request(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self._adaptive_prefill_observe_request_finished(request)
        self._clear_decode_timing(request_id)
        self._prefill_bias_phase3_preemption_count.pop(request_id, None)
        self._prefill_bias_phase3_last_preempted_ts.pop(request_id, None)
        self._prefill_bias_phase3_last_resumed_ts.pop(request_id, None)
        self._prefill_bias_phase3_candidate_backoff_until.pop(request_id, None)
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        if not self.defer_block_free or (
            # Last scheduled step already processed: no in-flight write remains
            # (always the case for a normal finish), so free now.
            request.last_sched_seq <= self.processed_step_seq
        ):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))

    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Entries are appended with monotonically non-decreasing fences, so
        stop at the first one that is still pending.
        """
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:
            return True
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_in_queues = (
            len(self.waiting) + len(self.skipped_waiting) + len(self.running)
        )
        return len(self.requests) > num_in_queues

    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or (self.connector is not None and self.connector.has_pending_push_work())
        )

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # For async scheduling, any output frames already in flight at
                # preemption time are now stale and must be discarded when they
                # return. num_output_placeholders is exactly that count: 0 if
                # the engine has drained (e.g. pause_generation(keep) waited
                # for idle), 1 for vanilla async mid-step, or 1 + spec/PP frames
                # otherwise.
                request.async_tokens_to_discard = request.num_output_placeholders
                request.num_output_placeholders = 0

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug(
                "reset_connector requested but no KV connector is configured; "
                "treating as no-op success."
            )
            return True

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_computed_tokens,
            num_prompt_tokens=request.num_prompt_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
            request_id=request.request_id,
            num_computed_tokens=request.num_computed_tokens,
        )

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to allocate to hold its full sequence."""
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )

    def _inflight_prefill_reserved_blocks(self) -> int:
        """Num blocks in-flight prefills still need to finish (their reservation)."""

        return sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills
        )

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not (structured_output_req and structured_output_req.grammar):
                return False
            request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids

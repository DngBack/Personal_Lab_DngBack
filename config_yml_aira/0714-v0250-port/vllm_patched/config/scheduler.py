# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Callable
from dataclasses import InitVar
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import Field, field_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.core.sched.interface import SchedulerInterface

logger = init_logger(__name__)

RunnerType = Literal["generate", "pooling", "draft"]
SchedulerPolicy = Literal["fcfs", "priority"]


@config
class SchedulerConfig:
    """Scheduler configuration."""

    max_model_len: InitVar[int]
    """Maximum length of a sequence (including prompt and generated text).

    Note: This is stored in the ModelConfig, and is used only here to
    provide fallbacks and validate other attributes."""

    is_encoder_decoder: InitVar[bool]
    """True if the model is an encoder-decoder model.

    Note: This is stored in the ModelConfig, and is used only here to
    disable chunked prefill and prefix caching for encoder-decoder models.
    """

    DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048
    DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP: ClassVar[int] = 256
    DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128

    runner_type: RunnerType = "generate"
    """The runner type to launch for the model."""

    max_num_batched_tokens: int = Field(default=DEFAULT_MAX_NUM_BATCHED_TOKENS, ge=1)
    """Maximum number of tokens that can be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    max_num_scheduled_tokens: int | None = Field(default=None, ge=0)
    """Maximum number of tokens that the scheduler may issue in a single iteration.
    
    This is usually equal to max_num_batched_tokens, but can be smaller in cases
    when the model might append tokens into the batch (such as speculative decoding).
    Defaults to max_num_batched_tokens."""

    max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)
    """Maximum number of sequences to be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    max_num_partial_prefills: int = Field(default=1, ge=1)
    """For chunked prefill, the maximum number of sequences that can be
    partially prefilled concurrently."""

    max_long_partial_prefills: int = Field(default=1, ge=1)
    """For chunked prefill, the maximum number of prompts longer than
    long_prefill_token_threshold that will be prefilled concurrently. Setting
    this less than max_num_partial_prefills will allow shorter prompts to jump
    the queue in front of longer prompts in some cases, improving latency."""

    long_prefill_token_threshold: int = Field(default=0, ge=0)
    """For chunked prefill, a request is considered long if the prompt is
    longer than this number of tokens. 0 disables the cap (default)."""

    enable_chunked_prefill: bool = True
    """If True, prefill requests can be chunked based
    on the remaining `max_num_batched_tokens`.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    is_multimodal_model: bool = False
    """True if the model is multimodal."""

    # TODO (ywang96): Make this configurable.
    max_num_encoder_input_tokens: int = Field(init=False)
    """Multimodal encoder compute budget, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    # TODO (ywang96): Make this configurable.
    encoder_cache_size: int = Field(init=False)
    """Multimodal encoder cache size, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:

    - "fcfs" means first come first served, i.e. requests are handled in order 
      of arrival.
    - "priority" means requests are handled based on given priority (lower
      value means earlier handling) and time of arrival deciding any ties)."""

    disable_chunked_mm_input: bool = False
    """If set to true and chunked prefill is enabled, we do not want to
    partially schedule a multimodal item. Only used in V1
    This ensures that if a request has a mixed prompt
    (like text tokens TTTT followed by image tokens IIIIIIIIII) where only
    some image tokens can be scheduled (like TTTTIIIII, leaving IIIII),
    it will be scheduled as TTTT in one step and IIIIIIIIII in the next."""

    # scheduler class or path. "vllm.v1.core.sched.scheduler.Scheduler"
    # (default) or "mod.custom_class".
    scheduler_cls: str | type[object] | None = None
    """The scheduler class to use. "vllm.v1.core.sched.scheduler.Scheduler" is
    the default scheduler. Can be a class directly or the path to a class of
    form "mod.custom_class"."""

    disable_hybrid_kv_cache_manager: bool | None = None
    """If set to True, KV cache manager will allocate the same size of KV cache
    for all attention layers even if there are multiple type of attention layers
    like full attention and sliding window attention.
    If set to None, the default value will be determined based on the environment
    and starting configuration.
    """

    scheduler_reserve_full_isl: bool = True
    """If True, the scheduler checks whether the full input sequence length
    fits in the KV cache before admitting a new request, rather than only
    checking the first chunk. Prevents over-admission and KV cache thrashing
    with chunked prefill."""

    watermark: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Fraction of total KV cache blocks to keep free (the watermark) when
    admitting waiting or preempted requests into the running queue. This headroom
    helps avoid frequent KV cache eviction and the resulting repeated preemption
    of requests when GPU memory is scarce. Must be in the range [0.0, 1.0); 0.0
    (the default) disables the watermark."""

    prefill_schedule_interval: int = Field(default=1, ge=1)
    """For data-parallel deployments, only admit new prefill requests
    once every N engine steps, aligned across DP ranks, to better balance
    per-step forward-pass times."""

    # VLLM_PREFILL_BIAS_PATCH: scheduler_config fields
    prefill_bias_enabled: bool = False
    """Enable experimental adaptive prefill-biased scheduling."""

    prefill_bias_wait_threshold_s: float = Field(default=0.03, ge=0.0)
    """Waiting age in seconds before a normal waiting prefill becomes urgent."""

    prefill_bias_reserve_tokens: int = Field(default=64, ge=0)
    """Scheduler token budget to hold for urgent prefills before scheduling decodes."""

    prefill_bias_max_requests_per_step: int = Field(default=1, ge=1)
    """Maximum urgent waiting prefills to promote in one scheduler step."""

    prefill_bias_cache_aware: bool = False
    """Enable cache-aware ordering among urgent waiting prefills."""

    prefill_bias_score_window_k: int = Field(default=16, ge=1)
    """Score at most this many eligible non-sticky waiting prefills."""

    # VLLM_PREFILL_BIAS_PHASE5_PATCH: bounded cache-aware candidate inspection.
    prefill_bias_candidate_scan_limit: int = Field(default=16, ge=1)
    """Inspect at most this many normal waiting prefills for cache-aware scoring."""

    prefill_bias_min_cached_tokens: int = Field(default=0, ge=0)
    """Treat cache hits below this token threshold as cold for ordering."""

    prefill_bias_starvation_s: float = Field(default=0.2, ge=0.0)
    """Waiting age after which FCFS order overrides cache-aware preference."""

    prefill_bias_remaining_token_buckets: tuple[int, ...] = (16, 64, 256, 1024)
    """Bucket edges for remaining prefill work in cache-aware ordering."""

    # VLLM_PREFILL_BIAS_PHASE2_PATCH: TBT guard config fields
    prefill_bias_tbt_guard_enabled: bool = False
    """Enable conservative TBT/ITL slack guard for prefill bias."""

    # VLLM_PREFILL_BIAS_PHASE6_PATCH: accepted-output guard compatibility.
    prefill_bias_tbt_guard_s: float = Field(default=0.0, ge=0.0)
    """Compatibility alias: >0 enables the TBT guard and supplies the SLO."""

    prefill_bias_tbt_slo_s: float = Field(default=0.0, ge=0.0)
    """Target engine-core inter-token interval in seconds. 0 disables guard config."""

    prefill_bias_tbt_safety_margin_s: float = Field(default=0.005, ge=0.0)
    """Extra guard margin for scheduling overhead and estimation error."""

    prefill_bias_initial_step_time_s: float = Field(default=0.01, gt=0.0)
    """Conservative prefill-containing step-time estimate before EWMA is trusted."""

    prefill_bias_step_time_ewma_alpha: float = Field(default=0.2, gt=0.0, le=1.0)
    """EWMA alpha for observed prefill-containing scheduler step duration."""

    prefill_bias_step_time_headroom_factor: float = Field(default=1.25, ge=1.0)
    """Multiplier applied to trusted observed prefill step-time EWMA."""

    prefill_bias_guard_unknown_decode: bool = True
    """Deny prefill bias when an active decode lacks last-output state."""

    prefill_bias_step_observation_min_samples: int = Field(default=3, ge=1)
    """Minimum prefill-containing batch observations before trusting EWMA."""

    # VLLM_PREFILL_BIAS_PHASE3_PATCH: conservative slot-swap config fields
    prefill_bias_slot_swap_enabled: bool = False
    """Enable conservative Phase 3 prefill admission swap."""

    # VLLM_PREFILL_BIAS_PHASE7_PATCH: TTFT deadline scheduling controls.
    prefill_bias_ttft_deadline_enabled: bool = False
    """Use predicted completion slack instead of a fixed waiting threshold."""

    prefill_bias_ttft_force_preempt_enabled: bool = False
    """Allow a projected TTFT miss to override TBT victim protection."""

    # VLLM_PREFILL_BIAS_PHASE8_PATCH: runtime-selectable predictive policy.
    prefill_bias_policy_mode: str = "legacy"
    """Select legacy, predictive-shadow, or predictive scheduling."""

    prefill_bias_predictive_min_chunk_tokens: int = Field(default=128, ge=1)
    prefill_bias_predictive_max_reserve_tokens: int = Field(default=3072, ge=0)
    prefill_bias_predictive_max_requests_per_step: int = Field(default=4, ge=1)
    prefill_bias_predictive_starvation_multiplier: float = Field(default=4.0, ge=1.0)

    # VLLM_PREFILL_BIAS_PHASE9_PATCH: global prefill batch-budget controls.
    prefill_bias_batch_min_prefill_tokens: int = Field(default=128, ge=1)
    prefill_bias_batch_max_prefill_tokens: int = Field(default=3072, ge=0)
    prefill_bias_batch_max_requests_per_step: int = Field(default=4, ge=1)
    prefill_bias_batch_running_scan_limit: int = Field(default=32, ge=1)
    prefill_bias_batch_metrics_log_interval: int = Field(default=100, ge=1)

    prefill_bias_ttft_slo_s: float | None = None
    """Target TTFT SLO in seconds for urgent prefill slot swaps."""

    prefill_bias_swap_slack_threshold_s: float = Field(default=0.0, ge=0.0)
    """Attempt a swap when predicted TTFT slack is at or below this value."""

    prefill_bias_max_swaps_per_step: int = Field(default=1, ge=1)
    """Maximum Phase 3 swaps per scheduler step."""

    prefill_bias_max_preemptions_per_request: int = Field(default=1, ge=1)
    """Maximum Phase 3 preemptions for a decode request."""

    prefill_bias_swap_cooldown_s: float = Field(default=0.2, ge=0.0)
    """Cooldown before a Phase 3 victim can be selected again."""

    prefill_bias_swap_failure_backoff_s: float = Field(default=0.2, ge=0.0)
    """Backoff for a candidate after commit/preflight failure."""

    prefill_bias_max_candidate_remaining_tokens: int = Field(default=256, ge=1)
    """Maximum remaining prefill tokens for a Phase 3 candidate."""

    prefill_bias_max_victim_recompute_tokens: int | None = None
    """Optional maximum recompute tokens for a Phase 3 victim."""

    prefill_bias_victim_tbt_margin_s: float = Field(default=0.0, ge=0.0)
    """Extra TBT slack margin required for the selected victim."""

    prefill_bias_require_cache_residency: bool = True
    """Require a positive local cache hit for Phase 3 swap candidates."""

    # VLLM_PREFILL_BIAS_PHASE4_PATCH: SLO/goodput adaptive controller fields
    adaptive_prefill_controller_enabled: bool = False
    """Enable bounded SLO/goodput-driven runtime policy adaptation."""

    adaptive_prefill_ttft_slo_s: float = Field(default=0.0, ge=0.0)
    adaptive_prefill_tbt_slo_s: float = Field(default=0.0, ge=0.0)
    adaptive_prefill_target_ttft_attainment: float = Field(default=0.95, gt=0.0, le=1.0)
    adaptive_prefill_target_tbt_attainment: float = Field(default=0.99, gt=0.0, le=1.0)
    adaptive_prefill_target_joint_attainment: float = Field(default=0.95, gt=0.0, le=1.0)
    adaptive_prefill_control_interval_s: float = Field(default=0.25, gt=0.0)
    adaptive_prefill_window_s: float = Field(default=2.0, gt=0.0)
    adaptive_prefill_min_samples: int = Field(default=8, ge=1)
    adaptive_prefill_ema_alpha: float = Field(default=0.2, gt=0.0, le=1.0)
    adaptive_prefill_enter_epochs: int = Field(default=2, ge=1)
    adaptive_prefill_exit_epochs: int = Field(default=3, ge=1)
    adaptive_prefill_cooldown_s: float = Field(default=0.25, ge=0.0)
    adaptive_prefill_max_level_step: int = Field(default=1, ge=1)
    adaptive_prefill_overload_epochs: int = Field(default=2, ge=1)
    adaptive_prefill_min_reserve_tokens: int = Field(default=0, ge=0)
    adaptive_prefill_max_reserve_tokens: int = Field(default=128, ge=0)
    adaptive_prefill_min_chunk_tokens: int = Field(default=16, ge=1)
    adaptive_prefill_max_chunk_tokens: int = Field(default=256, ge=1)
    adaptive_prefill_max_swaps_per_epoch: int = Field(default=1, ge=0)
    adaptive_prefill_tbt_emergency_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    adaptive_prefill_max_swap_failures_per_window: int = Field(default=3, ge=0)
    adaptive_prefill_max_recompute_tokens_per_window: int = Field(default=4096, ge=0)

    async_scheduling: bool | None = None
    """If set to False, disable async scheduling. Async scheduling helps to
    avoid gaps in GPU utilization, leading to better latency and throughput.
    """

    stream_interval: int = Field(default=1, ge=1)
    """The interval (or buffer size) for streaming in terms of token length.
    A smaller value (1) makes streaming smoother by sending each token immediately,
    while a larger value (e.g., 10) reduces host overhead and may increase throughput
    by batching multiple tokens before sending."""

    @staticmethod
    def default_factory(**kwargs):
        """
        Factory method to create `SchedulerConfig` with default values for `InitVar`s.
        """
        if "max_model_len" not in kwargs:
            kwargs["max_model_len"] = 8192
        if "is_encoder_decoder" not in kwargs:
            kwargs["is_encoder_decoder"] = False
        return SchedulerConfig(**kwargs)

    def get_scheduler_cls(self) -> type["SchedulerInterface"]:
        if self.scheduler_cls is None:
            if self.async_scheduling:
                from vllm.v1.core.sched.async_scheduler import AsyncScheduler

                return AsyncScheduler
            from vllm.v1.core.sched.scheduler import Scheduler

            return Scheduler

        # The first half of this warning can be removed once the Scheduler interface is
        # finalized and we can maintain support for scheduler classes that implement it
        logger.warning_once(
            "Using custom scheduler class %s. This scheduler interface is not public "
            "and compatibility may not be maintained. If you have subclassed Scheduler "
            "instead of AsyncScheduler, you will see degraded performance due to async "
            "scheduling being disabled.",
            self.scheduler_cls,  # type: ignore[arg-type]
        )
        if not isinstance(self.scheduler_cls, str):
            return cast(type["SchedulerInterface"], self.scheduler_cls)
        return resolve_obj_by_qualname(self.scheduler_cls)

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []

        # max_num_batched_tokens need to be included in the hash due
        # to two reasons:
        # 1. LoRA creates static buffers based on max_num_batched_tokens.
        #   The tensor sizes and strides get captured in the torch.compile
        #   graph explicitly.
        # 2. Inductor decides whether using 32-bit or 64-bit indexing integer
        #   based on the data sizes. `max_num_batched_tokens` has an
        #   impact on that. For more details, please check
        #   https://github.com/vllm-project/vllm/issues/29585
        factors.append(self.max_num_batched_tokens)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @field_validator("scheduler_cls", "async_scheduling", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        return None if value is None else handler(value)

    def __post_init__(self, max_model_len: int, is_encoder_decoder: bool) -> None:
        if is_encoder_decoder:
            # Chunked prefill should be disabled for encoder-decoder models.
            self.disable_chunked_mm_input = True
            self.enable_chunked_prefill = False
            self.long_prefill_token_threshold = 0
            logger.info(
                "Encoder-decoder models do not support chunked prefill nor"
                " prefix caching; disabling both."
            )

        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens

        if self.enable_chunked_prefill:
            logger.info_once(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens,
            )

        if self.max_num_partial_prefills > 1:
            if self.long_prefill_token_threshold == 0:
                self.long_prefill_token_threshold = int(max_model_len * 0.04)

            logger.info(
                "Concurrent partial prefills enabled with "
                "max_num_partial_prefills=%d, max_long_partial_prefills=%d, "
                "long_prefill_token_threshold=%d",
                self.max_num_partial_prefills,
                self.max_long_partial_prefills,
                self.long_prefill_token_threshold,
            )

        # VLLM_PREFILL_BIAS_PATCH: scheduler_config validation
        resolved_max_num_scheduled_tokens = (
            self.max_num_scheduled_tokens
            if self.max_num_scheduled_tokens is not None
            else self.max_num_batched_tokens
        )
        if self.prefill_bias_reserve_tokens > resolved_max_num_scheduled_tokens:
            raise ValueError(
                "prefill_bias_reserve_tokens "
                f"({self.prefill_bias_reserve_tokens}) must be less than or equal "
                "to resolved max_num_scheduled_tokens "
                f"({resolved_max_num_scheduled_tokens})."
            )
        if self.prefill_bias_max_requests_per_step > self.max_num_seqs:
            raise ValueError(
                "prefill_bias_max_requests_per_step "
                f"({self.prefill_bias_max_requests_per_step}) must be less than "
                f"or equal to max_num_seqs ({self.max_num_seqs})."
            )
        if self.prefill_bias_candidate_scan_limit < self.prefill_bias_max_requests_per_step:
            raise ValueError(
                "prefill_bias_candidate_scan_limit must be greater than or equal "
                "to prefill_bias_max_requests_per_step."
            )
        if self.prefill_bias_enabled and self.policy != "fcfs":
            raise ValueError(
                "prefill_bias_enabled requires the FCFS scheduling policy."
            )
        if self.prefill_bias_cache_aware:
            if not self.prefill_bias_enabled:
                raise ValueError(
                    "prefill_bias_cache_aware requires prefill_bias_enabled."
                )
            if self.policy != "fcfs":
                raise ValueError(
                    "prefill_bias_cache_aware requires the FCFS scheduling policy."
                )
        bucket_edges = tuple(self.prefill_bias_remaining_token_buckets)
        if any(edge <= 0 for edge in bucket_edges):
            raise ValueError(
                "prefill_bias_remaining_token_buckets must contain only "
                "positive integers."
            )
        if tuple(sorted(bucket_edges)) != bucket_edges or len(set(bucket_edges)) != len(
            bucket_edges
        ):
            raise ValueError(
                "prefill_bias_remaining_token_buckets must be strictly increasing "
                "and unique."
            )

        if self.prefill_bias_tbt_guard_s > 0.0:
            if self.prefill_bias_tbt_slo_s == 0.0:
                self.prefill_bias_tbt_slo_s = self.prefill_bias_tbt_guard_s
            elif self.prefill_bias_tbt_slo_s != self.prefill_bias_tbt_guard_s:
                logger.warning(
                    "prefill_bias_tbt_guard_s differs from prefill_bias_tbt_slo_s; "
                    "using prefill_bias_tbt_slo_s for the guard."
                )
            self.prefill_bias_tbt_guard_enabled = True

        # VLLM_PREFILL_BIAS_PHASE2_PATCH: TBT guard validation
        if self.prefill_bias_tbt_guard_enabled:
            if not self.prefill_bias_enabled:
                raise ValueError(
                    "prefill_bias_tbt_guard_enabled requires prefill_bias_enabled."
                )
            if self.prefill_bias_tbt_slo_s <= 0.0:
                raise ValueError(
                    "prefill_bias_tbt_guard_enabled requires "
                    "prefill_bias_tbt_slo_s > 0."
                )
            if self.prefill_bias_tbt_safety_margin_s >= self.prefill_bias_tbt_slo_s:
                raise ValueError(
                    "prefill_bias_tbt_safety_margin_s must be less than "
                    "prefill_bias_tbt_slo_s when the guard is enabled."
                )
            if (
                self.prefill_bias_initial_step_time_s
                + self.prefill_bias_tbt_safety_margin_s
                >= self.prefill_bias_tbt_slo_s
            ):
                logger.warning(
                    "prefill bias TBT guard will likely deny all bias activations: "
                    "initial_step_time_s + safety_margin_s >= tbt_slo_s"
                )

        # VLLM_PREFILL_BIAS_PHASE3_PATCH: slot-swap validation.
        phase3_float_fields = (
            ("prefill_bias_swap_slack_threshold_s", self.prefill_bias_swap_slack_threshold_s),
            ("prefill_bias_swap_cooldown_s", self.prefill_bias_swap_cooldown_s),
            ("prefill_bias_swap_failure_backoff_s", self.prefill_bias_swap_failure_backoff_s),
            ("prefill_bias_victim_tbt_margin_s", self.prefill_bias_victim_tbt_margin_s),
        )
        for field_name, value in phase3_float_fields:
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
        if self.prefill_bias_ttft_slo_s is not None and not math.isfinite(
            self.prefill_bias_ttft_slo_s
        ):
            raise ValueError("prefill_bias_ttft_slo_s must be finite when supplied.")
        if (
            self.prefill_bias_max_victim_recompute_tokens is not None
            and self.prefill_bias_max_victim_recompute_tokens <= 0
        ):
            raise ValueError(
                "prefill_bias_max_victim_recompute_tokens must be positive when supplied."
            )
        if self.prefill_bias_slot_swap_enabled:
            if not self.prefill_bias_enabled:
                raise ValueError(
                    "prefill_bias_slot_swap_enabled requires prefill_bias_enabled."
                )
            if not self.prefill_bias_cache_aware:
                raise ValueError(
                    "prefill_bias_slot_swap_enabled requires prefill_bias_cache_aware."
                )
            if not self.prefill_bias_tbt_guard_enabled:
                raise ValueError(
                    "prefill_bias_slot_swap_enabled requires prefill_bias_tbt_guard_enabled."
                )
            if self.prefill_bias_ttft_slo_s is None or self.prefill_bias_ttft_slo_s <= 0.0:
                raise ValueError(
                    "prefill_bias_slot_swap_enabled requires prefill_bias_ttft_slo_s > 0."
                )
            if self.async_scheduling:
                raise ValueError(
                    "prefill_bias_slot_swap_enabled is disabled for async_scheduling "
                    "in this conservative Phase 3 implementation."
                )

        # VLLM_PREFILL_BIAS_PHASE4_PATCH: adaptive controller validation.
        if self.adaptive_prefill_controller_enabled:
            if not self.prefill_bias_enabled:
                raise ValueError(
                    "adaptive_prefill_controller_enabled requires prefill_bias_enabled."
                )
            if self.adaptive_prefill_ttft_slo_s <= 0.0:
                raise ValueError(
                    "adaptive_prefill_controller_enabled requires "
                    "adaptive_prefill_ttft_slo_s > 0."
                )
            if self.adaptive_prefill_tbt_slo_s <= 0.0:
                raise ValueError(
                    "adaptive_prefill_controller_enabled requires "
                    "adaptive_prefill_tbt_slo_s > 0."
                )
            if self.adaptive_prefill_max_reserve_tokens < (
                self.adaptive_prefill_min_reserve_tokens
            ):
                raise ValueError(
                    "adaptive_prefill_max_reserve_tokens must be >= "
                    "adaptive_prefill_min_reserve_tokens."
                )
            if self.adaptive_prefill_max_chunk_tokens < (
                self.adaptive_prefill_min_chunk_tokens
            ):
                raise ValueError(
                    "adaptive_prefill_max_chunk_tokens must be >= "
                    "adaptive_prefill_min_chunk_tokens."
                )
            if self.adaptive_prefill_max_reserve_tokens > resolved_max_num_scheduled_tokens:
                raise ValueError(
                    "adaptive_prefill_max_reserve_tokens must be <= resolved "
                    "max_num_scheduled_tokens."
                )

        # VLLM_PREFILL_BIAS_PHASE7_PATCH: deadline-mode validation.
        if self.prefill_bias_ttft_deadline_enabled:
            if not self.prefill_bias_enabled:
                raise ValueError(
                    "prefill_bias_ttft_deadline_enabled requires prefill_bias_enabled."
                )
            if not self.prefill_bias_cache_aware:
                raise ValueError(
                    "prefill_bias_ttft_deadline_enabled requires "
                    "prefill_bias_cache_aware."
                )
            if self.prefill_bias_ttft_slo_s is None or self.prefill_bias_ttft_slo_s <= 0.0:
                raise ValueError(
                    "prefill_bias_ttft_deadline_enabled requires "
                    "prefill_bias_ttft_slo_s > 0."
                )

        if self.prefill_bias_ttft_force_preempt_enabled:
            if not self.prefill_bias_ttft_deadline_enabled:
                raise ValueError(
                    "prefill_bias_ttft_force_preempt_enabled requires "
                    "prefill_bias_ttft_deadline_enabled."
                )
            if not self.prefill_bias_slot_swap_enabled:
                raise ValueError(
                    "prefill_bias_ttft_force_preempt_enabled requires "
                    "prefill_bias_slot_swap_enabled."
                )
            if not self.prefill_bias_tbt_guard_enabled:
                raise ValueError(
                    "prefill_bias_ttft_force_preempt_enabled requires "
                    "prefill_bias_tbt_guard_enabled."
                )
            if self.async_scheduling:
                raise ValueError(
                    "prefill_bias_ttft_force_preempt_enabled is disabled for "
                    "async_scheduling."
                )

        # VLLM_PREFILL_BIAS_PHASE8_PATCH: predictive policy validation.
        predictive_modes = {"predictive-shadow", "predictive"}
        batch_budget_modes = {"batch-budget-shadow", "batch-budget"}
        if self.prefill_bias_policy_mode not in {
            "legacy",
            "predictive-shadow",
            "predictive",
            "batch-budget-shadow",
            "batch-budget",
        }:
            raise ValueError(
                "prefill_bias_policy_mode must be legacy, predictive-shadow, "
                "predictive, batch-budget-shadow, or batch-budget."
            )
        if self.prefill_bias_policy_mode in predictive_modes:
            if (
                self.prefill_bias_predictive_max_reserve_tokens
                > resolved_max_num_scheduled_tokens
            ):
                raise ValueError(
                    "prefill_bias_predictive_max_reserve_tokens must be <= resolved "
                    "max_num_scheduled_tokens."
                )
            if (
                self.prefill_bias_predictive_max_reserve_tokens > 0
                and self.prefill_bias_predictive_min_chunk_tokens
                > self.prefill_bias_predictive_max_reserve_tokens
            ):
                raise ValueError(
                    "prefill_bias_predictive_min_chunk_tokens must be <= "
                    "prefill_bias_predictive_max_reserve_tokens."
                )
            if self.prefill_bias_predictive_max_requests_per_step > self.max_num_seqs:
                raise ValueError(
                    "prefill_bias_predictive_max_requests_per_step must be <= "
                    "max_num_seqs."
                )
            if not self.prefill_bias_enabled:
                raise ValueError(
                    "predictive prefill bias requires prefill_bias_enabled."
                )
            if not self.prefill_bias_cache_aware:
                raise ValueError(
                    "predictive prefill bias requires prefill_bias_cache_aware."
                )
            if not self.prefill_bias_ttft_deadline_enabled:
                raise ValueError(
                    "predictive prefill bias requires prefill_bias_ttft_deadline_enabled."
                )
            if not self.prefill_bias_tbt_guard_enabled:
                raise ValueError(
                    "predictive prefill bias requires prefill_bias_tbt_guard_enabled."
                )
            if self.adaptive_prefill_controller_enabled:
                raise ValueError(
                    "adaptive_prefill_controller_enabled is incompatible with "
                    "predictive prefill bias modes."
                )

        # VLLM_PREFILL_BIAS_PHASE9_PATCH: batch-budget validation.
        if self.prefill_bias_policy_mode in batch_budget_modes:
            if not self.prefill_bias_enabled:
                raise ValueError("batch-budget mode requires prefill_bias_enabled.")
            if not self.prefill_bias_cache_aware:
                raise ValueError("batch-budget mode requires prefill_bias_cache_aware.")
            if not self.prefill_bias_ttft_deadline_enabled:
                raise ValueError(
                    "batch-budget mode requires prefill_bias_ttft_deadline_enabled."
                )
            if not self.prefill_bias_tbt_guard_enabled:
                raise ValueError(
                    "batch-budget mode requires prefill_bias_tbt_guard_enabled."
                )
            if self.adaptive_prefill_controller_enabled:
                raise ValueError(
                    "adaptive_prefill_controller_enabled is incompatible with "
                    "batch-budget modes."
                )
            if self.async_scheduling:
                raise ValueError("batch-budget modes require synchronous scheduling.")
            if self.prefill_bias_batch_max_prefill_tokens > resolved_max_num_scheduled_tokens:
                raise ValueError(
                    "prefill_bias_batch_max_prefill_tokens must be <= resolved "
                    "max_num_scheduled_tokens."
                )
            if (
                self.prefill_bias_batch_max_prefill_tokens > 0
                and self.prefill_bias_batch_min_prefill_tokens
                > self.prefill_bias_batch_max_prefill_tokens
            ):
                raise ValueError(
                    "prefill_bias_batch_min_prefill_tokens must be <= "
                    "prefill_bias_batch_max_prefill_tokens."
                )
            if self.prefill_bias_batch_max_requests_per_step > self.max_num_seqs:
                raise ValueError(
                    "prefill_bias_batch_max_requests_per_step must be <= max_num_seqs."
                )

        self.verify_max_model_len(max_model_len)

    def verify_max_model_len(self, max_model_len: int) -> Self:
        if (
            self.max_num_batched_tokens < max_model_len
            and not self.enable_chunked_prefill
        ):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len."
            )

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs})."
            )

        if self.max_num_batched_tokens > self.max_num_seqs * max_model_len:
            logger.warning(
                "max_num_batched_tokens (%d) exceeds max_num_seqs "
                "* max_model_len (%d). This may lead to unexpected behavior.",
                self.max_num_batched_tokens,
                self.max_num_seqs * max_model_len,
            )

        if self.max_num_partial_prefills > 1:
            if not self.enable_chunked_prefill:
                raise ValueError(
                    "Chunked prefill must be enabled to set "
                    "max_num_partial_prefills > 1."
                )

            if self.long_prefill_token_threshold > max_model_len:
                raise ValueError(
                    "long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) cannot be greater "
                    f"than the max_model_len ({max_model_len})."
                )

        if self.max_long_partial_prefills > self.max_num_partial_prefills:
            raise ValueError(
                f"{self.max_long_partial_prefills=} must be less than or equal to "
                f"{self.max_num_partial_prefills=}."
            )

        return self

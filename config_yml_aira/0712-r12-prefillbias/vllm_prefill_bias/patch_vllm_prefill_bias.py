#!/usr/bin/env python3
"""Patch vLLM 0.24.0 with experimental adaptive prefill bias."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import py_compile
import shutil
from pathlib import Path


PATCH_MARKER = "# VLLM_PREFILL_BIAS_PATCH"
PHASE2_MARKER = "# VLLM_PREFILL_BIAS_PHASE2_PATCH"
PHASE3_MARKER = "# VLLM_PREFILL_BIAS_PHASE3_PATCH"
PHASE4_MARKER = "# VLLM_PREFILL_BIAS_PHASE4_PATCH"
PHASE5_MARKER = "# VLLM_PREFILL_BIAS_PHASE5_PATCH"
PHASE6_MARKER = "# VLLM_PREFILL_BIAS_PHASE6_PATCH"
PHASE7_MARKER = "# VLLM_PREFILL_BIAS_PHASE7_PATCH"
PHASE8_MARKER = "# VLLM_PREFILL_BIAS_PHASE8_PATCH"
PHASE9_MARKER = "# VLLM_PREFILL_BIAS_PHASE9_PATCH"
EXPECTED_HASHES = {
    "scheduler": "df78076164fe02a63876d7690cefac810f1ba4d8fb58afc15e786c1608064090",
    "scheduler_config": "4102ccd3d73188200b25f47a0cba837853662647850f283012d8cc1f9e8c47c2",
    "arg_utils": "83e4066d3436b23a76ff02ecd46ac5a5e42d083b17ee40aaab2018955468b233",
    "kv_cache_manager": "2e91e77f84dfd9d4bed79f50392be72559207f54e845a3d3402baf88dc529320",
}


class PatchError(RuntimeError):
    pass


def find_targets(vllm_root: Path) -> dict[str, Path]:
    return {
        "scheduler": vllm_root / "vllm" / "v1" / "core" / "sched" / "scheduler.py",
        "scheduler_config": vllm_root / "vllm" / "config" / "scheduler.py",
        "arg_utils": vllm_root / "vllm" / "engine" / "arg_utils.py",
        "kv_cache_manager": vllm_root / "vllm" / "v1" / "core" / "kv_cache_manager.py",
    }


def find_vllm_root(explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root)
        return root if (root / "vllm").is_dir() else root.parent

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise PatchError("Unable to locate installed vllm package")
    return Path(spec.origin).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_targets(targets: dict[str, Path], *, skip_hash_check: bool) -> None:
    for key, path in targets.items():
        if not path.exists():
            raise PatchError(f"Missing target file: {path}")
        text = path.read_text()
        if PATCH_MARKER in text:
            continue
        if skip_hash_check:
            continue
        digest = sha256(path)
        expected = EXPECTED_HASHES[key]
        if digest != expected:
            raise PatchError(
                f"{key} hash mismatch for {path}: got {digest}, expected {expected}. "
                "Refusing to patch a materially different vLLM source."
            )


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(
            f"Expected exactly one anchor for {description}, found {count}."
        )
    return text.replace(old, new, 1)


def write_if_changed(path: Path, text: str) -> bool:
    old = path.read_text()
    if old == text:
        return False
    path.write_text(text)
    return True


def copy_policy_module(vllm_root: Path) -> Path:
    source = Path(__file__).resolve().parent / "patch" / "prefill_bias_vllm.py"
    if not source.exists():
        raise PatchError(f"Missing policy module source: {source}")
    dest = vllm_root / "vllm" / "v1" / "core" / "sched" / "prefill_bias.py"
    shutil.copyfile(source, dest)
    return dest


def patch_scheduler_config(path: Path) -> bool:
    text = path.read_text()
    if PATCH_MARKER in text:
        return False

    fields_anchor = '''    prefill_schedule_interval: int = Field(default=1, ge=1)
    """For data-parallel deployments, only admit new prefill requests
    once every N engine steps, aligned across DP ranks, to better balance
    per-step forward-pass times."""

    async_scheduling: bool | None = None
'''
    fields_new = '''    prefill_schedule_interval: int = Field(default=1, ge=1)
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

    async_scheduling: bool | None = None
'''
    text = replace_once(text, fields_anchor, fields_new, "SchedulerConfig fields")

    validation_anchor = """        self.verify_max_model_len(max_model_len)

    def verify_max_model_len(self, max_model_len: int) -> Self:
"""
    validation_new = """        # VLLM_PREFILL_BIAS_PATCH: scheduler_config validation
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

        self.verify_max_model_len(max_model_len)

    def verify_max_model_len(self, max_model_len: int) -> Self:
"""
    text = replace_once(
        text, validation_anchor, validation_new, "prefill bias validation"
    )
    return write_if_changed(path, text)


def patch_arg_utils(path: Path) -> bool:
    text = path.read_text()
    if PATCH_MARKER in text:
        return False

    fields_anchor = """    scheduler_reserve_full_isl: bool = SchedulerConfig.scheduler_reserve_full_isl
    prefill_schedule_interval: int = SchedulerConfig.prefill_schedule_interval

    watermark: float = SchedulerConfig.watermark
"""
    fields_new = """    scheduler_reserve_full_isl: bool = SchedulerConfig.scheduler_reserve_full_isl
    prefill_schedule_interval: int = SchedulerConfig.prefill_schedule_interval
    # VLLM_PREFILL_BIAS_PATCH: EngineArgs fields
    prefill_bias_enabled: bool = SchedulerConfig.prefill_bias_enabled
    prefill_bias_wait_threshold_s: float = (
        SchedulerConfig.prefill_bias_wait_threshold_s
    )
    prefill_bias_reserve_tokens: int = SchedulerConfig.prefill_bias_reserve_tokens
    prefill_bias_max_requests_per_step: int = (
        SchedulerConfig.prefill_bias_max_requests_per_step
    )
    prefill_bias_cache_aware: bool = SchedulerConfig.prefill_bias_cache_aware
    prefill_bias_score_window_k: int = SchedulerConfig.prefill_bias_score_window_k
    # VLLM_PREFILL_BIAS_PHASE5_PATCH: EngineArgs bounded scan field
    prefill_bias_candidate_scan_limit: int = (
        SchedulerConfig.prefill_bias_candidate_scan_limit
    )
    prefill_bias_min_cached_tokens: int = SchedulerConfig.prefill_bias_min_cached_tokens
    prefill_bias_starvation_s: float = SchedulerConfig.prefill_bias_starvation_s
    prefill_bias_remaining_token_buckets: tuple[int, ...] = (
        SchedulerConfig.prefill_bias_remaining_token_buckets
    )

    watermark: float = SchedulerConfig.watermark
"""
    text = replace_once(text, fields_anchor, fields_new, "EngineArgs fields")

    cli_anchor = """        scheduler_group.add_argument(
            "--prefill-schedule-interval",
            **scheduler_kwargs["prefill_schedule_interval"],
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
            **scheduler_kwargs["disable_hybrid_kv_cache_manager"],
        )
"""
    cli_new = """        scheduler_group.add_argument(
            "--prefill-schedule-interval",
            **scheduler_kwargs["prefill_schedule_interval"],
        )
        # VLLM_PREFILL_BIAS_PATCH: CLI flags
        scheduler_group.add_argument(
            "--prefill-bias-enabled",
            **scheduler_kwargs["prefill_bias_enabled"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-wait-threshold-s",
            **scheduler_kwargs["prefill_bias_wait_threshold_s"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-reserve-tokens",
            **scheduler_kwargs["prefill_bias_reserve_tokens"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-max-requests-per-step",
            **scheduler_kwargs["prefill_bias_max_requests_per_step"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-cache-aware",
            **scheduler_kwargs["prefill_bias_cache_aware"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-score-window-k",
            **scheduler_kwargs["prefill_bias_score_window_k"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-candidate-scan-limit",
            **scheduler_kwargs["prefill_bias_candidate_scan_limit"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-min-cached-tokens",
            **scheduler_kwargs["prefill_bias_min_cached_tokens"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-starvation-s",
            **scheduler_kwargs["prefill_bias_starvation_s"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-remaining-token-buckets",
            **scheduler_kwargs["prefill_bias_remaining_token_buckets"],
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
            **scheduler_kwargs["disable_hybrid_kv_cache_manager"],
        )
"""
    text = replace_once(text, cli_anchor, cli_new, "CLI flags")

    ctor_anchor = """            scheduler_reserve_full_isl=self.scheduler_reserve_full_isl,
            watermark=self.watermark,
            prefill_schedule_interval=self.prefill_schedule_interval,
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
"""
    ctor_new = """            scheduler_reserve_full_isl=self.scheduler_reserve_full_isl,
            watermark=self.watermark,
            prefill_schedule_interval=self.prefill_schedule_interval,
            prefill_bias_enabled=self.prefill_bias_enabled,
            prefill_bias_wait_threshold_s=self.prefill_bias_wait_threshold_s,
            prefill_bias_reserve_tokens=self.prefill_bias_reserve_tokens,
            prefill_bias_max_requests_per_step=self.prefill_bias_max_requests_per_step,
            prefill_bias_cache_aware=self.prefill_bias_cache_aware,
            prefill_bias_score_window_k=self.prefill_bias_score_window_k,
            prefill_bias_candidate_scan_limit=self.prefill_bias_candidate_scan_limit,
            prefill_bias_min_cached_tokens=self.prefill_bias_min_cached_tokens,
            prefill_bias_starvation_s=self.prefill_bias_starvation_s,
            prefill_bias_remaining_token_buckets=self.prefill_bias_remaining_token_buckets,
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
"""
    text = replace_once(text, ctor_anchor, ctor_new, "SchedulerConfig propagation")
    return write_if_changed(path, text)


def patch_kv_cache_manager(path: Path) -> bool:
    text = path.read_text()
    if PATCH_MARKER in text:
        return False

    method_anchor = '''    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
'''
    method_new = '''    # VLLM_PREFILL_BIAS_PATCH: read-only prefix-cache probe for scoring.
    def _find_computed_blocks(
        self,
        request: Request,
        *,
        record_stats: bool,
    ) -> tuple[KVCacheBlocks, int]:
        """Find locally reusable prefix-cache tokens for a request.

        When record_stats is False, this method is observationally read-only
        with respect to prefix-cache metrics and scheduler/request state.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        max_cache_hit_length = request.num_tokens - 1
        previous_common_prefix = getattr(
            self.coordinator,
            "num_uncached_common_prefix_tokens",
            None,
        )
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
        if not record_stats and previous_common_prefix is not None:
            self.coordinator.num_uncached_common_prefix_tokens = previous_common_prefix

        if record_stats and self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.
        """
        return self._find_computed_blocks(request, record_stats=True)

    def peek_num_computed_tokens(self, request: Request) -> int:
        """Return locally reusable prefix-cache tokens without recording stats."""
        _, num_new_computed_tokens = self._find_computed_blocks(
            request,
            record_stats=False,
        )
        return num_new_computed_tokens

    def allocate_slots(
'''
    text = replace_once(text, method_anchor, method_new, "KV cache read-only peek")
    return write_if_changed(path, text)


def patch_scheduler_config_phase2(path: Path) -> bool:
    text = path.read_text()
    if PHASE2_MARKER in text:
        return False

    fields_anchor = '''    prefill_bias_remaining_token_buckets: tuple[int, ...] = (16, 64, 256, 1024)
    """Bucket edges for remaining prefill work in cache-aware ordering."""

    async_scheduling: bool | None = None
'''
    fields_new = '''    prefill_bias_remaining_token_buckets: tuple[int, ...] = (16, 64, 256, 1024)
    """Bucket edges for remaining prefill work in cache-aware ordering."""

    # VLLM_PREFILL_BIAS_PHASE2_PATCH: TBT guard config fields
    prefill_bias_tbt_guard_enabled: bool = False
    """Enable conservative TBT/ITL slack guard for prefill bias."""

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

    async_scheduling: bool | None = None
'''
    text = replace_once(
        text, fields_anchor, fields_new, "Phase 2 SchedulerConfig fields"
    )

    validation_anchor = """        if tuple(sorted(bucket_edges)) != bucket_edges or len(set(bucket_edges)) != len(
            bucket_edges
        ):
            raise ValueError(
                "prefill_bias_remaining_token_buckets must be strictly increasing "
                "and unique."
            )

        self.verify_max_model_len(max_model_len)
"""
    validation_new = """        if tuple(sorted(bucket_edges)) != bucket_edges or len(set(bucket_edges)) != len(
            bucket_edges
        ):
            raise ValueError(
                "prefill_bias_remaining_token_buckets must be strictly increasing "
                "and unique."
            )

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

        self.verify_max_model_len(max_model_len)
"""
    text = replace_once(text, validation_anchor, validation_new, "Phase 2 validation")
    return write_if_changed(path, text)


def patch_arg_utils_phase2(path: Path) -> bool:
    text = path.read_text()
    if PHASE2_MARKER in text:
        return False

    fields_anchor = """    prefill_bias_remaining_token_buckets: tuple[int, ...] = (
        SchedulerConfig.prefill_bias_remaining_token_buckets
    )

    watermark: float = SchedulerConfig.watermark
"""
    fields_new = """    prefill_bias_remaining_token_buckets: tuple[int, ...] = (
        SchedulerConfig.prefill_bias_remaining_token_buckets
    )
    # VLLM_PREFILL_BIAS_PHASE2_PATCH: EngineArgs TBT guard fields
    prefill_bias_tbt_guard_enabled: bool = (
        SchedulerConfig.prefill_bias_tbt_guard_enabled
    )
    prefill_bias_tbt_slo_s: float = SchedulerConfig.prefill_bias_tbt_slo_s
    prefill_bias_tbt_safety_margin_s: float = (
        SchedulerConfig.prefill_bias_tbt_safety_margin_s
    )
    prefill_bias_initial_step_time_s: float = (
        SchedulerConfig.prefill_bias_initial_step_time_s
    )
    prefill_bias_step_time_ewma_alpha: float = (
        SchedulerConfig.prefill_bias_step_time_ewma_alpha
    )
    prefill_bias_step_time_headroom_factor: float = (
        SchedulerConfig.prefill_bias_step_time_headroom_factor
    )
    prefill_bias_guard_unknown_decode: bool = (
        SchedulerConfig.prefill_bias_guard_unknown_decode
    )
    prefill_bias_step_observation_min_samples: int = (
        SchedulerConfig.prefill_bias_step_observation_min_samples
    )

    watermark: float = SchedulerConfig.watermark
"""
    text = replace_once(text, fields_anchor, fields_new, "Phase 2 EngineArgs fields")

    cli_anchor = """        scheduler_group.add_argument(
            "--prefill-bias-remaining-token-buckets",
            **scheduler_kwargs["prefill_bias_remaining_token_buckets"],
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
"""
    cli_new = """        scheduler_group.add_argument(
            "--prefill-bias-remaining-token-buckets",
            **scheduler_kwargs["prefill_bias_remaining_token_buckets"],
        )
        # VLLM_PREFILL_BIAS_PHASE2_PATCH: CLI TBT guard flags
        scheduler_group.add_argument(
            "--prefill-bias-tbt-guard-enabled",
            **scheduler_kwargs["prefill_bias_tbt_guard_enabled"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-tbt-slo-s",
            **scheduler_kwargs["prefill_bias_tbt_slo_s"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-tbt-safety-margin-s",
            **scheduler_kwargs["prefill_bias_tbt_safety_margin_s"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-initial-step-time-s",
            **scheduler_kwargs["prefill_bias_initial_step_time_s"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-step-time-ewma-alpha",
            **scheduler_kwargs["prefill_bias_step_time_ewma_alpha"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-step-time-headroom-factor",
            **scheduler_kwargs["prefill_bias_step_time_headroom_factor"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-guard-unknown-decode",
            **scheduler_kwargs["prefill_bias_guard_unknown_decode"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-step-observation-min-samples",
            **scheduler_kwargs["prefill_bias_step_observation_min_samples"],
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
"""
    text = replace_once(text, cli_anchor, cli_new, "Phase 2 CLI flags")

    ctor_anchor = """            prefill_bias_starvation_s=self.prefill_bias_starvation_s,
            prefill_bias_remaining_token_buckets=self.prefill_bias_remaining_token_buckets,
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
"""
    ctor_new = """            prefill_bias_starvation_s=self.prefill_bias_starvation_s,
            prefill_bias_remaining_token_buckets=self.prefill_bias_remaining_token_buckets,
            prefill_bias_tbt_guard_enabled=self.prefill_bias_tbt_guard_enabled,
            prefill_bias_tbt_slo_s=self.prefill_bias_tbt_slo_s,
            prefill_bias_tbt_safety_margin_s=self.prefill_bias_tbt_safety_margin_s,
            prefill_bias_initial_step_time_s=self.prefill_bias_initial_step_time_s,
            prefill_bias_step_time_ewma_alpha=self.prefill_bias_step_time_ewma_alpha,
            prefill_bias_step_time_headroom_factor=(
                self.prefill_bias_step_time_headroom_factor
            ),
            prefill_bias_guard_unknown_decode=self.prefill_bias_guard_unknown_decode,
            prefill_bias_step_observation_min_samples=(
                self.prefill_bias_step_observation_min_samples
            ),
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
"""
    text = replace_once(
        text, ctor_anchor, ctor_new, "Phase 2 SchedulerConfig propagation"
    )
    return write_if_changed(path, text)


def patch_scheduler_config_phase3(path: Path) -> bool:
    text = path.read_text()
    if PHASE3_MARKER in text:
        return False

    text = replace_once(
        text,
        "from collections.abc import Callable\n",
        "import math\nfrom collections.abc import Callable\n",
        "Phase 3 math import",
    )

    fields_anchor = '''    prefill_bias_step_observation_min_samples: int = Field(default=3, ge=1)
    """Minimum prefill-containing batch observations before trusting EWMA."""

    async_scheduling: bool | None = None
'''
    fields_new = '''    prefill_bias_step_observation_min_samples: int = Field(default=3, ge=1)
    """Minimum prefill-containing batch observations before trusting EWMA."""

    # VLLM_PREFILL_BIAS_PHASE3_PATCH: conservative slot-swap config fields
    prefill_bias_slot_swap_enabled: bool = False
    """Enable conservative Phase 3 prefill admission swap."""

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

    async_scheduling: bool | None = None
'''
    text = replace_once(
        text, fields_anchor, fields_new, "Phase 3 SchedulerConfig fields"
    )

    validation_anchor = """        if self.prefill_bias_tbt_guard_enabled:
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

        self.verify_max_model_len(max_model_len)
"""
    validation_new = """        if self.prefill_bias_tbt_guard_enabled:
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

        self.verify_max_model_len(max_model_len)
"""
    text = replace_once(text, validation_anchor, validation_new, "Phase 3 validation")
    return write_if_changed(path, text)


def patch_arg_utils_phase3(path: Path) -> bool:
    text = path.read_text()
    if PHASE3_MARKER in text:
        return False

    fields_anchor = """    prefill_bias_step_observation_min_samples: int = (
        SchedulerConfig.prefill_bias_step_observation_min_samples
    )

    watermark: float = SchedulerConfig.watermark
"""
    fields_new = """    prefill_bias_step_observation_min_samples: int = (
        SchedulerConfig.prefill_bias_step_observation_min_samples
    )
    # VLLM_PREFILL_BIAS_PHASE3_PATCH: EngineArgs slot-swap fields
    prefill_bias_slot_swap_enabled: bool = (
        SchedulerConfig.prefill_bias_slot_swap_enabled
    )
    prefill_bias_ttft_slo_s: float | None = SchedulerConfig.prefill_bias_ttft_slo_s
    prefill_bias_swap_slack_threshold_s: float = (
        SchedulerConfig.prefill_bias_swap_slack_threshold_s
    )
    prefill_bias_max_swaps_per_step: int = (
        SchedulerConfig.prefill_bias_max_swaps_per_step
    )
    prefill_bias_max_preemptions_per_request: int = (
        SchedulerConfig.prefill_bias_max_preemptions_per_request
    )
    prefill_bias_swap_cooldown_s: float = SchedulerConfig.prefill_bias_swap_cooldown_s
    prefill_bias_swap_failure_backoff_s: float = (
        SchedulerConfig.prefill_bias_swap_failure_backoff_s
    )
    prefill_bias_max_candidate_remaining_tokens: int = (
        SchedulerConfig.prefill_bias_max_candidate_remaining_tokens
    )
    prefill_bias_max_victim_recompute_tokens: int | None = (
        SchedulerConfig.prefill_bias_max_victim_recompute_tokens
    )
    prefill_bias_victim_tbt_margin_s: float = (
        SchedulerConfig.prefill_bias_victim_tbt_margin_s
    )
    prefill_bias_require_cache_residency: bool = (
        SchedulerConfig.prefill_bias_require_cache_residency
    )

    watermark: float = SchedulerConfig.watermark
"""
    text = replace_once(text, fields_anchor, fields_new, "Phase 3 EngineArgs fields")

    cli_anchor = """        scheduler_group.add_argument(
            "--prefill-bias-step-observation-min-samples",
            **scheduler_kwargs["prefill_bias_step_observation_min_samples"],
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
"""
    cli_new = """        scheduler_group.add_argument(
            "--prefill-bias-step-observation-min-samples",
            **scheduler_kwargs["prefill_bias_step_observation_min_samples"],
        )
        # VLLM_PREFILL_BIAS_PHASE3_PATCH: CLI slot-swap flags
        scheduler_group.add_argument(
            "--prefill-bias-slot-swap-enabled",
            **scheduler_kwargs["prefill_bias_slot_swap_enabled"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-ttft-slo-s",
            **scheduler_kwargs["prefill_bias_ttft_slo_s"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-swap-slack-threshold-s",
            **scheduler_kwargs["prefill_bias_swap_slack_threshold_s"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-max-swaps-per-step",
            **scheduler_kwargs["prefill_bias_max_swaps_per_step"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-max-preemptions-per-request",
            **scheduler_kwargs["prefill_bias_max_preemptions_per_request"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-swap-cooldown-s",
            **scheduler_kwargs["prefill_bias_swap_cooldown_s"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-swap-failure-backoff-s",
            **scheduler_kwargs["prefill_bias_swap_failure_backoff_s"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-max-candidate-remaining-tokens",
            **scheduler_kwargs["prefill_bias_max_candidate_remaining_tokens"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-max-victim-recompute-tokens",
            **scheduler_kwargs["prefill_bias_max_victim_recompute_tokens"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-victim-tbt-margin-s",
            **scheduler_kwargs["prefill_bias_victim_tbt_margin_s"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-require-cache-residency",
            **scheduler_kwargs["prefill_bias_require_cache_residency"],
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
"""
    text = replace_once(text, cli_anchor, cli_new, "Phase 3 CLI flags")

    ctor_anchor = """            prefill_bias_step_observation_min_samples=(
                self.prefill_bias_step_observation_min_samples
            ),
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
"""
    ctor_new = """            prefill_bias_step_observation_min_samples=(
                self.prefill_bias_step_observation_min_samples
            ),
            prefill_bias_slot_swap_enabled=self.prefill_bias_slot_swap_enabled,
            prefill_bias_ttft_slo_s=self.prefill_bias_ttft_slo_s,
            prefill_bias_swap_slack_threshold_s=self.prefill_bias_swap_slack_threshold_s,
            prefill_bias_max_swaps_per_step=self.prefill_bias_max_swaps_per_step,
            prefill_bias_max_preemptions_per_request=(
                self.prefill_bias_max_preemptions_per_request
            ),
            prefill_bias_swap_cooldown_s=self.prefill_bias_swap_cooldown_s,
            prefill_bias_swap_failure_backoff_s=(
                self.prefill_bias_swap_failure_backoff_s
            ),
            prefill_bias_max_candidate_remaining_tokens=(
                self.prefill_bias_max_candidate_remaining_tokens
            ),
            prefill_bias_max_victim_recompute_tokens=(
                self.prefill_bias_max_victim_recompute_tokens
            ),
            prefill_bias_victim_tbt_margin_s=self.prefill_bias_victim_tbt_margin_s,
            prefill_bias_require_cache_residency=(
                self.prefill_bias_require_cache_residency
            ),
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
"""
    text = replace_once(
        text, ctor_anchor, ctor_new, "Phase 3 SchedulerConfig propagation"
    )
    return write_if_changed(path, text)


def patch_scheduler_config_phase4(path: Path) -> bool:
    text = path.read_text()
    if PHASE4_MARKER in text:
        return False

    text = replace_once(
        text,
        "    async_scheduling: bool | None = None\n",
        '''    # VLLM_PREFILL_BIAS_PHASE4_PATCH: SLO/goodput adaptive controller fields
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
''',
        "Phase 4 SchedulerConfig fields",
    )
    text = replace_once(
        text,
        "        self.verify_max_model_len(max_model_len)\n",
        """        # VLLM_PREFILL_BIAS_PHASE4_PATCH: adaptive controller validation.
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

        self.verify_max_model_len(max_model_len)
""",
        "Phase 4 validation",
    )
    return write_if_changed(path, text)


def patch_arg_utils_phase4(path: Path) -> bool:
    text = path.read_text()
    if PHASE4_MARKER in text:
        return False

    fields_anchor = """    prefill_bias_require_cache_residency: bool = (
        SchedulerConfig.prefill_bias_require_cache_residency
    )

    watermark: float = SchedulerConfig.watermark
"""
    fields_new = """    prefill_bias_require_cache_residency: bool = (
        SchedulerConfig.prefill_bias_require_cache_residency
    )
    # VLLM_PREFILL_BIAS_PHASE4_PATCH: EngineArgs adaptive-controller fields
    adaptive_prefill_controller_enabled: bool = (
        SchedulerConfig.adaptive_prefill_controller_enabled
    )
    adaptive_prefill_ttft_slo_s: float = SchedulerConfig.adaptive_prefill_ttft_slo_s
    adaptive_prefill_tbt_slo_s: float = SchedulerConfig.adaptive_prefill_tbt_slo_s
    adaptive_prefill_target_ttft_attainment: float = (
        SchedulerConfig.adaptive_prefill_target_ttft_attainment
    )
    adaptive_prefill_target_tbt_attainment: float = (
        SchedulerConfig.adaptive_prefill_target_tbt_attainment
    )
    adaptive_prefill_target_joint_attainment: float = (
        SchedulerConfig.adaptive_prefill_target_joint_attainment
    )
    adaptive_prefill_control_interval_s: float = (
        SchedulerConfig.adaptive_prefill_control_interval_s
    )
    adaptive_prefill_window_s: float = SchedulerConfig.adaptive_prefill_window_s
    adaptive_prefill_min_samples: int = SchedulerConfig.adaptive_prefill_min_samples
    adaptive_prefill_ema_alpha: float = SchedulerConfig.adaptive_prefill_ema_alpha
    adaptive_prefill_enter_epochs: int = SchedulerConfig.adaptive_prefill_enter_epochs
    adaptive_prefill_exit_epochs: int = SchedulerConfig.adaptive_prefill_exit_epochs
    adaptive_prefill_cooldown_s: float = SchedulerConfig.adaptive_prefill_cooldown_s
    adaptive_prefill_max_level_step: int = SchedulerConfig.adaptive_prefill_max_level_step
    adaptive_prefill_overload_epochs: int = SchedulerConfig.adaptive_prefill_overload_epochs
    adaptive_prefill_min_reserve_tokens: int = (
        SchedulerConfig.adaptive_prefill_min_reserve_tokens
    )
    adaptive_prefill_max_reserve_tokens: int = (
        SchedulerConfig.adaptive_prefill_max_reserve_tokens
    )
    adaptive_prefill_min_chunk_tokens: int = (
        SchedulerConfig.adaptive_prefill_min_chunk_tokens
    )
    adaptive_prefill_max_chunk_tokens: int = (
        SchedulerConfig.adaptive_prefill_max_chunk_tokens
    )
    adaptive_prefill_max_swaps_per_epoch: int = (
        SchedulerConfig.adaptive_prefill_max_swaps_per_epoch
    )
    adaptive_prefill_tbt_emergency_ratio: float = (
        SchedulerConfig.adaptive_prefill_tbt_emergency_ratio
    )
    adaptive_prefill_max_swap_failures_per_window: int = (
        SchedulerConfig.adaptive_prefill_max_swap_failures_per_window
    )
    adaptive_prefill_max_recompute_tokens_per_window: int = (
        SchedulerConfig.adaptive_prefill_max_recompute_tokens_per_window
    )

    watermark: float = SchedulerConfig.watermark
"""
    text = replace_once(text, fields_anchor, fields_new, "Phase 4 EngineArgs fields")

    text = replace_once(
        text,
        """        scheduler_group.add_argument(
            "--prefill-bias-require-cache-residency",
            **scheduler_kwargs["prefill_bias_require_cache_residency"],
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
""",
        """        scheduler_group.add_argument(
            "--prefill-bias-require-cache-residency",
            **scheduler_kwargs["prefill_bias_require_cache_residency"],
        )
        # VLLM_PREFILL_BIAS_PHASE4_PATCH: CLI adaptive-controller flags
        for name in (
            "adaptive_prefill_controller_enabled",
            "adaptive_prefill_ttft_slo_s",
            "adaptive_prefill_tbt_slo_s",
            "adaptive_prefill_target_ttft_attainment",
            "adaptive_prefill_target_tbt_attainment",
            "adaptive_prefill_target_joint_attainment",
            "adaptive_prefill_control_interval_s",
            "adaptive_prefill_window_s",
            "adaptive_prefill_min_samples",
            "adaptive_prefill_ema_alpha",
            "adaptive_prefill_enter_epochs",
            "adaptive_prefill_exit_epochs",
            "adaptive_prefill_cooldown_s",
            "adaptive_prefill_max_level_step",
            "adaptive_prefill_overload_epochs",
            "adaptive_prefill_min_reserve_tokens",
            "adaptive_prefill_max_reserve_tokens",
            "adaptive_prefill_min_chunk_tokens",
            "adaptive_prefill_max_chunk_tokens",
            "adaptive_prefill_max_swaps_per_epoch",
            "adaptive_prefill_tbt_emergency_ratio",
            "adaptive_prefill_max_swap_failures_per_window",
            "adaptive_prefill_max_recompute_tokens_per_window",
        ):
            scheduler_group.add_argument(
                "--" + name.replace("_", "-"),
                **scheduler_kwargs[name],
            )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
""",
        "Phase 4 CLI flags",
    )

    text = replace_once(
        text,
        """            prefill_bias_require_cache_residency=(
                self.prefill_bias_require_cache_residency
            ),
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
""",
        """            prefill_bias_require_cache_residency=(
                self.prefill_bias_require_cache_residency
            ),
            adaptive_prefill_controller_enabled=self.adaptive_prefill_controller_enabled,
            adaptive_prefill_ttft_slo_s=self.adaptive_prefill_ttft_slo_s,
            adaptive_prefill_tbt_slo_s=self.adaptive_prefill_tbt_slo_s,
            adaptive_prefill_target_ttft_attainment=(
                self.adaptive_prefill_target_ttft_attainment
            ),
            adaptive_prefill_target_tbt_attainment=(
                self.adaptive_prefill_target_tbt_attainment
            ),
            adaptive_prefill_target_joint_attainment=(
                self.adaptive_prefill_target_joint_attainment
            ),
            adaptive_prefill_control_interval_s=self.adaptive_prefill_control_interval_s,
            adaptive_prefill_window_s=self.adaptive_prefill_window_s,
            adaptive_prefill_min_samples=self.adaptive_prefill_min_samples,
            adaptive_prefill_ema_alpha=self.adaptive_prefill_ema_alpha,
            adaptive_prefill_enter_epochs=self.adaptive_prefill_enter_epochs,
            adaptive_prefill_exit_epochs=self.adaptive_prefill_exit_epochs,
            adaptive_prefill_cooldown_s=self.adaptive_prefill_cooldown_s,
            adaptive_prefill_max_level_step=self.adaptive_prefill_max_level_step,
            adaptive_prefill_overload_epochs=self.adaptive_prefill_overload_epochs,
            adaptive_prefill_min_reserve_tokens=self.adaptive_prefill_min_reserve_tokens,
            adaptive_prefill_max_reserve_tokens=self.adaptive_prefill_max_reserve_tokens,
            adaptive_prefill_min_chunk_tokens=self.adaptive_prefill_min_chunk_tokens,
            adaptive_prefill_max_chunk_tokens=self.adaptive_prefill_max_chunk_tokens,
            adaptive_prefill_max_swaps_per_epoch=(
                self.adaptive_prefill_max_swaps_per_epoch
            ),
            adaptive_prefill_tbt_emergency_ratio=(
                self.adaptive_prefill_tbt_emergency_ratio
            ),
            adaptive_prefill_max_swap_failures_per_window=(
                self.adaptive_prefill_max_swap_failures_per_window
            ),
            adaptive_prefill_max_recompute_tokens_per_window=(
                self.adaptive_prefill_max_recompute_tokens_per_window
            ),
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
""",
        "Phase 4 SchedulerConfig propagation",
    )
    return write_if_changed(path, text)


def patch_scheduler_config_phase5(path: Path) -> bool:
    text = path.read_text()
    if PHASE5_MARKER in text:
        return False

    if "prefill_bias_candidate_scan_limit" not in text:
        text = replace_once(
            text,
            '''    prefill_bias_score_window_k: int = Field(default=16, ge=1)
    """Score at most this many eligible non-sticky waiting prefills."""

    prefill_bias_min_cached_tokens: int = Field(default=0, ge=0)
''',
            '''    prefill_bias_score_window_k: int = Field(default=16, ge=1)
    """Score at most this many eligible non-sticky waiting prefills."""

    # VLLM_PREFILL_BIAS_PHASE5_PATCH: bounded cache-aware candidate inspection.
    prefill_bias_candidate_scan_limit: int = Field(default=16, ge=1)
    """Inspect at most this many normal waiting prefills for cache-aware scoring."""

    prefill_bias_min_cached_tokens: int = Field(default=0, ge=0)
''',
            "Phase 5 SchedulerConfig field",
        )
    else:
        text = text.replace(
            "    prefill_bias_candidate_scan_limit: int = Field(default=16, ge=1)\n",
            "    # VLLM_PREFILL_BIAS_PHASE5_PATCH: bounded cache-aware candidate inspection.\n"
            "    prefill_bias_candidate_scan_limit: int = Field(default=16, ge=1)\n",
            1,
        )

    if "prefill_bias_candidate_scan_limit must be greater" not in text:
        text = replace_once(
            text,
            """        if self.prefill_bias_max_requests_per_step > self.max_num_seqs:
            raise ValueError(
                "prefill_bias_max_requests_per_step "
                f"({self.prefill_bias_max_requests_per_step}) must be less than "
                f"or equal to max_num_seqs ({self.max_num_seqs})."
            )
""",
            """        if self.prefill_bias_max_requests_per_step > self.max_num_seqs:
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
""",
            "Phase 5 SchedulerConfig validation",
        )
    return write_if_changed(path, text)


def patch_arg_utils_phase5(path: Path) -> bool:
    text = path.read_text()
    if PHASE5_MARKER in text:
        return False

    if "prefill_bias_candidate_scan_limit" not in text:
        text = replace_once(
            text,
            """    prefill_bias_cache_aware: bool = SchedulerConfig.prefill_bias_cache_aware
    prefill_bias_score_window_k: int = SchedulerConfig.prefill_bias_score_window_k
    prefill_bias_min_cached_tokens: int = SchedulerConfig.prefill_bias_min_cached_tokens
""",
            """    prefill_bias_cache_aware: bool = SchedulerConfig.prefill_bias_cache_aware
    prefill_bias_score_window_k: int = SchedulerConfig.prefill_bias_score_window_k
    # VLLM_PREFILL_BIAS_PHASE5_PATCH: EngineArgs bounded scan field
    prefill_bias_candidate_scan_limit: int = (
        SchedulerConfig.prefill_bias_candidate_scan_limit
    )
    prefill_bias_min_cached_tokens: int = SchedulerConfig.prefill_bias_min_cached_tokens
""",
            "Phase 5 EngineArgs field",
        )
    else:
        text = text.replace(
            "    prefill_bias_candidate_scan_limit: int = (\n",
            "    # VLLM_PREFILL_BIAS_PHASE5_PATCH: EngineArgs bounded scan field\n"
            "    prefill_bias_candidate_scan_limit: int = (\n",
            1,
        )

    if "--prefill-bias-candidate-scan-limit" not in text:
        text = replace_once(
            text,
            """        scheduler_group.add_argument(
            "--prefill-bias-score-window-k",
            **scheduler_kwargs["prefill_bias_score_window_k"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-min-cached-tokens",
""",
            """        scheduler_group.add_argument(
            "--prefill-bias-score-window-k",
            **scheduler_kwargs["prefill_bias_score_window_k"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-candidate-scan-limit",
            **scheduler_kwargs["prefill_bias_candidate_scan_limit"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-min-cached-tokens",
""",
            "Phase 5 CLI flag",
        )

    if (
        "prefill_bias_candidate_scan_limit=self.prefill_bias_candidate_scan_limit"
        not in text
    ):
        text = replace_once(
            text,
            """            prefill_bias_cache_aware=self.prefill_bias_cache_aware,
            prefill_bias_score_window_k=self.prefill_bias_score_window_k,
            prefill_bias_min_cached_tokens=self.prefill_bias_min_cached_tokens,
""",
            """            prefill_bias_cache_aware=self.prefill_bias_cache_aware,
            prefill_bias_score_window_k=self.prefill_bias_score_window_k,
            prefill_bias_candidate_scan_limit=self.prefill_bias_candidate_scan_limit,
            prefill_bias_min_cached_tokens=self.prefill_bias_min_cached_tokens,
""",
            "Phase 5 SchedulerConfig propagation",
        )
    return write_if_changed(path, text)


def patch_scheduler_config_phase6(path: Path) -> bool:
    text = path.read_text()
    if PHASE6_MARKER in text:
        return False

    if "prefill_bias_tbt_guard_s" not in text:
        text = replace_once(
            text,
            '''    prefill_bias_tbt_guard_enabled: bool = False
    """Enable conservative TBT/ITL slack guard for prefill bias."""

    prefill_bias_tbt_slo_s: float = Field(default=0.0, ge=0.0)
''',
            '''    prefill_bias_tbt_guard_enabled: bool = False
    """Enable conservative TBT/ITL slack guard for prefill bias."""

    # VLLM_PREFILL_BIAS_PHASE6_PATCH: accepted-output guard compatibility.
    prefill_bias_tbt_guard_s: float = Field(default=0.0, ge=0.0)
    """Compatibility alias: >0 enables the TBT guard and supplies the SLO."""

    prefill_bias_tbt_slo_s: float = Field(default=0.0, ge=0.0)
''',
            "Phase 6 SchedulerConfig guard alias field",
        )
    else:
        text = text.replace(
            "    prefill_bias_tbt_guard_s: float = Field(default=0.0, ge=0.0)\n",
            "    # VLLM_PREFILL_BIAS_PHASE6_PATCH: accepted-output guard compatibility.\n"
            "    prefill_bias_tbt_guard_s: float = Field(default=0.0, ge=0.0)\n",
            1,
        )

    alias_validation = """        if self.prefill_bias_tbt_guard_s > 0.0:
            if self.prefill_bias_tbt_slo_s == 0.0:
                self.prefill_bias_tbt_slo_s = self.prefill_bias_tbt_guard_s
            elif self.prefill_bias_tbt_slo_s != self.prefill_bias_tbt_guard_s:
                logger.warning(
                    "prefill_bias_tbt_guard_s differs from prefill_bias_tbt_slo_s; "
                    "using prefill_bias_tbt_slo_s for the guard."
                )
            self.prefill_bias_tbt_guard_enabled = True

"""
    if "prefill_bias_tbt_guard_s differs from prefill_bias_tbt_slo_s" not in text:
        text = replace_once(
            text,
            "        # VLLM_PREFILL_BIAS_PHASE2_PATCH: TBT guard validation\n",
            alias_validation
            + "        # VLLM_PREFILL_BIAS_PHASE2_PATCH: TBT guard validation\n",
            "Phase 6 guard alias validation",
        )
    return write_if_changed(path, text)


def patch_arg_utils_phase6(path: Path) -> bool:
    text = path.read_text()
    if PHASE6_MARKER in text:
        return False

    if "prefill_bias_tbt_guard_s" not in text:
        text = replace_once(
            text,
            """    prefill_bias_tbt_guard_enabled: bool = (
        SchedulerConfig.prefill_bias_tbt_guard_enabled
    )
    prefill_bias_tbt_slo_s: float = SchedulerConfig.prefill_bias_tbt_slo_s
""",
            """    prefill_bias_tbt_guard_enabled: bool = (
        SchedulerConfig.prefill_bias_tbt_guard_enabled
    )
    # VLLM_PREFILL_BIAS_PHASE6_PATCH: accepted-output guard alias.
    prefill_bias_tbt_guard_s: float = SchedulerConfig.prefill_bias_tbt_guard_s
    prefill_bias_tbt_slo_s: float = SchedulerConfig.prefill_bias_tbt_slo_s
""",
            "Phase 6 EngineArgs guard alias field",
        )
    else:
        text = text.replace(
            "    prefill_bias_tbt_guard_s: float = SchedulerConfig.prefill_bias_tbt_guard_s\n",
            "    # VLLM_PREFILL_BIAS_PHASE6_PATCH: accepted-output guard alias.\n"
            "    prefill_bias_tbt_guard_s: float = SchedulerConfig.prefill_bias_tbt_guard_s\n",
            1,
        )

    if "--prefill-bias-tbt-guard-s" not in text:
        text = replace_once(
            text,
            """        scheduler_group.add_argument(
            "--prefill-bias-tbt-guard-enabled",
            **scheduler_kwargs["prefill_bias_tbt_guard_enabled"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-tbt-slo-s",
""",
            """        scheduler_group.add_argument(
            "--prefill-bias-tbt-guard-enabled",
            **scheduler_kwargs["prefill_bias_tbt_guard_enabled"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-tbt-guard-s",
            **scheduler_kwargs["prefill_bias_tbt_guard_s"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-tbt-slo-s",
""",
            "Phase 6 CLI guard alias",
        )

    if "prefill_bias_tbt_guard_s=self.prefill_bias_tbt_guard_s" not in text:
        text = replace_once(
            text,
            """            prefill_bias_tbt_guard_enabled=self.prefill_bias_tbt_guard_enabled,
            prefill_bias_tbt_slo_s=self.prefill_bias_tbt_slo_s,
""",
            """            prefill_bias_tbt_guard_enabled=self.prefill_bias_tbt_guard_enabled,
            prefill_bias_tbt_guard_s=self.prefill_bias_tbt_guard_s,
            prefill_bias_tbt_slo_s=self.prefill_bias_tbt_slo_s,
""",
            "Phase 6 SchedulerConfig guard alias propagation",
        )
    return write_if_changed(path, text)


def patch_scheduler_config_phase7(path: Path) -> bool:
    text = path.read_text()
    if PHASE7_MARKER in text:
        return False

    text = replace_once(
        text,
        '''    prefill_bias_slot_swap_enabled: bool = False
    """Enable conservative Phase 3 prefill admission swap."""

    prefill_bias_ttft_slo_s: float | None = None
''',
        '''    prefill_bias_slot_swap_enabled: bool = False
    """Enable conservative Phase 3 prefill admission swap."""

    # VLLM_PREFILL_BIAS_PHASE7_PATCH: TTFT deadline scheduling controls.
    prefill_bias_ttft_deadline_enabled: bool = False
    """Use predicted completion slack instead of a fixed waiting threshold."""

    prefill_bias_ttft_force_preempt_enabled: bool = False
    """Allow a projected TTFT miss to override TBT victim protection."""

    prefill_bias_ttft_slo_s: float | None = None
''',
        "Phase 7 SchedulerConfig deadline fields",
    )

    validation = """        # VLLM_PREFILL_BIAS_PHASE7_PATCH: deadline-mode validation.
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

"""
    text = replace_once(
        text,
        "        self.verify_max_model_len(max_model_len)\n",
        validation + "        self.verify_max_model_len(max_model_len)\n",
        "Phase 7 SchedulerConfig deadline validation",
    )
    return write_if_changed(path, text)


def patch_arg_utils_phase7(path: Path) -> bool:
    text = path.read_text()
    if PHASE7_MARKER in text:
        return False

    text = replace_once(
        text,
        """    prefill_bias_slot_swap_enabled: bool = (
        SchedulerConfig.prefill_bias_slot_swap_enabled
    )
    prefill_bias_ttft_slo_s: float | None = SchedulerConfig.prefill_bias_ttft_slo_s
""",
        """    prefill_bias_slot_swap_enabled: bool = (
        SchedulerConfig.prefill_bias_slot_swap_enabled
    )
    # VLLM_PREFILL_BIAS_PHASE7_PATCH: EngineArgs TTFT deadline fields.
    prefill_bias_ttft_deadline_enabled: bool = (
        SchedulerConfig.prefill_bias_ttft_deadline_enabled
    )
    prefill_bias_ttft_force_preempt_enabled: bool = (
        SchedulerConfig.prefill_bias_ttft_force_preempt_enabled
    )
    prefill_bias_ttft_slo_s: float | None = SchedulerConfig.prefill_bias_ttft_slo_s
""",
        "Phase 7 EngineArgs deadline fields",
    )

    text = replace_once(
        text,
        """        scheduler_group.add_argument(
            "--prefill-bias-slot-swap-enabled",
            **scheduler_kwargs["prefill_bias_slot_swap_enabled"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-ttft-slo-s",
""",
        """        scheduler_group.add_argument(
            "--prefill-bias-slot-swap-enabled",
            **scheduler_kwargs["prefill_bias_slot_swap_enabled"],
        )
        # VLLM_PREFILL_BIAS_PHASE7_PATCH: CLI TTFT deadline flags.
        scheduler_group.add_argument(
            "--prefill-bias-ttft-deadline-enabled",
            **scheduler_kwargs["prefill_bias_ttft_deadline_enabled"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-ttft-force-preempt-enabled",
            **scheduler_kwargs["prefill_bias_ttft_force_preempt_enabled"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-ttft-slo-s",
""",
        "Phase 7 CLI deadline flags",
    )

    text = replace_once(
        text,
        """            prefill_bias_slot_swap_enabled=self.prefill_bias_slot_swap_enabled,
            prefill_bias_ttft_slo_s=self.prefill_bias_ttft_slo_s,
""",
        """            prefill_bias_slot_swap_enabled=self.prefill_bias_slot_swap_enabled,
            prefill_bias_ttft_deadline_enabled=(
                self.prefill_bias_ttft_deadline_enabled
            ),
            prefill_bias_ttft_force_preempt_enabled=(
                self.prefill_bias_ttft_force_preempt_enabled
            ),
            prefill_bias_ttft_slo_s=self.prefill_bias_ttft_slo_s,
""",
        "Phase 7 SchedulerConfig deadline propagation",
    )
    return write_if_changed(path, text)


def patch_scheduler_config_phase8(path: Path) -> bool:
    text = path.read_text()
    if PHASE8_MARKER in text:
        return False

    text = replace_once(
        text,
        '''    prefill_bias_ttft_force_preempt_enabled: bool = False
    """Allow a projected TTFT miss to override TBT victim protection."""

    prefill_bias_ttft_slo_s: float | None = None
''',
        '''    prefill_bias_ttft_force_preempt_enabled: bool = False
    """Allow a projected TTFT miss to override TBT victim protection."""

    # VLLM_PREFILL_BIAS_PHASE8_PATCH: runtime-selectable predictive policy.
    prefill_bias_policy_mode: str = "legacy"
    """Select legacy, predictive-shadow, or predictive scheduling."""

    prefill_bias_predictive_min_chunk_tokens: int = Field(default=128, ge=1)
    prefill_bias_predictive_max_reserve_tokens: int = Field(default=3072, ge=0)
    prefill_bias_predictive_max_requests_per_step: int = Field(default=4, ge=1)
    prefill_bias_predictive_starvation_multiplier: float = Field(default=4.0, ge=1.0)

    prefill_bias_ttft_slo_s: float | None = None
''',
        "Phase 8 SchedulerConfig predictive fields",
    )

    validation = '''        # VLLM_PREFILL_BIAS_PHASE8_PATCH: predictive policy validation.
        predictive_modes = {"predictive-shadow", "predictive"}
        if self.prefill_bias_policy_mode not in {
            "legacy", "predictive-shadow", "predictive"
        }:
            raise ValueError(
                "prefill_bias_policy_mode must be legacy, predictive-shadow, "
                "or predictive."
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

'''
    text = replace_once(
        text,
        "        self.verify_max_model_len(max_model_len)\n",
        validation + "        self.verify_max_model_len(max_model_len)\n",
        "Phase 8 SchedulerConfig predictive validation",
    )
    return write_if_changed(path, text)


def patch_arg_utils_phase8(path: Path) -> bool:
    text = path.read_text()
    if PHASE8_MARKER in text:
        return False

    text = replace_once(
        text,
        '''    prefill_bias_ttft_force_preempt_enabled: bool = (
        SchedulerConfig.prefill_bias_ttft_force_preempt_enabled
    )
    prefill_bias_ttft_slo_s: float | None = SchedulerConfig.prefill_bias_ttft_slo_s
''',
        '''    prefill_bias_ttft_force_preempt_enabled: bool = (
        SchedulerConfig.prefill_bias_ttft_force_preempt_enabled
    )
    # VLLM_PREFILL_BIAS_PHASE8_PATCH: EngineArgs predictive policy fields.
    prefill_bias_policy_mode: str = SchedulerConfig.prefill_bias_policy_mode
    prefill_bias_predictive_min_chunk_tokens: int = (
        SchedulerConfig.prefill_bias_predictive_min_chunk_tokens
    )
    prefill_bias_predictive_max_reserve_tokens: int = (
        SchedulerConfig.prefill_bias_predictive_max_reserve_tokens
    )
    prefill_bias_predictive_max_requests_per_step: int = (
        SchedulerConfig.prefill_bias_predictive_max_requests_per_step
    )
    prefill_bias_predictive_starvation_multiplier: float = (
        SchedulerConfig.prefill_bias_predictive_starvation_multiplier
    )
    prefill_bias_ttft_slo_s: float | None = SchedulerConfig.prefill_bias_ttft_slo_s
''',
        "Phase 8 EngineArgs predictive fields",
    )

    text = replace_once(
        text,
        '''        scheduler_group.add_argument(
            "--prefill-bias-ttft-force-preempt-enabled",
            **scheduler_kwargs["prefill_bias_ttft_force_preempt_enabled"],
        )
        scheduler_group.add_argument(
            "--prefill-bias-ttft-slo-s",
''',
        '''        scheduler_group.add_argument(
            "--prefill-bias-ttft-force-preempt-enabled",
            **scheduler_kwargs["prefill_bias_ttft_force_preempt_enabled"],
        )
        # VLLM_PREFILL_BIAS_PHASE8_PATCH: CLI predictive policy flags.
        for name in (
            "prefill_bias_policy_mode",
            "prefill_bias_predictive_min_chunk_tokens",
            "prefill_bias_predictive_max_reserve_tokens",
            "prefill_bias_predictive_max_requests_per_step",
            "prefill_bias_predictive_starvation_multiplier",
        ):
            scheduler_group.add_argument(
                "--" + name.replace("_", "-"),
                **scheduler_kwargs[name],
            )
        scheduler_group.add_argument(
            "--prefill-bias-ttft-slo-s",
''',
        "Phase 8 CLI predictive flags",
    )

    text = replace_once(
        text,
        '''            prefill_bias_ttft_force_preempt_enabled=(
                self.prefill_bias_ttft_force_preempt_enabled
            ),
            prefill_bias_ttft_slo_s=self.prefill_bias_ttft_slo_s,
''',
        '''            prefill_bias_ttft_force_preempt_enabled=(
                self.prefill_bias_ttft_force_preempt_enabled
            ),
            prefill_bias_policy_mode=self.prefill_bias_policy_mode,
            prefill_bias_predictive_min_chunk_tokens=(
                self.prefill_bias_predictive_min_chunk_tokens
            ),
            prefill_bias_predictive_max_reserve_tokens=(
                self.prefill_bias_predictive_max_reserve_tokens
            ),
            prefill_bias_predictive_max_requests_per_step=(
                self.prefill_bias_predictive_max_requests_per_step
            ),
            prefill_bias_predictive_starvation_multiplier=(
                self.prefill_bias_predictive_starvation_multiplier
            ),
            prefill_bias_ttft_slo_s=self.prefill_bias_ttft_slo_s,
''',
        "Phase 8 SchedulerConfig predictive propagation",
    )
    return write_if_changed(path, text)


def patch_scheduler_config_phase9(path: Path) -> bool:
    text = path.read_text()
    if PHASE9_MARKER in text:
        return False

    text = replace_once(
        text,
        '''    prefill_bias_predictive_starvation_multiplier: float = Field(default=4.0, ge=1.0)

    prefill_bias_ttft_slo_s: float | None = None
''',
        '''    prefill_bias_predictive_starvation_multiplier: float = Field(default=4.0, ge=1.0)

    # VLLM_PREFILL_BIAS_PHASE9_PATCH: global prefill batch-budget controls.
    prefill_bias_batch_min_prefill_tokens: int = Field(default=128, ge=1)
    prefill_bias_batch_max_prefill_tokens: int = Field(default=3072, ge=0)
    prefill_bias_batch_max_requests_per_step: int = Field(default=4, ge=1)
    prefill_bias_batch_running_scan_limit: int = Field(default=32, ge=1)
    prefill_bias_batch_metrics_log_interval: int = Field(default=100, ge=1)

    prefill_bias_ttft_slo_s: float | None = None
''',
        "Phase 9 SchedulerConfig batch-budget fields",
    )
    text = replace_once(
        text,
        '''        predictive_modes = {"predictive-shadow", "predictive"}
        if self.prefill_bias_policy_mode not in {
            "legacy", "predictive-shadow", "predictive"
        }:
            raise ValueError(
                "prefill_bias_policy_mode must be legacy, predictive-shadow, "
                "or predictive."
            )
''',
        '''        predictive_modes = {"predictive-shadow", "predictive"}
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
''',
        "Phase 9 policy mode choices",
    )
    validation = '''        # VLLM_PREFILL_BIAS_PHASE9_PATCH: batch-budget validation.
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

'''
    text = replace_once(
        text,
        "        self.verify_max_model_len(max_model_len)\n",
        validation + "        self.verify_max_model_len(max_model_len)\n",
        "Phase 9 batch-budget validation",
    )
    return write_if_changed(path, text)


def patch_arg_utils_phase9(path: Path) -> bool:
    text = path.read_text()
    if PHASE9_MARKER in text:
        return False

    text = replace_once(
        text,
        '''    prefill_bias_predictive_starvation_multiplier: float = (
        SchedulerConfig.prefill_bias_predictive_starvation_multiplier
    )
    prefill_bias_ttft_slo_s: float | None = SchedulerConfig.prefill_bias_ttft_slo_s
''',
        '''    prefill_bias_predictive_starvation_multiplier: float = (
        SchedulerConfig.prefill_bias_predictive_starvation_multiplier
    )
    # VLLM_PREFILL_BIAS_PHASE9_PATCH: EngineArgs batch-budget fields.
    prefill_bias_batch_min_prefill_tokens: int = (
        SchedulerConfig.prefill_bias_batch_min_prefill_tokens
    )
    prefill_bias_batch_max_prefill_tokens: int = (
        SchedulerConfig.prefill_bias_batch_max_prefill_tokens
    )
    prefill_bias_batch_max_requests_per_step: int = (
        SchedulerConfig.prefill_bias_batch_max_requests_per_step
    )
    prefill_bias_batch_running_scan_limit: int = (
        SchedulerConfig.prefill_bias_batch_running_scan_limit
    )
    prefill_bias_batch_metrics_log_interval: int = (
        SchedulerConfig.prefill_bias_batch_metrics_log_interval
    )
    prefill_bias_ttft_slo_s: float | None = SchedulerConfig.prefill_bias_ttft_slo_s
''',
        "Phase 9 EngineArgs batch-budget fields",
    )
    text = replace_once(
        text,
        '''            "prefill_bias_predictive_starvation_multiplier",
        ):
''',
        '''            "prefill_bias_predictive_starvation_multiplier",
            "prefill_bias_batch_min_prefill_tokens",
            "prefill_bias_batch_max_prefill_tokens",
            "prefill_bias_batch_max_requests_per_step",
            "prefill_bias_batch_running_scan_limit",
            "prefill_bias_batch_metrics_log_interval",
        ):
''',
        "Phase 9 CLI batch-budget flags",
    )
    text = replace_once(
        text,
        '''            prefill_bias_predictive_starvation_multiplier=(
                self.prefill_bias_predictive_starvation_multiplier
            ),
            prefill_bias_ttft_slo_s=self.prefill_bias_ttft_slo_s,
''',
        '''            prefill_bias_predictive_starvation_multiplier=(
                self.prefill_bias_predictive_starvation_multiplier
            ),
            prefill_bias_batch_min_prefill_tokens=(
                self.prefill_bias_batch_min_prefill_tokens
            ),
            prefill_bias_batch_max_prefill_tokens=(
                self.prefill_bias_batch_max_prefill_tokens
            ),
            prefill_bias_batch_max_requests_per_step=(
                self.prefill_bias_batch_max_requests_per_step
            ),
            prefill_bias_batch_running_scan_limit=(
                self.prefill_bias_batch_running_scan_limit
            ),
            prefill_bias_batch_metrics_log_interval=(
                self.prefill_bias_batch_metrics_log_interval
            ),
            prefill_bias_ttft_slo_s=self.prefill_bias_ttft_slo_s,
''',
        "Phase 9 SchedulerConfig batch-budget propagation",
    )
    return write_if_changed(path, text)


def patch_scheduler(path: Path) -> bool:
    text = path.read_text()
    if PATCH_MARKER in text:
        return False

    import_anchor = """from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
"""
    import_new = """from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
# VLLM_PREFILL_BIAS_PATCH: imports
from vllm.v1.core.sched.prefill_bias import (
    PrefillBiasController,
    PrefillBiasDecision,
)
"""
    text = replace_once(text, import_anchor, import_new, "scheduler imports")

    init_anchor = """        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

    def _mamba_block_aligned_split(
"""
    init_new = """        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

        # VLLM_PREFILL_BIAS_PATCH: controller and phase-0 counters
        self.prefill_bias_controller = PrefillBiasController(self.scheduler_config)
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
        self._prefill_bias_last_decision = PrefillBiasDecision(
            active=False,
            reserve_tokens=0,
            candidate_request_ids=(),
            reason="disabled",
        )

    def _prefill_bias_request_is_schedulable(self, request: Request) -> bool:
        return not self._is_blocked_waiting_status(request.status)

    def _prefill_bias_decode_floor(self, token_budget: int) -> int:
        active_decode_requests = [
            req for req in self.running if not req.is_prefill_chunk
        ]
        tokens_per_decode = 1 + self.num_spec_tokens if self.num_spec_tokens > 0 else 1
        return min(token_budget, len(active_decode_requests) * tokens_per_decode)

    def _prefill_bias_prepare(
        self,
        *,
        token_budget: int,
        defer_prefills: bool,
    ) -> tuple[PrefillBiasDecision, int]:
        decode_floor = self._prefill_bias_decode_floor(token_budget)
        max_safe_reserve = max(0, token_budget - decode_floor)
        decision = self.prefill_bias_controller.decide(
            waiting=self.waiting,
            policy=self.policy,
            paused=self._pause_state != PauseState.UNPAUSED,
            throttle_prefills=defer_prefills,
            max_safe_reserve=max_safe_reserve,
            is_request_schedulable=self._prefill_bias_request_is_schedulable,
        )
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
        if (
            not decision.active
            or not self.scheduler_config.prefill_bias_cache_aware
        ):
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
                if age_s >= self.scheduler_config.prefill_bias_starvation_s:
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
"""
    text = replace_once(text, init_anchor, init_new, "controller init/helpers")

    schedule_anchor = """        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # First, schedule the RUNNING requests.
"""
    schedule_new = """        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # VLLM_PREFILL_BIAS_PATCH: reserve a safe slice for urgent waiting prefills.
        prefill_bias_held_tokens = 0
        prefill_bias_candidate_ids: set[str] = set()
        prefill_bias_decision, _ = self._prefill_bias_prepare(
            token_budget=token_budget,
            defer_prefills=defer_prefills,
        )
        if prefill_bias_decision.active:
            prefill_bias_held_tokens = prefill_bias_decision.reserve_tokens
            prefill_bias_candidate_ids = set(
                prefill_bias_decision.candidate_request_ids
            )
            token_budget -= prefill_bias_held_tokens

        # First, schedule the RUNNING requests.
"""
    text = replace_once(text, schedule_anchor, schedule_new, "running-loop prelude")

    waiting_anchor = """        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
"""
    waiting_new = """        # VLLM_PREFILL_BIAS_PATCH: restore held tokens exactly once before WAITING.
        if prefill_bias_held_tokens:
            token_budget += prefill_bias_held_tokens
            prefill_bias_held_tokens = 0
            prefill_bias_decision = self._prefill_bias_score_after_running(
                prefill_bias_decision
            )
            prefill_bias_candidate_ids = set(
                prefill_bias_decision.candidate_request_ids
            )
            self._prefill_bias_promote_waiting(
                prefill_bias_decision.candidate_request_ids
            )

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
"""
    text = replace_once(text, waiting_anchor, waiting_new, "restore before waiting")

    maxseq_anchor = """                if len(self.running) == self.max_num_running_reqs:
                    break
"""
    maxseq_new = """                if len(self.running) == self.max_num_running_reqs:
                    if prefill_bias_candidate_ids:
                        self._prefill_blocked_max_num_seqs += 1
                    break
"""
    text = replace_once(text, maxseq_anchor, maxseq_new, "max_num_seqs blocker")

    nokv_anchor = """                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
"""
    nokv_new = """                if new_blocks is None:
                    # The request cannot be scheduled.
                    if request_id in prefill_bias_candidate_ids:
                        self._prefill_blocked_no_kv += 1

                    # NOTE: we need to untouch the request from the encode cache
"""
    text = replace_once(text, nokv_anchor, nokv_new, "no kv blocker")

    admit_anchor = """                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
"""
    admit_new = """                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                    if request_id in prefill_bias_candidate_ids:
                        self._prefill_bias_admitted_requests += 1
                elif request.status == RequestStatus.PREEMPTED:
"""
    text = replace_once(text, admit_anchor, admit_new, "admitted counter")
    return write_if_changed(path, text)


def patch_scheduler_phase2(path: Path) -> bool:
    text = path.read_text()
    if PHASE2_MARKER in text:
        return False

    text = replace_once(
        text,
        "from dataclasses import replace\n",
        "from dataclasses import dataclass, replace\n",
        "Phase 2 dataclass import",
    )
    text = replace_once(
        text,
        """from vllm.v1.core.sched.prefill_bias import (
    PrefillBiasController,
    PrefillBiasDecision,
)
""",
        """from vllm.v1.core.sched.prefill_bias import (
    PrefillBiasController,
    PrefillBiasDecision,
    PrefillStepTimeEstimator,
    TBTGuardSnapshot,
)
""",
        "Phase 2 policy imports",
    )
    text = replace_once(
        text,
        """logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
""",
        """logger = init_logger(__name__)


@dataclass
class ScheduledBatchTiming:
    started_at: float
    prefill_tokens: int
    total_tokens: int


class Scheduler(SchedulerInterface):
""",
        "ScheduledBatchTiming",
    )
    text = replace_once(
        text,
        "        self.prefill_bias_controller = PrefillBiasController(self.scheduler_config)\n",
        """        self._prefill_bias_monotonic_clock = time.monotonic
        self.prefill_bias_controller = PrefillBiasController(
            self.scheduler_config,
            monotonic_clock=self._prefill_bias_monotonic_clock,
        )
""",
        "Phase 2 controller clock",
    )
    text = replace_once(
        text,
        """        self._prefill_bias_starvation_overrides = 0
        self._prefill_bias_last_decision = PrefillBiasDecision(
""",
        """        self._prefill_bias_starvation_overrides = 0
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
        self._prefill_bias_last_decision = PrefillBiasDecision(
""",
        "Phase 2 scheduler state",
    )
    text = replace_once(
        text,
        """    def _prefill_bias_decode_floor(self, token_budget: int) -> int:
        active_decode_requests = [
            req for req in self.running if not req.is_prefill_chunk
        ]
        tokens_per_decode = 1 + self.num_spec_tokens if self.num_spec_tokens > 0 else 1
        return min(token_budget, len(active_decode_requests) * tokens_per_decode)

    def _prefill_bias_prepare(
""",
        """    def _prefill_bias_decode_floor(self, token_budget: int) -> int:
        active_decode_requests = [
            req for req in self.running if not req.is_prefill_chunk
        ]
        tokens_per_decode = 1 + self.num_spec_tokens if self.num_spec_tokens > 0 else 1
        return min(token_budget, len(active_decode_requests) * tokens_per_decode)

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
        return self._prefill_bias_step_time_estimator.estimate(
            initial_step_time_s=self.scheduler_config.prefill_bias_initial_step_time_s,
            headroom_factor=(
                self.scheduler_config.prefill_bias_step_time_headroom_factor
            ),
            min_samples=(
                self.scheduler_config.prefill_bias_step_observation_min_samples
            ),
        )

    def _prefill_bias_apply_tbt_guard(
        self,
        decision: PrefillBiasDecision,
    ) -> PrefillBiasDecision:
        if (
            not decision.active
            or not self.scheduler_config.prefill_bias_tbt_guard_enabled
        ):
            return decision

        snapshot = self.prefill_bias_controller.evaluate_tbt_guard(
            now_monotonic=self._prefill_bias_monotonic_clock(),
            active_decode_request_ids=self._prefill_bias_active_decode_request_ids(),
            last_output_ts=self._last_accepted_output_ts,
            tbt_slo_s=self.scheduler_config.prefill_bias_tbt_slo_s,
            predicted_step_time_s=self._prefill_bias_estimated_step_time(),
            safety_margin_s=self.scheduler_config.prefill_bias_tbt_safety_margin_s,
            guard_unknown_decode=(
                self.scheduler_config.prefill_bias_guard_unknown_decode
            ),
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
    ) -> None:
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return
        self._scheduled_batch_timing[id(scheduler_output)] = ScheduledBatchTiming(
            started_at=self._prefill_bias_monotonic_clock(),
            prefill_tokens=prefill_tokens,
            total_tokens=scheduler_output.total_num_scheduled_tokens,
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

    def _prefill_bias_prepare(
""",
        "Phase 2 scheduler helpers",
    )
    text = replace_once(
        text,
        """        self._prefill_bias_last_decision = decision
        if not decision.active:
""",
        """        decision = self._prefill_bias_apply_tbt_guard(decision)
        self._prefill_bias_last_decision = decision
        if not decision.active:
""",
        "Phase 2 guard before reservation",
    )
    text = replace_once(
        text,
        """        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
""",
        """        # VLLM_PREFILL_BIAS_PHASE2_PATCH: classify scheduled prefill work before
        # _update_after_schedule mutates request.num_computed_tokens/is_prefill_chunk.
        prefill_tokens_in_batch = sum(
            num_tokens
            for request_id, num_tokens in num_scheduled_tokens.items()
            if self.requests[request_id].num_output_tokens == 0
        )

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
""",
        "Phase 2 prefill token accounting",
    )
    text = replace_once(
        text,
        """        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output
""",
        """        self._prefill_bias_track_scheduled_batch(
            scheduler_output,
            prefill_tokens_in_batch,
        )

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output
""",
        "Phase 2 scheduled batch timing",
    )
    text = replace_once(
        text,
        """        cudagraph_stats = model_runner_output.cudagraph_stats

        # Every GPU write enqueued by this and earlier steps has completed, so it is
""",
        """        cudagraph_stats = model_runner_output.cudagraph_stats

        self._prefill_bias_observe_batch_timing(scheduler_output)

        # Every GPU write enqueued by this and earlier steps has completed, so it is
""",
        "Phase 2 observe batch timing",
    )
    text = replace_once(
        text,
        """            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
""",
        """            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
                if new_token_ids:
                    self._last_accepted_output_ts[req_id] = (
                        self._prefill_bias_monotonic_clock()
                    )
            elif request.pooling_params and pooler_output is not None:
""",
        "Phase 2 accepted output timestamp",
    )
    text = replace_once(
        text,
        """        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
""",
        """        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self._last_accepted_output_ts.pop(request_id, None)
        self.finished_req_ids.add(request_id)
""",
        "Phase 2 cleanup timestamp",
    )
    return write_if_changed(path, text)


def patch_scheduler_phase3(path: Path) -> bool:
    text = path.read_text()
    if PHASE3_MARKER in text:
        return False

    text = replace_once(
        text,
        """from vllm.v1.core.sched.prefill_bias import (
    PrefillBiasController,
    PrefillBiasDecision,
    PrefillStepTimeEstimator,
    TBTGuardSnapshot,
)
""",
        """from vllm.v1.core.sched.prefill_bias import (
    PrefillAdmissionSwapResult,
    PrefillBiasController,
    PrefillBiasDecision,
    PrefillStepTimeEstimator,
    SlotSwapBlocker,
    SlotSwapRejectReason,
    TBTGuardSnapshot,
)
""",
        "Phase 3 policy imports",
    )

    text = replace_once(
        text,
        """        self._prefill_bias_last_tbt_snapshot = TBTGuardSnapshot(
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
        self._prefill_bias_last_decision = PrefillBiasDecision(
""",
        """        self._prefill_bias_last_tbt_snapshot = TBTGuardSnapshot(
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
        self._prefill_bias_last_decision = PrefillBiasDecision(
""",
        "Phase 3 scheduler state",
    )

    text = replace_once(
        text,
        """    def _prefill_bias_prepare(
""",
        """    def _prefill_slot_swap_reject_candidate(
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
        cached_tokens = max(0, self.kv_cache_manager.peek_num_computed_tokens(candidate))
        effective_cached_tokens = cached_tokens
        if cached_tokens < self.scheduler_config.prefill_bias_min_cached_tokens:
            effective_cached_tokens = 0
        remaining_tokens = max(candidate.num_tokens - effective_cached_tokens, 1)
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
        now = self._prefill_bias_monotonic_clock()
        tbt_slo_s = self.scheduler_config.prefill_bias_tbt_slo_s
        safety_margin_s = self.scheduler_config.prefill_bias_tbt_safety_margin_s
        margin_s = self.scheduler_config.prefill_bias_victim_tbt_margin_s
        max_recompute = self.scheduler_config.prefill_bias_max_victim_recompute_tokens
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
            last_output_ts = self._last_accepted_output_ts.get(request.request_id)
            if last_output_ts is None:
                continue
            elapsed_s = now - last_output_ts
            if elapsed_s < 0.0 or elapsed_s >= tbt_slo_s:
                continue
            cooldown_until = self._prefill_bias_phase3_last_preempted_ts.get(
                request.request_id,
                -1.0,
            ) + self.scheduler_config.prefill_bias_swap_cooldown_s
            resume_cooldown_until = self._prefill_bias_phase3_last_resumed_ts.get(
                request.request_id,
                -1.0,
            ) + self.scheduler_config.prefill_bias_swap_cooldown_s
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
                >= self.scheduler_config.prefill_bias_max_preemptions_per_request
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
                + self.scheduler_config.prefill_bias_initial_step_time_s
                + safety_margin_s
            )
            projected_tbt_s = elapsed_s + victim_resume_cost_s
            tbt_limit = tbt_slo_s - margin_s
            if projected_tbt_s > tbt_limit:
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
    ) -> tuple[PrefillAdmissionSwapResult, int]:
        if not self.scheduler_config.prefill_bias_slot_swap_enabled:
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.FEATURE_DISABLED,
            ), token_budget
        self._prefill_slot_swap_attempts_total += 1
        if self.policy != SchedulingPolicy.FCFS or self._pause_state != PauseState.UNPAUSED:
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.FEATURE_DISABLED,
            ), token_budget
        if swaps_this_step >= self.scheduler_config.prefill_bias_max_swaps_per_step:
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.FEATURE_DISABLED,
            ), token_budget
        if preempted_reqs:
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.BLOCKER_NOT_MAX_NUM_SEQS,
            ), token_budget

        candidate = self._prefill_slot_swap_candidate_from_decision(decision)
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
            predicted_step_time_s = self._prefill_bias_estimated_step_time()
            should_swap, ttft_slack_s = (
                self.prefill_bias_controller.should_try_slot_swap(
                    now_wall=time.time(),
                    arrival_time=candidate.arrival_time,
                    ttft_slo_s=self.scheduler_config.prefill_bias_ttft_slo_s,
                    predicted_prefill_s=predicted_step_time_s,
                    slack_threshold_s=(
                        self.scheduler_config.prefill_bias_swap_slack_threshold_s
                    ),
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
            > self.scheduler_config.prefill_bias_max_candidate_remaining_tokens
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
        if self.scheduler_config.prefill_bias_require_cache_residency and cached_tokens <= 0:
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
            last_output_ts=self._last_accepted_output_ts,
            tbt_slo_s=self.scheduler_config.prefill_bias_tbt_slo_s,
            predicted_step_time_s=predicted_step_time_s,
            safety_margin_s=self.scheduler_config.prefill_bias_tbt_safety_margin_s,
            guard_unknown_decode=self.scheduler_config.prefill_bias_guard_unknown_decode,
        )
        if not guard.allowed:
            self._prefill_slot_swap_reject_candidate(SlotSwapRejectReason.TBT_GUARD)
            return self._prefill_slot_swap_result(
                candidate_request_id=candidate_id,
                blocker_reason=SlotSwapBlocker.TBT_GUARD,
                reject_reason=SlotSwapRejectReason.TBT_GUARD,
                predicted_ttft_slack_s=ttft_slack_s,
                candidate_remaining_tokens=candidate_remaining_tokens,
            ), token_budget

        victim, projected_tbt_s, recompute_tokens = self._prefill_slot_swap_select_victim(
            candidate_id=candidate_id,
            predicted_step_time_s=predicted_step_time_s,
        )
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
            token_budget += num_scheduled_tokens.pop(victim_id, 0)
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

    def _prefill_bias_prepare(
""",
        "Phase 3 scheduler helpers",
    )

    text = replace_once(
        text,
        """        prefill_bias_held_tokens = 0
        prefill_bias_candidate_ids: set[str] = set()
""",
        """        prefill_bias_held_tokens = 0
        prefill_bias_candidate_ids: set[str] = set()
        prefill_bias_swap_performed = False
        prefill_bias_swap_candidate_id: str | None = None
        prefill_bias_swaps_this_step = 0
""",
        "Phase 3 schedule state",
    )

    text = replace_once(
        text,
        """            self._prefill_bias_promote_waiting(
                prefill_bias_decision.candidate_request_ids
            )

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
""",
        """            self._prefill_bias_promote_waiting(
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
""",
        "Phase 3 swap before waiting",
    )

    text = replace_once(
        text,
        """                if new_blocks is None:
                    # The request cannot be scheduled.
                    if request_id in prefill_bias_candidate_ids:
                        self._prefill_blocked_no_kv += 1

                    # NOTE: we need to untouch the request from the encode cache
""",
        """                if new_blocks is None:
                    # The request cannot be scheduled.
                    if request_id in prefill_bias_candidate_ids:
                        self._prefill_blocked_no_kv += 1
                    if request_id == prefill_bias_swap_candidate_id:
                        self._prefill_slot_swap_commit_failures_total += 1
                        self._prefill_bias_phase3_candidate_backoff_until[
                            request_id
                        ] = (
                            self._prefill_bias_monotonic_clock()
                            + self.scheduler_config.prefill_bias_swap_failure_backoff_s
                        )

                    # NOTE: we need to untouch the request from the encode cache
""",
        "Phase 3 commit failure backoff",
    )

    text = replace_once(
        text,
        """                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                    if request_id in prefill_bias_candidate_ids:
                        self._prefill_bias_admitted_requests += 1
                elif request.status == RequestStatus.PREEMPTED:
""",
        """                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                    if request_id in prefill_bias_candidate_ids:
                        self._prefill_bias_admitted_requests += 1
                    if request_id == prefill_bias_swap_candidate_id:
                        self._prefill_slot_swap_candidate_admissions_total += 1
                elif request.status == RequestStatus.PREEMPTED:
                    self._prefill_bias_phase3_last_resumed_ts[request_id] = (
                        self._prefill_bias_monotonic_clock()
                    )
""",
        "Phase 3 admission/resume counters",
    )

    text = replace_once(
        text,
        """        self._last_accepted_output_ts.pop(request_id, None)
        self.finished_req_ids.add(request_id)
""",
        """        self._last_accepted_output_ts.pop(request_id, None)
        self._prefill_bias_phase3_preemption_count.pop(request_id, None)
        self._prefill_bias_phase3_last_preempted_ts.pop(request_id, None)
        self._prefill_bias_phase3_last_resumed_ts.pop(request_id, None)
        self._prefill_bias_phase3_candidate_backoff_until.pop(request_id, None)
        self.finished_req_ids.add(request_id)
""",
        "Phase 3 cleanup state",
    )
    return write_if_changed(path, text)


def patch_scheduler_phase4(path: Path) -> bool:
    text = path.read_text()
    if PHASE4_MARKER in text:
        return False

    text = replace_once(
        text,
        """from vllm.v1.core.sched.prefill_bias import (
    PrefillAdmissionSwapResult,
    PrefillBiasController,
    PrefillBiasDecision,
    PrefillStepTimeEstimator,
    SlotSwapBlocker,
    SlotSwapRejectReason,
    TBTGuardSnapshot,
)
""",
        """from vllm.v1.core.sched.prefill_bias import (
    AdaptivePrefillController,
    PrefillAdmissionSwapResult,
    PrefillBiasController,
    PrefillBiasDecision,
    PrefillStepTimeEstimator,
    SlotSwapBlocker,
    SlotSwapRejectReason,
    TBTGuardSnapshot,
)
""",
        "Phase 4 policy imports",
    )

    text = replace_once(
        text,
        """        self._prefill_slot_swap_last_result = PrefillAdmissionSwapResult(
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
        self._prefill_bias_last_decision = PrefillBiasDecision(
""",
        """        self._prefill_slot_swap_last_result = PrefillAdmissionSwapResult(
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
        self._prefill_bias_last_decision = PrefillBiasDecision(
""",
        "Phase 4 scheduler state",
    )

    text = replace_once(
        text,
        """    def _prefill_bias_request_is_schedulable(self, request: Request) -> bool:
        return not self._is_blocked_waiting_status(request.status)

    def _prefill_bias_decode_floor(self, token_budget: int) -> int:
""",
        """    def _adaptive_prefill_enabled(self) -> bool:
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
""",
        "Phase 4 scheduler helpers",
    )

    text = replace_once(
        text,
        """    def _prefill_bias_estimated_step_time(self) -> float:
        return self._prefill_bias_step_time_estimator.estimate(
            initial_step_time_s=self.scheduler_config.prefill_bias_initial_step_time_s,
            headroom_factor=(
                self.scheduler_config.prefill_bias_step_time_headroom_factor
            ),
            min_samples=(
                self.scheduler_config.prefill_bias_step_observation_min_samples
            ),
        )
""",
        """    def _prefill_bias_estimated_step_time(self) -> float:
        config = self._adaptive_prefill_effective_config()
        return self._prefill_bias_step_time_estimator.estimate(
            initial_step_time_s=config.prefill_bias_initial_step_time_s,
            headroom_factor=config.prefill_bias_step_time_headroom_factor,
            min_samples=config.prefill_bias_step_observation_min_samples,
        )
""",
        "Phase 4 estimated step config",
    )

    text = replace_once(
        text,
        """        if (
            not decision.active
            or not self.scheduler_config.prefill_bias_tbt_guard_enabled
        ):
            return decision

        snapshot = self.prefill_bias_controller.evaluate_tbt_guard(
            now_monotonic=self._prefill_bias_monotonic_clock(),
            active_decode_request_ids=self._prefill_bias_active_decode_request_ids(),
            last_output_ts=self._last_accepted_output_ts,
            tbt_slo_s=self.scheduler_config.prefill_bias_tbt_slo_s,
            predicted_step_time_s=self._prefill_bias_estimated_step_time(),
            safety_margin_s=self.scheduler_config.prefill_bias_tbt_safety_margin_s,
            guard_unknown_decode=(
                self.scheduler_config.prefill_bias_guard_unknown_decode
            ),
        )
""",
        """        config = self._adaptive_prefill_effective_config()
        if not decision.active or not config.prefill_bias_tbt_guard_enabled:
            return decision

        snapshot = self.prefill_bias_controller.evaluate_tbt_guard(
            now_monotonic=self._prefill_bias_monotonic_clock(),
            active_decode_request_ids=self._prefill_bias_active_decode_request_ids(),
            last_output_ts=self._last_accepted_output_ts,
            tbt_slo_s=config.prefill_bias_tbt_slo_s,
            predicted_step_time_s=self._prefill_bias_estimated_step_time(),
            safety_margin_s=config.prefill_bias_tbt_safety_margin_s,
            guard_unknown_decode=config.prefill_bias_guard_unknown_decode,
        )
""",
        "Phase 4 tbt guard config",
    )

    text = replace_once(
        text,
        """        if (
            not decision.active
            or not self.scheduler_config.prefill_bias_cache_aware
        ):
            return decision
""",
        """        config = self._adaptive_prefill_effective_config()
        if not decision.active or not config.prefill_bias_cache_aware:
            return decision
""",
        "Phase 4 score config",
    )

    text = replace_once(
        text,
        """                if age_s >= self.scheduler_config.prefill_bias_starvation_s:
                    self._prefill_bias_starvation_overrides += 1
""",
        """                if age_s >= config.prefill_bias_starvation_s:
                    self._prefill_bias_starvation_overrides += 1
""",
        "Phase 4 starvation config",
    )

    text = (
        replace_once(
            text,
            """        if not self.scheduler_config.prefill_bias_slot_swap_enabled:
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.FEATURE_DISABLED,
            ), token_budget
        self._prefill_slot_swap_attempts_total += 1
        if self.policy != SchedulingPolicy.FCFS or self._pause_state != RequestStatus.UNPAUSED:
""",
            """        config = self._adaptive_prefill_effective_config()
        if not config.prefill_bias_slot_swap_enabled:
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.FEATURE_DISABLED,
            ), token_budget
        self._prefill_slot_swap_attempts_total += 1
        if self.policy != SchedulingPolicy.FCFS or self._pause_state != PauseState.UNPAUSED:
""",
            "Phase 4 slot swap enabled config",
        )
        if """self._pause_state != RequestStatus.UNPAUSED""" in text
        else text
    )

    text = text.replace(
        "        if not self.scheduler_config.prefill_bias_slot_swap_enabled:\n",
        "        config = self._adaptive_prefill_effective_config()\n        if not config.prefill_bias_slot_swap_enabled:\n",
        1,
    )
    text = text.replace(
        "        if swaps_this_step >= self.scheduler_config.prefill_bias_max_swaps_per_step:\n",
        "        if swaps_this_step >= config.prefill_bias_max_swaps_per_step:\n",
        1,
    )
    text = text.replace(
        "            > self.scheduler_config.prefill_bias_max_candidate_remaining_tokens\n",
        "            > config.prefill_bias_max_candidate_remaining_tokens\n",
        1,
    )

    text = replace_once(
        text,
        """        self.kv_cache_manager.new_step_starts()

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
""",
        """        self.kv_cache_manager.new_step_starts()

        self._adaptive_prefill_observe_pressure(token_budget)
        self._adaptive_prefill_refresh_policy()

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
""",
        "Phase 4 schedule policy refresh",
    )

    text = replace_once(
        text,
        """                if new_token_ids:
                    self._last_accepted_output_ts[req_id] = (
                        self._prefill_bias_monotonic_clock()
                    )
""",
        """                if new_token_ids:
                    accepted_ts = self._prefill_bias_monotonic_clock()
                    self._last_accepted_output_ts[req_id] = accepted_ts
                    self._adaptive_prefill_observe_accepted_tokens(
                        request,
                        len(new_token_ids),
                        now_monotonic=accepted_ts,
                    )
""",
        "Phase 4 accepted token observations",
    )

    text = replace_once(
        text,
        """        request_id = request.request_id
        self._last_accepted_output_ts.pop(request_id, None)
""",
        """        request_id = request.request_id
        self._adaptive_prefill_observe_request_finished(request)
        self._last_accepted_output_ts.pop(request_id, None)
""",
        "Phase 4 request completion observation",
    )
    return write_if_changed(path, text)


def patch_scheduler_phase6(path: Path) -> bool:
    text = path.read_text()
    if PHASE6_MARKER in text:
        return False

    if "DecodeTimingState" not in text:
        text = replace_once(
            text,
            "    PrefillAdmissionSwapResult,\n",
            "    DecodeTimingState,\n    PrefillAdmissionSwapResult,\n",
            "Phase 6 DecodeTimingState import",
        )

    text = replace_once(
        text,
        "        self._last_accepted_output_ts: dict[str, float] = {}\n",
        """        # VLLM_PREFILL_BIAS_PHASE6_PATCH: accepted-output timing is keyed by
        # request id and guarded by Request object identity to prevent stale async
        # output or request-id reuse from reusing an old decode timestamp.
        self._decode_timing: dict[str, DecodeTimingState] = {}
        self._last_accepted_output_ts: dict[str, float] = {}
""",
        "Phase 6 decode timing state",
    )

    helpers = """    # VLLM_PREFILL_BIAS_PHASE6_PATCH: accepted-output timing helpers.
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

"""
    text = replace_once(
        text,
        "    def _is_active_decode_request(self, request: Request) -> bool:\n",
        helpers
        + "    def _is_active_decode_request(self, request: Request) -> bool:\n",
        "Phase 6 accepted-output helpers",
    )

    text = text.replace(
        "last_output_ts=self._last_accepted_output_ts,",
        "last_output_ts=self._prefill_bias_last_output_timestamps(),",
    )

    if "self._last_accepted_output_ts.get(request.request_id)" in text:
        text = text.replace(
            "            last_output_ts = self._last_accepted_output_ts.get(request.request_id)\n",
            "            last_output_ts = self._prefill_bias_last_output_timestamp(request)\n",
            1,
        )

    if (
        "self._record_accepted_decode_output(request, new_token_ids, accepted_ts)"
        not in text
    ):
        if (
            """                if new_token_ids:
                    accepted_ts = self._prefill_bias_monotonic_clock()
                    self._last_accepted_output_ts[req_id] = accepted_ts
                    self._adaptive_prefill_observe_accepted_tokens(
                        request,
                        len(new_token_ids),
                        now_monotonic=accepted_ts,
                    )
"""
            in text
        ):
            text = replace_once(
                text,
                """                if new_token_ids:
                    accepted_ts = self._prefill_bias_monotonic_clock()
                    self._last_accepted_output_ts[req_id] = accepted_ts
                    self._adaptive_prefill_observe_accepted_tokens(
                        request,
                        len(new_token_ids),
                        now_monotonic=accepted_ts,
                    )
""",
                """                if new_token_ids:
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
""",
                "Phase 6 accepted output hook with adaptive observation",
            )
        else:
            text = replace_once(
                text,
                """                if new_token_ids:
                    self._last_accepted_output_ts[req_id] = (
                        self._prefill_bias_monotonic_clock()
                    )
""",
                """                if new_token_ids:
                    self._record_accepted_decode_output(request, new_token_ids)
""",
                "Phase 6 accepted output hook",
            )

    if "self._clear_decode_timing(request.request_id)" not in text:
        text = replace_once(
            text,
            """        else:
            if request.resumable:
""",
            """        else:
            self._clear_decode_timing(request.request_id)
            if request.resumable:
""",
            "Phase 6 request-id reuse cleanup",
        )

    if "self._adaptive_prefill_observe_request_finished(request)" in text:
        text = replace_once(
            text,
            """        request_id = request.request_id
        self._adaptive_prefill_observe_request_finished(request)
        self._last_accepted_output_ts.pop(request_id, None)
""",
            """        request_id = request.request_id
        self._adaptive_prefill_observe_request_finished(request)
        self._clear_decode_timing(request_id)
""",
            "Phase 6 finished request cleanup",
        )
    else:
        text = replace_once(
            text,
            """        request_id = request.request_id
        self._last_accepted_output_ts.pop(request_id, None)
""",
            """        request_id = request.request_id
        self._clear_decode_timing(request_id)
""",
            "Phase 6 finished request cleanup",
        )

    return write_if_changed(path, text)


def patch_scheduler_phase7(path: Path) -> bool:
    text = path.read_text()
    if PHASE7_MARKER in text:
        return False

    text = replace_once(
        text,
        """        self._prefill_bias_last_decision = PrefillBiasDecision(
            active=False,
            reserve_tokens=0,
            candidate_request_ids=(),
            reason="disabled",
        )

    def _adaptive_prefill_enabled(self) -> bool:
""",
        """        # VLLM_PREFILL_BIAS_PHASE7_PATCH: bounded TTFT-deadline metrics.
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
        self._prefill_bias_last_decision = PrefillBiasDecision(
            active=False,
            reserve_tokens=0,
            candidate_request_ids=(),
            reason="disabled",
        )

    def _adaptive_prefill_enabled(self) -> bool:
""",
        "Phase 7 deadline metrics state",
    )

    text = replace_once(
        text,
        """        if snapshot.allowed:
            return decision

        logger.debug(
            "Prefill bias blocked by TBT guard: reason=%s active_decodes=%d "
""",
        """        if snapshot.allowed:
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
""",
        "Phase 7 forced TTFT guard override",
    )

    text = replace_once(
        text,
        """    def _prefill_slot_swap_candidate_remaining(
        self,
        candidate: Request,
    ) -> tuple[int, int]:
        cached_tokens = max(0, self.kv_cache_manager.peek_num_computed_tokens(candidate))
        effective_cached_tokens = cached_tokens
        if cached_tokens < self.scheduler_config.prefill_bias_min_cached_tokens:
            effective_cached_tokens = 0
        remaining_tokens = max(candidate.num_tokens - effective_cached_tokens, 1)
        return remaining_tokens, cached_tokens
""",
        """    def _prefill_slot_swap_candidate_remaining(
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
""",
        "Phase 7 authoritative candidate remaining tokens",
    )

    text = replace_once(
        text,
        """    ) -> tuple[Request | None, float | None, int | None]:
        now = self._prefill_bias_monotonic_clock()
        tbt_slo_s = self.scheduler_config.prefill_bias_tbt_slo_s
        safety_margin_s = self.scheduler_config.prefill_bias_tbt_safety_margin_s
        margin_s = self.scheduler_config.prefill_bias_victim_tbt_margin_s
        max_recompute = self.scheduler_config.prefill_bias_max_victim_recompute_tokens
""",
        """    ) -> tuple[Request | None, float | None, int | None]:
        config = self._adaptive_prefill_effective_config()
        now = self._prefill_bias_monotonic_clock()
        tbt_slo_s = config.prefill_bias_tbt_slo_s
        safety_margin_s = config.prefill_bias_tbt_safety_margin_s
        margin_s = config.prefill_bias_victim_tbt_margin_s
        max_recompute = config.prefill_bias_max_victim_recompute_tokens
""",
        "Phase 7 safe victim effective config",
    )
    text = text.replace(
        " + self.scheduler_config.prefill_bias_swap_cooldown_s\n",
        " + config.prefill_bias_swap_cooldown_s\n",
        2,
    )
    text = replace_once(
        text,
        """                preemption_count
                >= self.scheduler_config.prefill_bias_max_preemptions_per_request
""",
        """                preemption_count
                >= config.prefill_bias_max_preemptions_per_request
""",
        "Phase 7 safe victim preemption cap",
    )
    text = replace_once(
        text,
        """                predicted_step_time_s
                + self.scheduler_config.prefill_bias_initial_step_time_s
                + safety_margin_s
""",
        """                predicted_step_time_s
                + config.prefill_bias_initial_step_time_s
                + safety_margin_s
""",
        "Phase 7 safe victim resume estimate",
    )
    text = replace_once(
        text,
        "            if projected_tbt_s > tbt_limit:\n",
        """            if not self.prefill_bias_controller.is_safe_projected_tbt(
                projected_tbt_s=projected_tbt_s,
                tbt_limit_s=tbt_limit,
            ):
""",
        "Phase 7 strict safe-victim TBT boundary",
    )

    forced_helper = """    def _prefill_slot_swap_select_forced_victim(
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

"""
    text = replace_once(
        text,
        "    def _try_prefill_admission_swap(\n",
        forced_helper + "    def _try_prefill_admission_swap(\n",
        "Phase 7 forced victim selector",
    )

    text = replace_once(
        text,
        """        guard = self.prefill_bias_controller.evaluate_tbt_guard(
            now_monotonic=now_monotonic,
            active_decode_request_ids=self._prefill_bias_active_decode_request_ids(),
            last_output_ts=self._prefill_bias_last_output_timestamps(),
            tbt_slo_s=self.scheduler_config.prefill_bias_tbt_slo_s,
            predicted_step_time_s=predicted_step_time_s,
            safety_margin_s=self.scheduler_config.prefill_bias_tbt_safety_margin_s,
            guard_unknown_decode=self.scheduler_config.prefill_bias_guard_unknown_decode,
        )
        if not guard.allowed:
""",
        """        guard = self.prefill_bias_controller.evaluate_tbt_guard(
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
""",
        "Phase 7 slot-swap TBT override",
    )

    text = replace_once(
        text,
        """        victim, projected_tbt_s, recompute_tokens = self._prefill_slot_swap_select_victim(
            candidate_id=candidate_id,
            predicted_step_time_s=predicted_step_time_s,
        )
        if victim is None or projected_tbt_s is None or recompute_tokens is None:
""",
        """        forced_swap = False
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
""",
        "Phase 7 safe-then-forced victim selection",
    )

    text = replace_once(
        text,
        """        if not self._prefill_slot_swap_kv_preflight(
            candidate=candidate,
            victim=victim,
            candidate_remaining_tokens=candidate_remaining_tokens,
        ):
            self._prefill_slot_swap_reject_candidate(
""",
        """        if not self._prefill_slot_swap_kv_preflight(
            candidate=candidate,
            victim=victim,
            candidate_remaining_tokens=candidate_remaining_tokens,
        ):
            if forced_swap:
                self._prefill_bias_forced_kv_preflight_blocked_total += 1
            self._prefill_slot_swap_reject_candidate(
""",
        "Phase 7 forced KV preflight metric",
    )

    text = replace_once(
        text,
        """        self._prefill_slot_swap_preemptions_total += 1
        self._prefill_slot_swaps_total += 1
        self._prefill_slot_swap_force_candidate_first(candidate_id)
""",
        """        self._prefill_slot_swap_preemptions_total += 1
        self._prefill_slot_swaps_total += 1
        if forced_swap:
            self._prefill_bias_forced_preemptions_total += 1
        else:
            self._prefill_bias_safe_swaps_total += 1
        self._prefill_slot_swap_force_candidate_first(candidate_id)
""",
        "Phase 7 safe and forced swap metrics",
    )

    text = replace_once(
        text,
        """            predicted_step_time_s = self._prefill_bias_estimated_step_time()
            should_swap, ttft_slack_s = (
                self.prefill_bias_controller.should_try_slot_swap(
                    now_wall=time.time(),
                    arrival_time=candidate.arrival_time,
                    ttft_slo_s=self.scheduler_config.prefill_bias_ttft_slo_s,
                    predicted_prefill_s=predicted_step_time_s,
                    slack_threshold_s=(
                        self.scheduler_config.prefill_bias_swap_slack_threshold_s
                    ),
                )
            )
""",
        """            predicted_step_time_s = self._prefill_bias_estimated_step_time()
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
""",
        "Phase 7 slot-swap deadline decision",
    )
    text = text.replace(
        "        if self.scheduler_config.prefill_bias_require_cache_residency and cached_tokens <= 0:\n",
        "        if config.prefill_bias_require_cache_residency and cached_tokens <= 0:\n",
        1,
    )
    text = text.replace(
        " + self.scheduler_config.prefill_bias_swap_failure_backoff_s\n",
        " + self._adaptive_prefill_effective_config().prefill_bias_swap_failure_backoff_s\n",
        1,
    )

    text = replace_once(
        text,
        """        decision = self.prefill_bias_controller.decide(
            waiting=self.waiting,
            policy=self.policy,
            paused=self._pause_state != PauseState.UNPAUSED,
            throttle_prefills=defer_prefills,
            max_safe_reserve=max_safe_reserve,
            is_request_schedulable=self._prefill_bias_request_is_schedulable,
        )
""",
        """        config = self._adaptive_prefill_effective_config()
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
        )
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
""",
        "Phase 7 deadline-aware scheduler decision",
    )

    text = replace_once(
        text,
        """        config = self._adaptive_prefill_effective_config()
        if not decision.active or not config.prefill_bias_cache_aware:
            return decision
""",
        """        config = self._adaptive_prefill_effective_config()
        if config.prefill_bias_ttft_deadline_enabled:
            return decision
        if not decision.active or not config.prefill_bias_cache_aware:
            return decision
""",
        "Phase 7 preserve deadline candidate after running",
    )

    return write_if_changed(path, text)


def patch_scheduler_phase8(path: Path) -> bool:
    text = path.read_text()
    if PHASE8_MARKER in text:
        return False

    text = replace_once(
        text,
        "import itertools\n",
        "import itertools\nimport math\n",
        "Phase 8 math import",
    )
    text = replace_once(
        text,
        "    PrefillBiasDecision,\n",
        "    PrefillBiasDecision,\n    PrefillBiasPolicyRouter,\n",
        "Phase 8 policy router import",
    )
    text = replace_once(
        text,
        """class ScheduledBatchTiming:
    started_at: float
    prefill_tokens: int
    total_tokens: int
""",
        """class ScheduledBatchTiming:
    started_at: float
    prefill_tokens: int
    total_tokens: int
    active_decode_count: int
""",
        "Phase 8 scheduled batch decode count",
    )
    text = replace_once(
        text,
        """        self.prefill_bias_controller = PrefillBiasController(
            self.scheduler_config,
            monotonic_clock=self._prefill_bias_monotonic_clock,
        )
""",
        """        # VLLM_PREFILL_BIAS_PHASE8_PATCH: runtime policy router.
        self.prefill_bias_controller = PrefillBiasPolicyRouter(
            self.scheduler_config,
            monotonic_clock=self._prefill_bias_monotonic_clock,
        )
""",
        "Phase 8 policy router init",
    )

    text = replace_once(
        text,
        """        self._prefill_bias_forced_preemption_limit_blocked_total = 0
        self._prefill_bias_last_decision = PrefillBiasDecision(
""",
        """        self._prefill_bias_forced_preemption_limit_blocked_total = 0
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
        self._prefill_bias_last_decision = PrefillBiasDecision(
""",
        "Phase 8 predictive metrics state",
    )

    predictive_helpers = """    def _prefill_bias_predictive_context(self) -> tuple[int, float, int]:
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

"""
    text = replace_once(
        text,
        "    def _prefill_bias_prepare(\n",
        predictive_helpers + "    def _prefill_bias_prepare(\n",
        "Phase 8 predictive scheduler helpers",
    )

    text = replace_once(
        text,
        """        config = self._adaptive_prefill_effective_config()
        deadline_start_ns = time.perf_counter_ns()
        decision = self.prefill_bias_controller.decide(
""",
        """        config = self._adaptive_prefill_effective_config()
        active_decode_count, minimum_tbt_slack_s, unknown_decode_count = (
            self._prefill_bias_predictive_context()
        )
        deadline_start_ns = time.perf_counter_ns()
        decision = self.prefill_bias_controller.decide(
""",
        "Phase 8 predictive context before decision",
    )
    text = replace_once(
        text,
        """            peek_cached_tokens=self.kv_cache_manager.peek_num_computed_tokens,
            predicted_step_time_s=self._prefill_bias_estimated_step_time(),
        )
        if config.prefill_bias_ttft_deadline_enabled:
""",
        """            peek_cached_tokens=self.kv_cache_manager.peek_num_computed_tokens,
            predicted_step_time_s=self._prefill_bias_estimated_step_time(),
            active_decode_count=active_decode_count,
            minimum_tbt_slack_s=minimum_tbt_slack_s,
            unknown_decode_count=unknown_decode_count,
        )
        self._prefill_bias_sync_predictive_metrics(decision)
        if config.prefill_bias_ttft_deadline_enabled:
""",
        "Phase 8 predictive decision inputs",
    )

    text = replace_once(
        text,
        """            tbt_slo_s=config.prefill_bias_tbt_slo_s,
            predicted_step_time_s=self._prefill_bias_estimated_step_time(),
            safety_margin_s=config.prefill_bias_tbt_safety_margin_s,
""",
        """            tbt_slo_s=config.prefill_bias_tbt_slo_s,
            predicted_step_time_s=(
                decision.predicted_step_time_s
                if decision.predicted_step_time_s > 0.0
                else self._prefill_bias_estimated_step_time()
            ),
            safety_margin_s=config.prefill_bias_tbt_safety_margin_s,
""",
        "Phase 8 decision-specific TBT prediction",
    )

    text = replace_once(
        text,
        """    def _prefill_bias_track_scheduled_batch(
        self,
        scheduler_output: SchedulerOutput,
        prefill_tokens: int,
    ) -> None:
""",
        """    def _prefill_bias_track_scheduled_batch(
        self,
        scheduler_output: SchedulerOutput,
        prefill_tokens: int,
        active_decode_count: int,
    ) -> None:
""",
        "Phase 8 track decode count signature",
    )
    text = replace_once(
        text,
        """            prefill_tokens=prefill_tokens,
            total_tokens=scheduler_output.total_num_scheduled_tokens,
        )
""",
        """            prefill_tokens=prefill_tokens,
            total_tokens=scheduler_output.total_num_scheduled_tokens,
            active_decode_count=active_decode_count,
        )
""",
        "Phase 8 track decode count value",
    )
    text = replace_once(
        text,
        """            self._prefill_bias_step_time_ewma_s = (
                self._prefill_bias_step_time_estimator.ewma_step_time_s or 0.0
            )

    def _prefill_slot_swap_reject_candidate(
""",
        """            self._prefill_bias_step_time_ewma_s = (
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
""",
        "Phase 8 predictive timing observation",
    )

    text = replace_once(
        text,
        """        prefill_tokens_in_batch = sum(
            num_tokens
            for request_id, num_tokens in num_scheduled_tokens.items()
            if self.requests[request_id].num_output_tokens == 0
        )

        # Check if the scheduling constraints are satisfied.
""",
        """        prefill_tokens_in_batch = sum(
            num_tokens
            for request_id, num_tokens in num_scheduled_tokens.items()
            if self.requests[request_id].num_output_tokens == 0
        )
        decode_requests_in_batch = sum(
            1
            for request_id in num_scheduled_tokens
            if self.requests[request_id].num_output_tokens > 0
        )

        # Check if the scheduling constraints are satisfied.
""",
        "Phase 8 decode request accounting",
    )
    text = replace_once(
        text,
        """        self._prefill_bias_track_scheduled_batch(
            scheduler_output,
            prefill_tokens_in_batch,
        )
""",
        """        self._prefill_bias_track_scheduled_batch(
            scheduler_output,
            prefill_tokens_in_batch,
            decode_requests_in_batch,
        )
""",
        "Phase 8 track scheduled decode count",
    )

    text = replace_once(
        text,
        """        prefill_bias_candidate_ids: set[str] = set()
        prefill_bias_swap_performed = False
""",
        """        prefill_bias_candidate_ids: set[str] = set()
        prefill_bias_candidate_token_caps: dict[str, int] = {}
        prefill_bias_swap_performed = False
""",
        "Phase 8 candidate token caps state",
    )
    text = replace_once(
        text,
        """            prefill_bias_candidate_ids = set(
                prefill_bias_decision.candidate_request_ids
            )
            token_budget -= prefill_bias_held_tokens
""",
        """            prefill_bias_candidate_ids = set(
                prefill_bias_decision.candidate_request_ids
            )
            prefill_bias_candidate_token_caps = dict(
                prefill_bias_decision.candidate_token_caps
            )
            token_budget -= prefill_bias_held_tokens
""",
        "Phase 8 initial token caps",
    )
    text = replace_once(
        text,
        """            prefill_bias_candidate_ids = set(
                prefill_bias_decision.candidate_request_ids
            )
            self._prefill_bias_promote_waiting(
""",
        """            prefill_bias_candidate_ids = set(
                prefill_bias_decision.candidate_request_ids
            )
            prefill_bias_candidate_token_caps = dict(
                prefill_bias_decision.candidate_token_caps
            )
            self._prefill_bias_promote_waiting(
""",
        "Phase 8 restored token caps",
    )
    text = replace_once(
        text,
        """                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
""",
        """                    num_new_tokens = min(num_new_tokens, token_budget)
                    candidate_cap = prefill_bias_candidate_token_caps.get(request_id)
                    if candidate_cap is not None:
                        num_new_tokens = min(num_new_tokens, candidate_cap)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
""",
        "Phase 8 waiting candidate token cap",
    )

    text = replace_once(
        text,
        """        candidate = self._prefill_slot_swap_candidate_from_decision(decision)
        if candidate is None:
""",
        """        candidate = self._prefill_slot_swap_candidate_from_decision(decision)
        if decision.policy_mode == "predictive" and not decision.slot_swap_eligible:
            self._prefill_slot_swap_reject_candidate(
                SlotSwapRejectReason.CANDIDATE_NOT_URGENT
            )
            return self._prefill_slot_swap_result(
                reject_reason=SlotSwapRejectReason.CANDIDATE_NOT_URGENT,
            ), token_budget
        if candidate is None:
""",
        "Phase 8 predictive slot-swap eligibility",
    )
    text = replace_once(
        text,
        "            predicted_step_time_s = self._prefill_bias_estimated_step_time()\n",
        """            predicted_step_time_s = (
                decision.predicted_step_time_s
                if decision.predicted_step_time_s > 0.0
                else self._prefill_bias_estimated_step_time()
            )
""",
        "Phase 8 slot-swap predicted chunk latency",
    )

    return write_if_changed(path, text)


def patch_scheduler_phase9(path: Path) -> bool:
    text = path.read_text()
    if PHASE9_MARKER in text:
        return False

    text = replace_once(
        text,
        "    PrefillBiasDecision,\n    PrefillBiasPolicyRouter,\n",
        "    PrefillBatchBudgetDecision,\n"
        "    PrefillBiasDecision,\n"
        "    PrefillBiasPolicyRouter,\n"
        "    PrefillBudgetAccountant,\n",
        "Phase 9 batch-budget imports",
    )

    text = replace_once(
        text,
        '''        self._prefill_bias_last_policy_mode = "legacy"
        self._prefill_bias_last_decision = PrefillBiasDecision(
''',
        '''        self._prefill_bias_last_policy_mode = "legacy"
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
''',
        "Phase 9 batch-budget metrics state",
    )

    batch_helpers = '''    def _prefill_batch_budget_mode(self) -> str | None:
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

'''
    text = replace_once(
        text,
        "    def _prefill_bias_prepare(\n",
        batch_helpers + "    def _prefill_bias_prepare(\n",
        "Phase 9 batch-budget scheduler helpers",
    )

    text = replace_once(
        text,
        '''        candidate = self._prefill_slot_swap_candidate_from_decision(decision)
        if decision.policy_mode == "predictive" and not decision.slot_swap_eligible:
''',
        '''        candidate = self._prefill_slot_swap_candidate_from_decision(decision)
        if (
            decision.policy_mode in {"predictive", "batch-budget"}
            and not decision.slot_swap_eligible
        ):
''',
        "Phase 9 batch-budget slot-swap eligibility",
    )
    text = replace_once(
        text,
        '''        scheduled_encoder_inputs: dict[str, list[int]],
        swaps_this_step: int,
    ) -> tuple[PrefillAdmissionSwapResult, int]:
''',
        '''        scheduled_encoder_inputs: dict[str, list[int]],
        swaps_this_step: int,
        prefill_budget_accountant: PrefillBudgetAccountant | None = None,
    ) -> tuple[PrefillAdmissionSwapResult, int]:
''',
        "Phase 9 slot-swap accounting parameter",
    )
    text = replace_once(
        text,
        '''        if victim in scheduled_running_reqs:
            scheduled_running_reqs.remove(victim)
            token_budget += num_scheduled_tokens.pop(victim_id, 0)
            req_to_new_blocks.pop(victim_id, None)
''',
        '''        if victim in scheduled_running_reqs:
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
''',
        "Phase 9 slot-swap prefill budget restore",
    )

    text = replace_once(
        text,
        '''        # VLLM_PREFILL_BIAS_PATCH: reserve a safe slice for urgent waiting prefills.
        prefill_bias_held_tokens = 0
        prefill_bias_candidate_ids: set[str] = set()
        prefill_bias_candidate_token_caps: dict[str, int] = {}
        prefill_bias_swap_performed = False
        prefill_bias_swap_candidate_id: str | None = None
        prefill_bias_swaps_this_step = 0
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
''',
        '''        # VLLM_PREFILL_BIAS_PATCH: reserve a safe slice for urgent waiting prefills.
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
''',
        "Phase 9 batch-budget schedule prelude",
    )

    text = replace_once(
        text,
        '''            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            num_new_tokens = (
''',
        '''            if defer_prefills and request.is_prefill_chunk:
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
''',
        "Phase 9 skip unselected running prefills",
    )
    text = replace_once(
        text,
        '''            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )

            # Schedule encoder inputs.
''',
        '''            num_new_tokens = min(
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
''',
        "Phase 9 clamp running prefill",
    )
    text = replace_once(
        text,
        '''                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
''',
        '''                            restored_tokens = num_scheduled_tokens.pop(
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
''',
        "Phase 9 running preemption budget restore",
    )
    text = replace_once(
        text,
        '''            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1
''',
        '''            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            if batch_budget_enforced and request.is_prefill_chunk:
                batch_budget_accountant.commit(
                    request_id,
                    num_new_tokens,
                    source="running",
                )
            req_index += 1
''',
        "Phase 9 commit running prefill budget",
    )

    text = replace_once(
        text,
        '''            scheduled_encoder_inputs=scheduled_encoder_inputs,
            swaps_this_step=prefill_bias_swaps_this_step,
        )
''',
        '''            scheduled_encoder_inputs=scheduled_encoder_inputs,
            swaps_this_step=prefill_bias_swaps_this_step,
            prefill_budget_accountant=batch_budget_accountant,
        )
''',
        "Phase 9 pass slot-swap accountant",
    )

    text = replace_once(
        text,
        '''                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                elif defer_prefills and request.num_computed_tokens == 0:
''',
        '''                if load_kv_async:
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
                elif defer_prefills and request.num_computed_tokens == 0:
''',
        "Phase 9 skip unselected waiting compute",
    )
    text = replace_once(
        text,
        '''                    candidate_cap = prefill_bias_candidate_token_caps.get(request_id)
                    if candidate_cap is not None:
                        num_new_tokens = min(num_new_tokens, candidate_cap)
                    assert num_new_tokens > 0
''',
        '''                    candidate_cap = prefill_bias_candidate_token_caps.get(request_id)
                    if candidate_cap is not None:
                        num_new_tokens = min(num_new_tokens, candidate_cap)
                    if batch_budget_enforced:
                        num_new_tokens = batch_budget_accountant.clamp(
                            request_id,
                            num_new_tokens,
                        )
                    assert num_new_tokens > 0
''',
        "Phase 9 clamp waiting prefill",
    )
    text = replace_once(
        text,
        '''                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
''',
        '''                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                if batch_budget_enforced:
                    batch_budget_accountant.commit(
                        request_id,
                        num_new_tokens,
                        source="waiting",
                    )
                request.status = RequestStatus.RUNNING
''',
        "Phase 9 commit waiting prefill budget",
    )

    text = replace_once(
        text,
        '''        prefill_tokens_in_batch = sum(
            num_tokens
            for request_id, num_tokens in num_scheduled_tokens.items()
            if self.requests[request_id].num_output_tokens == 0
        )
        decode_requests_in_batch = sum(
            1
            for request_id in num_scheduled_tokens
            if self.requests[request_id].num_output_tokens > 0
        )

        # Check if the scheduling constraints are satisfied.
''',
        '''        prefill_tokens_in_batch = sum(
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
''',
        "Phase 9 final prefill accounting and metrics",
    )

    return write_if_changed(path, text)


def patch_all(vllm_root: Path, *, skip_hash_check: bool = False) -> list[Path]:
    targets = find_targets(vllm_root)
    validate_targets(targets, skip_hash_check=skip_hash_check)
    policy_path = vllm_root / "vllm" / "v1" / "core" / "sched" / "prefill_bias.py"
    backup_paths = set(targets.values()) | {policy_path}
    backups = {
        path: path.read_bytes() if path.exists() else None for path in backup_paths
    }
    changed: list[Path] = []
    try:
        changed.append(copy_policy_module(vllm_root))
        for func, key in (
            (patch_scheduler_config, "scheduler_config"),
            (patch_scheduler_config_phase5, "scheduler_config"),
            (patch_scheduler_config_phase2, "scheduler_config"),
            (patch_scheduler_config_phase3, "scheduler_config"),
            (patch_scheduler_config_phase4, "scheduler_config"),
            (patch_scheduler_config_phase6, "scheduler_config"),
            (patch_scheduler_config_phase7, "scheduler_config"),
            (patch_scheduler_config_phase8, "scheduler_config"),
            (patch_scheduler_config_phase9, "scheduler_config"),
            (patch_arg_utils, "arg_utils"),
            (patch_arg_utils_phase5, "arg_utils"),
            (patch_arg_utils_phase2, "arg_utils"),
            (patch_arg_utils_phase3, "arg_utils"),
            (patch_arg_utils_phase4, "arg_utils"),
            (patch_arg_utils_phase6, "arg_utils"),
            (patch_arg_utils_phase7, "arg_utils"),
            (patch_arg_utils_phase8, "arg_utils"),
            (patch_arg_utils_phase9, "arg_utils"),
            (patch_kv_cache_manager, "kv_cache_manager"),
            (patch_scheduler, "scheduler"),
            (patch_scheduler_phase2, "scheduler"),
            (patch_scheduler_phase3, "scheduler"),
            (patch_scheduler_phase4, "scheduler"),
            (patch_scheduler_phase6, "scheduler"),
            (patch_scheduler_phase7, "scheduler"),
            (patch_scheduler_phase8, "scheduler"),
            (patch_scheduler_phase9, "scheduler"),
        ):
            path = targets[key]
            if func(path):
                changed.append(path)
        for path in set(changed):
            py_compile.compile(str(path), doraise=True)
    except Exception:
        for path, content in backups.items():
            if content is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(content)
        raise
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "vllm_root",
        nargs="?",
        help="Path to a vLLM checkout root or installed package root.",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Patch by structural anchors without exact source hash validation.",
    )
    args = parser.parse_args()

    try:
        vllm_root = find_vllm_root(args.vllm_root)
        changed = patch_all(vllm_root, skip_hash_check=args.skip_hash_check)
    except Exception as exc:
        raise SystemExit(f"Prefill-bias patch failed: {exc}") from exc

    for path in changed:
        print(f"patched {path}")


if __name__ == "__main__":
    main()

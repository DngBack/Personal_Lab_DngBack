#!/usr/bin/env python3
"""Enable tuned concurrent partial prefill for vLLM 0.24.x images.

The upstream v0.24.0 image exposes max_num_partial_prefills and
max_long_partial_prefills in SchedulerConfig, but arg_utils rejects non-default
values at startup. This patch removes that guard and adds scheduler-side logic
to chunk multiple active prefills fairly, including all-long tail waves.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ARG_UTILS_GUARD = """        # No Concurrent Partial Prefills so far.
        if (
            self.max_num_partial_prefills != SchedulerConfig.max_num_partial_prefills
            or self.max_long_partial_prefills
            != SchedulerConfig.max_long_partial_prefills
        ):
            _raise_unsupported_error(feature_name="Concurrent Partial Prefill")

"""

DATACLASSES = '''
@dataclass
class PartialPrefillMetadata:
    """Track partial prefill pressure within a scheduler step."""

    schedulable_prefills: int
    active_prefills: int
    long_prefills: int
    max_prefills: int
    max_long_prefills: int
    long_prefill_threshold: int

    def can_schedule_prefill(self, remaining_tokens: int) -> bool:
        if self.active_prefills >= self.max_prefills:
            return False
        if (
            remaining_tokens > self.long_prefill_threshold
            and self.long_prefills >= self.max_long_prefills
        ):
            return False
        return True

    def record_partial_prefill(self, remaining_tokens: int) -> None:
        self.active_prefills += 1
        if remaining_tokens > self.long_prefill_threshold:
            self.long_prefills += 1


@dataclass
class PrefillState:
    is_prefill: bool
    remaining_tokens: int
    is_long_prefill: bool


'''

INIT_BLOCK = '''
        max_partial_prefills = self.scheduler_config.max_num_partial_prefills
        budget_list_size = max(1, max_partial_prefills) + 1
        self._prefill_slot_budgets = [self.max_num_scheduled_tokens] * budget_list_size
        for prefill_slots in range(1, budget_list_size):
            self._prefill_slot_budgets[prefill_slots] = max(
                1, self.max_num_scheduled_tokens // prefill_slots
            )
        self.enable_concurrent_partial_prefill_scheduling = (
            max_partial_prefills > 1
            and self.scheduler_config.enable_chunked_prefill
        )
        self.enable_short_prefill_priority = self.scheduler_config.enable_chunked_prefill
        self._waiting_prefill_scan_limit = 256

'''

SCHEDULE_SETUP = '''
        partial_prefill_metadata = None
        partial_prefill_slot_budget = None
        if self.enable_concurrent_partial_prefill_scheduling:
            partial_prefill_metadata = self._build_partial_prefill_metadata()
            partial_prefill_slot_budget = self._get_prefill_slot_budget(
                partial_prefill_metadata
            )

'''

RUNNING_SLOT_CAP = '''
            if (
                self.enable_concurrent_partial_prefill_scheduling
                and partial_prefill_slot_budget is not None
            ):
                running_prefill_state = self._get_request_prefill_state(
                    request, request.num_computed_tokens
                )
                if running_prefill_state.is_prefill:
                    num_new_tokens = min(num_new_tokens, partial_prefill_slot_budget)

'''

WAITING_REORDER = '''
                if self.enable_short_prefill_priority:
                    self._reorder_waiting_for_short_prefills(
                        self.waiting, partial_prefill_metadata
                    )

'''

WAITING_STATE_AND_SKIP = '''
                waiting_prefill_state = None
                if (
                    self.enable_concurrent_partial_prefill_scheduling
                    or self.enable_short_prefill_priority
                ):
                    waiting_prefill_state = self._get_request_prefill_state(
                        request, num_computed_tokens
                    )
                if (
                    self.enable_concurrent_partial_prefill_scheduling
                    and waiting_prefill_state is not None
                    and waiting_prefill_state.is_prefill
                    and partial_prefill_metadata is not None
                    and not partial_prefill_metadata.can_schedule_prefill(
                        waiting_prefill_state.remaining_tokens
                    )
                ):
                    request_queue.pop_request()
                    step_skipped_waiting.add_request(request)
                    continue

'''

WAITING_SLOT_CAP = '''
                    if (
                        self.enable_concurrent_partial_prefill_scheduling
                        and waiting_prefill_state is not None
                        and waiting_prefill_state.is_prefill
                        and partial_prefill_slot_budget is not None
                    ):
                        num_new_tokens = min(num_new_tokens, partial_prefill_slot_budget)

'''

WAITING_RECORD = '''
                if (
                    self.enable_concurrent_partial_prefill_scheduling
                    and partial_prefill_metadata is not None
                    and waiting_prefill_state is not None
                    and waiting_prefill_state.is_prefill
                    and num_computed_tokens + num_new_tokens < request.num_tokens
                ):
                    partial_prefill_metadata.record_partial_prefill(
                        waiting_prefill_state.remaining_tokens
                    )

'''

HELPERS = '''
    def _reorder_waiting_for_short_prefills(
        self,
        request_queue: RequestQueue,
        metadata: PartialPrefillMetadata | None,
    ) -> None:
        """Promote the shortest schedulable prefill near the head of FCFS waves."""
        if self.policy != SchedulingPolicy.FCFS or not request_queue:
            return

        scan_limit = min(len(request_queue), self._waiting_prefill_scan_limit)
        if scan_limit <= 1:
            return

        candidates: list[tuple[int, int, Request]] = []
        for idx, request in enumerate(request_queue):
            if idx >= scan_limit:
                break
            if self._is_blocked_waiting_status(request.status):
                continue
            prefill_state = self._get_request_prefill_state(
                request, request.num_computed_tokens
            )
            if not prefill_state.is_prefill:
                continue
            if (
                metadata is not None
                and not metadata.can_schedule_prefill(prefill_state.remaining_tokens)
            ):
                continue
            candidates.append((prefill_state.remaining_tokens, idx, request))

        if not candidates:
            return

        candidates.sort(key=lambda item: (item[0], item[1]))
        _, best_idx, best_request = candidates[0]
        if best_idx == 0:
            return

        request_queue.remove_request(best_request)
        request_queue.prepend_request(best_request)

    def _build_partial_prefill_metadata(self) -> PartialPrefillMetadata:
        max_prefills = self.scheduler_config.max_num_partial_prefills
        max_long_prefills = self.scheduler_config.max_long_partial_prefills
        threshold = self.scheduler_config.long_prefill_token_threshold

        active_prefills = 0
        active_long_prefills = 0
        for request in self.running:
            prefill_state = self._get_request_prefill_state(
                request, request.num_computed_tokens
            )
            if not prefill_state.is_prefill:
                continue
            active_prefills += 1
            if prefill_state.is_long_prefill:
                active_long_prefills += 1

        schedulable_prefills = active_prefills
        planned_long_prefills = active_long_prefills
        waiting_queues = (self.skipped_waiting, self.waiting)
        for request_queue in waiting_queues:
            for request in request_queue:
                if schedulable_prefills >= max_prefills:
                    break
                if self._is_blocked_waiting_status(request.status):
                    continue
                prefill_state = self._get_request_prefill_state(
                    request, request.num_computed_tokens
                )
                if not prefill_state.is_prefill:
                    continue
                if (
                    prefill_state.is_long_prefill
                    and planned_long_prefills >= max_long_prefills
                ):
                    continue
                schedulable_prefills += 1
                if prefill_state.is_long_prefill:
                    planned_long_prefills += 1
            if schedulable_prefills >= max_prefills:
                break

        return PartialPrefillMetadata(
            schedulable_prefills=max(1, min(schedulable_prefills, max_prefills)),
            active_prefills=active_prefills,
            long_prefills=active_long_prefills,
            max_prefills=max_prefills,
            max_long_prefills=max_long_prefills,
            long_prefill_threshold=threshold,
        )

    def _get_prefill_slot_budget(
        self, metadata: PartialPrefillMetadata
    ) -> int | None:
        index = min(
            metadata.schedulable_prefills,
            len(self._prefill_slot_budgets) - 1,
        )
        return self._prefill_slot_budgets[index]

    def _get_request_prefill_state(
        self, request: Request, num_computed_tokens: int
    ) -> PrefillState:
        remaining_tokens = max(request.num_prompt_tokens - num_computed_tokens, 0)
        is_prefill = (
            request.num_output_tokens == 0
            and num_computed_tokens < request.num_prompt_tokens
        )
        is_long_prefill = (
            is_prefill
            and remaining_tokens > self.scheduler_config.long_prefill_token_threshold
        )
        return PrefillState(
            is_prefill=is_prefill,
            remaining_tokens=remaining_tokens,
            is_long_prefill=is_long_prefill,
        )

'''


def vllm_package_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vllm is not installed in this environment")
    return Path(spec.origin).resolve().parent


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"anchor not found for {label}")
    return text.replace(old, new, 1)


def patch_arg_utils(vllm_root: Path) -> None:
    path = vllm_root / "engine" / "arg_utils.py"
    text = path.read_text(encoding="utf-8")
    if ARG_UTILS_GUARD not in text:
        if "Concurrent Partial Prefill" not in text:
            print(f"[patch] arg_utils already patched: {path}", flush=True)
            return
        raise RuntimeError(f"unexpected arg_utils layout: {path}")
    path.write_text(text.replace(ARG_UTILS_GUARD, ""), encoding="utf-8")
    print(f"[patch] removed startup guard: {path}", flush=True)


def patch_scheduler(vllm_root: Path) -> None:
    path = vllm_root / "v1" / "core" / "sched" / "scheduler.py"
    text = path.read_text(encoding="utf-8")
    if "class PartialPrefillMetadata" in text:
        print(f"[patch] scheduler already patched: {path}", flush=True)
        return

    text = replace_once(
        text,
        "from dataclasses import replace\n",
        "from dataclasses import dataclass, replace\n",
        "dataclass import",
    )
    text = replace_once(
        text,
        "logger = init_logger(__name__)\n\n\nclass Scheduler(SchedulerInterface):",
        f"logger = init_logger(__name__)\n\n{DATACLASSES}\nclass Scheduler(SchedulerInterface):",
        "metadata classes",
    )
    text = replace_once(
        text,
        "        self.scheduler_reserve_full_isl = (\n"
        "            self.scheduler_config.scheduler_reserve_full_isl\n"
        "        )\n\n"
        "        self.has_mamba_layers = kv_cache_config.has_mamba_layers\n",
        "        self.scheduler_reserve_full_isl = (\n"
        "            self.scheduler_config.scheduler_reserve_full_isl\n"
        "        )\n"
        f"{INIT_BLOCK}\n"
        "        self.has_mamba_layers = kv_cache_config.has_mamba_layers\n",
        "scheduler init",
    )
    text = replace_once(
        text,
        "        scheduled_timestamp = time.monotonic()\n\n"
        "        self.kv_cache_manager.new_step_starts()\n",
        "        scheduled_timestamp = time.monotonic()\n"
        f"{SCHEDULE_SETUP}\n"
        "        self.kv_cache_manager.new_step_starts()\n",
        "schedule setup",
    )
    text = replace_once(
        text,
        "            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n"
        "                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
        "            num_new_tokens = min(num_new_tokens, token_budget)\n\n"
        "            # Make sure the input position does not exceed the max model len.\n",
        "            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n"
        "                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
        "            num_new_tokens = min(num_new_tokens, token_budget)\n"
        f"{RUNNING_SLOT_CAP}\n"
        "            # Make sure the input position does not exceed the max model len.\n",
        "running prefill cap",
    )
    text = replace_once(
        text,
        "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
        "                if len(self.running) == self.max_num_running_reqs:\n"
        "                    break\n\n"
        "                request_queue = self._select_waiting_queue_for_scheduling()\n",
        "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
        "                if len(self.running) == self.max_num_running_reqs:\n"
        "                    break\n"
        f"{WAITING_REORDER}\n"
        "                request_queue = self._select_waiting_queue_for_scheduling()\n",
        "waiting short-prefill reorder",
    )
    text = replace_once(
        text,
        "                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks\n"
        "                    num_new_local_computed_tokens = 0\n"
        "                    num_computed_tokens = request.num_computed_tokens\n\n"
        "                encoder_inputs_to_schedule = None\n",
        "                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks\n"
        "                    num_new_local_computed_tokens = 0\n"
        "                    num_computed_tokens = request.num_computed_tokens\n\n"
        f"{WAITING_STATE_AND_SKIP}\n"
        "                encoder_inputs_to_schedule = None\n",
        "waiting prefill quota",
    )
    text = replace_once(
        text,
        "                    num_new_tokens = min(num_new_tokens, token_budget)\n"
        "                    assert num_new_tokens > 0\n\n"
        "                    # Schedule encoder inputs.\n",
        "                    num_new_tokens = min(num_new_tokens, token_budget)\n"
        f"{WAITING_SLOT_CAP}\n"
        "                    assert num_new_tokens > 0\n\n"
        "                    # Schedule encoder inputs.\n",
        "waiting prefill cap",
    )
    text = replace_once(
        text,
        "                request.status = RequestStatus.RUNNING\n"
        "                request.num_computed_tokens = num_computed_tokens\n"
        "                # Only track requests that will still be prefilling after this chunk.\n",
        "                request.status = RequestStatus.RUNNING\n"
        "                request.num_computed_tokens = num_computed_tokens\n"
        f"{WAITING_RECORD}\n"
        "                # Only track requests that will still be prefilling after this chunk.\n",
        "waiting prefill record",
    )
    text = replace_once(
        text,
        "    def _mamba_block_aligned_split(\n",
        f"{HELPERS}\n"
        "    def _mamba_block_aligned_split(\n",
        "helper methods",
    )
    path.write_text(text, encoding="utf-8")
    print(f"[patch] applied scheduler patch: {path}", flush=True)


def check(vllm_root: Path) -> None:
    arg_utils = (vllm_root / "engine" / "arg_utils.py").read_text(encoding="utf-8")
    scheduler = (
        vllm_root / "v1" / "core" / "sched" / "scheduler.py"
    ).read_text(encoding="utf-8")
    if "Concurrent Partial Prefill" in arg_utils:
        raise RuntimeError("arg_utils guard is still present")
    required = [
        "class PartialPrefillMetadata",
        "enable_concurrent_partial_prefill_scheduling",
        "_reorder_waiting_for_short_prefills",
        "can_schedule_prefill",
    ]
    missing = [item for item in required if item not in scheduler]
    if missing:
        raise RuntimeError(f"scheduler patch missing markers: {missing}")
    print("patched", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve() if args.root is not None else vllm_package_root()
    if args.check:
        check(root)
        return 0
    patch_arg_utils(root)
    patch_scheduler(root)
    check(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

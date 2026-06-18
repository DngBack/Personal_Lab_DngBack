"""Surgical concurrent-partial-prefill port for vLLM 0.23 scheduler."""

from __future__ import annotations

DATACLASSES = '''
@dataclass
class PartialPrefillMetadata:
    """Track concurrent partial prefills for one scheduler step."""

    schedulable_prefills: int
    long_prefills: int
    active_prefills: int
    max_long_prefills: int
    max_prefills: int
    long_prefill_threshold: int

    def can_schedule(self, remaining_tokens: int) -> bool:
        if remaining_tokens <= self.long_prefill_threshold:
            return True
        return self.long_prefills < self.max_long_prefills

    def record_new_prefill(self, remaining_tokens: int) -> None:
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
        self.enable_short_prefill_priority = (
            self.scheduler_config.enable_chunked_prefill
        )
        self._waiting_prefill_scan_limit = 256
        self.enable_concurrent_partial_prefill_scheduling = (
            max_partial_prefills > 1
            and self.scheduler_config.enable_chunked_prefill
        )

'''

SCHEDULE_SETUP = '''
        partial_prefill_metadata = None
        partial_prefill_slot_budget = None
        if self.enable_concurrent_partial_prefill_scheduling:
            partial_prefill_metadata = self._build_partial_prefill_metadata()
            partial_prefill_slot_budget = self._get_prefill_slot_budget(
                partial_prefill_metadata
            )
        elif self.enable_short_prefill_priority:
            partial_prefill_metadata = self._build_partial_prefill_metadata()

'''

REORDER_BEFORE_WAITING_LOOP = '''
                if self.enable_short_prefill_priority:
                    self._reorder_waiting_for_short_prefills(
                        partial_prefill_metadata
                    )

'''

RUNNING_SLOT_CAP = '''
            if (
                self.enable_concurrent_partial_prefill_scheduling
                and partial_prefill_slot_budget is not None
            ):
                running_req_prefill_state = self._get_request_prefill_state(
                    request, request.num_computed_tokens
                )
                if running_req_prefill_state.is_prefill:
                    num_new_tokens = min(num_new_tokens, partial_prefill_slot_budget)

'''

WAITING_STATE_AND_SKIP = '''
                waiting_req_prefill_state = (
                    self._get_request_prefill_state(request, num_computed_tokens)
                    if (
                        self.enable_concurrent_partial_prefill_scheduling
                        or self.enable_short_prefill_priority
                    )
                    else None
                )
                if (
                    self.enable_concurrent_partial_prefill_scheduling
                    and waiting_req_prefill_state is not None
                    and waiting_req_prefill_state.is_prefill
                    and partial_prefill_metadata is not None
                    and not partial_prefill_metadata.can_schedule(
                        waiting_req_prefill_state.remaining_tokens
                    )
                ):
                    request_queue.pop_request()
                    step_skipped_waiting.add_request(request)
                    continue

'''

WAITING_SLOT_CAP = '''
                    if (
                        self.enable_concurrent_partial_prefill_scheduling
                        and waiting_req_prefill_state is not None
                        and waiting_req_prefill_state.is_prefill
                        and partial_prefill_slot_budget is not None
                    ):
                        num_new_tokens = min(num_new_tokens, partial_prefill_slot_budget)

'''

WAITING_RECORD = '''
                if (
                    self.enable_concurrent_partial_prefill_scheduling
                    and partial_prefill_metadata is not None
                    and waiting_req_prefill_state is not None
                    and waiting_req_prefill_state.is_prefill
                    and waiting_req_prefill_state.remaining_tokens > 0
                ):
                    partial_prefill_metadata.record_new_prefill(
                        waiting_req_prefill_state.remaining_tokens
                    )

'''

HELPER_METHODS = '''
    def _reorder_waiting_for_short_prefills(
        self, metadata: PartialPrefillMetadata | None
    ) -> None:
        """Promote the shortest schedulable prefill to the queue head."""
        if not self.waiting:
            return

        scan_limit = min(len(self.waiting), self._waiting_prefill_scan_limit)
        if scan_limit <= 1:
            return

        candidates: list[tuple[int, int, Request]] = []
        for idx, request in enumerate(self.waiting):
            if idx >= scan_limit:
                break
            prefill_state = self._get_request_prefill_state(
                request, request.num_computed_tokens
            )
            if not prefill_state.is_prefill:
                continue
            if (
                metadata is not None
                and not metadata.can_schedule(prefill_state.remaining_tokens)
            ):
                continue
            candidates.append(
                (prefill_state.remaining_tokens, idx, request)
            )

        if not candidates:
            return

        candidates.sort(key=lambda item: (item[0], item[1]))
        best_remaining, best_idx, best_request = candidates[0]
        if best_idx == 0:
            return

        self.waiting.remove_request(best_request)
        self.waiting.prepend_request(best_request)

    def _build_partial_prefill_metadata(self) -> PartialPrefillMetadata:
        max_partial_prefills = self.scheduler_config.max_num_partial_prefills
        long_limit = self.scheduler_config.max_long_partial_prefills
        threshold = self.scheduler_config.long_prefill_token_threshold

        active_prefills = 0
        long_prefills = 0
        for request in self.running:
            prefill_state = self._get_request_prefill_state(
                request, request.num_computed_tokens
            )
            if prefill_state.is_prefill:
                active_prefills += 1
                if prefill_state.is_long_prefill:
                    long_prefills += 1

        prefills = active_prefills
        waiting_long_prefills = 0
        for request in self.waiting:
            if prefills >= max_partial_prefills:
                break
            prefill_state = self._get_request_prefill_state(
                request, request.num_computed_tokens
            )
            if not prefill_state.is_prefill:
                continue
            if (
                prefill_state.is_long_prefill
                and (long_prefills + waiting_long_prefills) >= long_limit
            ):
                continue
            if prefill_state.is_long_prefill:
                waiting_long_prefills += 1
            prefills += 1

        return PartialPrefillMetadata(
            schedulable_prefills=min(prefills, max_partial_prefills),
            long_prefills=long_prefills,
            active_prefills=active_prefills,
            max_long_prefills=long_limit,
            max_prefills=max_partial_prefills,
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
        remaining_tokens = self._remaining_prefill_tokens_with_tokens(
            request, num_computed_tokens
        )
        is_prefill = self._is_prefill_with_tokens(request, num_computed_tokens)
        threshold = self.scheduler_config.long_prefill_token_threshold
        is_long_prefill = is_prefill and remaining_tokens > threshold
        return PrefillState(
            is_prefill=is_prefill,
            remaining_tokens=remaining_tokens,
            is_long_prefill=is_long_prefill,
        )

    @staticmethod
    def _is_prefill_with_tokens(request: Request, num_computed_tokens: int) -> bool:
        return (
            request.num_output_tokens == 0
            and num_computed_tokens < request.num_prompt_tokens
        )

    @staticmethod
    def _remaining_prefill_tokens_with_tokens(
        request: Request, num_computed_tokens: int
    ) -> int:
        return max(request.num_prompt_tokens - num_computed_tokens, 0)

'''


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise ValueError(f"anchor not found for {label}")
    return text.replace(old, new, 1)


def apply_scheduler_patch(scheduler_path) -> None:
    text = scheduler_path.read_text(encoding="utf-8")
    if "enable_short_prefill_priority" in text:
        # Upgrade scan_limit from 64 to 256 if this is an older patched scheduler.
        if "_waiting_prefill_scan_limit = 64" in text:
            text = text.replace(
                "_waiting_prefill_scan_limit = 64",
                "_waiting_prefill_scan_limit = 256",
            )
            scheduler_path.write_text(text, encoding="utf-8")
            print(f"[patch] upgraded scan_limit 64→256: {scheduler_path}", flush=True)
        else:
            print(f"[patch] scheduler already up to date (scan_limit=256): {scheduler_path}")
        return

    # Legacy patch without short-priority: upgrade in place.
    if "enable_concurrent_partial_prefill_scheduling" in text:
        text = text.replace(
            "step_skipped_waiting.prepend_request(request)\n                    continue\n\n"
            "                encoder_inputs_to_schedule = None\n",
            "step_skipped_waiting.add_request(request)\n                    continue\n\n"
            "                encoder_inputs_to_schedule = None\n",
            1,
        )
        if SCHEDULE_SETUP.strip() not in text:
            text = _replace_once(
                text,
                "        if self.enable_concurrent_partial_prefill_scheduling:\n"
                "            partial_prefill_metadata = self._build_partial_prefill_metadata()\n"
                "            partial_prefill_slot_budget = self._get_prefill_slot_budget(\n"
                "                partial_prefill_metadata\n"
                "            )\n"
                "        else:\n"
                "            partial_prefill_metadata = None\n"
                "            partial_prefill_slot_budget = None\n",
                SCHEDULE_SETUP,
                "schedule setup upgrade",
            )
        if "_reorder_waiting_for_short_prefills" not in text:
            text = _replace_once(
                text,
                "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
                "                if len(self.running) == self.max_num_running_reqs:\n"
                "                    break\n\n"
                "                request_queue = self._select_waiting_queue_for_scheduling()\n",
                "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
                "                if len(self.running) == self.max_num_running_reqs:\n"
                "                    break\n"
                f"{REORDER_BEFORE_WAITING_LOOP}\n"
                "                request_queue = self._select_waiting_queue_for_scheduling()\n",
                "reorder inside waiting loop",
            )
            text = _replace_once(
                text,
                "        self.enable_concurrent_partial_prefill_scheduling = (\n"
                "            max_partial_prefills > 1\n"
                "            and self.scheduler_config.enable_chunked_prefill\n"
                "        )\n",
                "        self.enable_short_prefill_priority = (\n"
                "            self.scheduler_config.enable_chunked_prefill\n"
                "        )\n"
                "        self._waiting_prefill_scan_limit = 256\n"
                "        self.enable_concurrent_partial_prefill_scheduling = (\n"
                "            max_partial_prefills > 1\n"
                "            and self.scheduler_config.enable_chunked_prefill\n"
                "        )\n",
                "init short priority upgrade",
            )
            text = _replace_once(
                text,
                "    def _build_partial_prefill_metadata(self) -> PartialPrefillMetadata:\n",
                "    def _reorder_waiting_for_short_prefills(\n"
                "        self, metadata: PartialPrefillMetadata | None\n"
                "    ) -> None:\n"
                '        """Promote the shortest schedulable prefill to the queue head."""\n'
                "        if not self.waiting:\n"
                "            return\n\n"
                "        scan_limit = min(len(self.waiting), self._waiting_prefill_scan_limit)\n"
                "        if scan_limit <= 1:\n"
                "            return\n\n"
                "        candidates: list[tuple[int, int, Request]] = []\n"
                "        for idx, request in enumerate(self.waiting):\n"
                "            if idx >= scan_limit:\n"
                "                break\n"
                "            prefill_state = self._get_request_prefill_state(\n"
                "                request, request.num_computed_tokens\n"
                "            )\n"
                "            if not prefill_state.is_prefill:\n"
                "                continue\n"
                "            if (\n"
                "                metadata is not None\n"
                "                and not metadata.can_schedule(prefill_state.remaining_tokens)\n"
                "            ):\n"
                "                continue\n"
                "            candidates.append(\n"
                "                (prefill_state.remaining_tokens, idx, request)\n"
                "            )\n\n"
                "        if not candidates:\n"
                "            return\n\n"
                "        candidates.sort(key=lambda item: (item[0], item[1]))\n"
                "        best_remaining, best_idx, best_request = candidates[0]\n"
                "        if best_idx == 0:\n"
                "            return\n\n"
                "        self.waiting.remove_request(best_request)\n"
                "        self.waiting.prepend_request(best_request)\n\n"
                "    def _build_partial_prefill_metadata(self) -> PartialPrefillMetadata:\n",
                "reorder helper upgrade",
            )
        scheduler_path.write_text(text, encoding="utf-8")
        print(f"[patch] upgraded scheduler for short-prefill priority: {scheduler_path}", flush=True)
        return

    text = _replace_once(
        text,
        "from dataclasses import replace\n",
        "from dataclasses import dataclass, replace\n",
        "dataclass import",
    )
    text = _replace_once(
        text,
        "logger = init_logger(__name__)\n\n\nclass Scheduler(SchedulerInterface):",
        f"logger = init_logger(__name__)\n\n{DATACLASSES}\nclass Scheduler(SchedulerInterface):",
        "dataclass block",
    )
    text = _replace_once(
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
        "init block",
    )
    text = _replace_once(
        text,
        "        scheduled_timestamp = time.monotonic()\n\n"
        "        self.kv_cache_manager.new_step_starts()\n",
        "        scheduled_timestamp = time.monotonic()\n"
        f"{SCHEDULE_SETUP}\n"
        "        self.kv_cache_manager.new_step_starts()\n",
        "schedule setup",
    )
    text = _replace_once(
        text,
        "            step_skipped_waiting = create_request_queue(self.policy)\n\n"
        "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
        "                if len(self.running) == self.max_num_running_reqs:\n"
        "                    break\n\n"
        "                request_queue = self._select_waiting_queue_for_scheduling()\n",
        "            step_skipped_waiting = create_request_queue(self.policy)\n\n"
        "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
        "                if len(self.running) == self.max_num_running_reqs:\n"
        "                    break\n"
        f"{REORDER_BEFORE_WAITING_LOOP}\n"
        "                request_queue = self._select_waiting_queue_for_scheduling()\n",
        "reorder inside waiting loop",
    )
    text = _replace_once(
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
        "running slot cap",
    )
    text = _replace_once(
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
        "waiting skip",
    )
    text = _replace_once(
        text,
        "                    num_new_tokens = min(num_new_tokens, token_budget)\n"
        "                    assert num_new_tokens > 0\n\n"
        "                    # Schedule encoder inputs.\n"
        "                    if request.has_encoder_inputs:\n",
        "                    num_new_tokens = min(num_new_tokens, token_budget)\n"
        f"{WAITING_SLOT_CAP}\n"
        "                    assert num_new_tokens > 0\n\n"
        "                    # Schedule encoder inputs.\n"
        "                    if request.has_encoder_inputs:\n",
        "waiting slot cap",
    )
    text = _replace_once(
        text,
        "                request.status = RequestStatus.RUNNING\n"
        "                request.num_computed_tokens = num_computed_tokens\n"
        "                # Only track requests that will still be prefilling after this chunk.\n",
        "                request.status = RequestStatus.RUNNING\n"
        "                request.num_computed_tokens = num_computed_tokens\n"
        f"{WAITING_RECORD}\n"
        "                # Only track requests that will still be prefilling after this chunk.\n",
        "waiting record",
    )
    text = _replace_once(
        text,
        "    def _mamba_block_aligned_split(\n",
        f"{HELPER_METHODS}\n"
        "    def _mamba_block_aligned_split(\n",
        "helper methods",
    )

    scheduler_path.write_text(text, encoding="utf-8")
    print(f"[patch] applied concurrent partial prefill to {scheduler_path}", flush=True)

# r13-schedfix-v1 = r11-rustwarm-v1 (ENGPACE + Rust frontend + fixed
# warmup/pack/gc-freeze) + a source patch baking a smart default for
# long_prefill_token_threshold directly into SchedulerConfig, so no CLI flag
# is needed and nothing else changes.
#
# Root cause (source-read of vllm/v1/core/sched/scheduler.py, verified against
# THIS image): with long_prefill_token_threshold left at 0 (never set on any
# r11 compose), a single big-prompt request's chunk is capped only by the
# step's total max_num_batched_tokens -- the first large request in a wave
# can consume the entire step budget, starving its wave-mates that step.
# This was historically the strongest, most reproducibly non-negative lever
# in this project (a STATIC threshold, not the adaptive DNG_INTERLEAVE that
# was reverted after a real production regression).
#
# See patch_sched_threshold_default.py for the exact mechanism: self-derives
# long_prefill_token_threshold = max_num_batched_tokens // 2 whenever left
# unset, scaling automatically with whatever budget is configured.
FROM dngback/vllm-v0.24.0-ubuntu2404-opt-2:r11-rustwarm-v1

COPY patch_sched_threshold_default.py /app/patch_sched_threshold_default.py
RUN python3 /app/patch_sched_threshold_default.py \
        /usr/local/lib/python3.12/dist-packages \
    && python3 -c "import vllm.config.scheduler"

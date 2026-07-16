# r12-prefillbias-v2 = r11-rustwarm-v2 (ENGPACE + Rust frontend + fixed
# warmup/pack/gc-freeze + AIR_LMHEAD_FP8 hook, see
# contest-phase1-lmhead-fp8.md) + minhlam284's prefill-bias scheduler patch
# (SLO-aware urgent-prefill promotion + TBT-guard, ported from
# minhlam284/tseb:test, hash-verified compatible with this image's stock
# scheduler.py/scheduler_config.py/arg_utils.py/kv_cache_manager.py).
#
# NOTE: prefill_bias_slot_swap_enabled is incompatible with --async-scheduling
# in this patch's current implementation (source-verified error) -- this
# project uses --async-scheduling everywhere, so slot-swap stays OFF. Only
# urgent-prefill promotion + cache-aware ordering + TBT-guard are exercised.
FROM dngback/vllm-v0.24.0-ubuntu2404-opt-2:r11-rustwarm-v2

COPY vllm_prefill_bias /app/vllm_prefill_bias
RUN python3 /app/vllm_prefill_bias/patch_vllm_prefill_bias.py \
        /usr/local/lib/python3.12/dist-packages \
    && python3 -c "import vllm.v1.core.sched.prefill_bias" \
    && python3 -c "import vllm.v1.core.sched.scheduler"

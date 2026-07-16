# r15-prefillbias-v1: the full r12-prefillbias-v2 optimization stack, ported
# from the vLLM 0.24.0 base onto the stock vLLM 0.25.0 base.
#
#   dngback/vllm-v0.25.0-ubuntu2404-opt:r15-prefillbias-v1
#     = vllm/vllm-openai:v0.25.0-ubuntu2404 (stock)
#     + ENGPACE admission pacing            (v1/engine/core.py)
#     + P36-RUST-DISPATCH                   (entrypoints/openai/api_server.py)
#     + P26-GDN-SCANFUSE                    (mamba/gdn/qwen_gdn_linear_attn.py)
#     + prefill-bias scheduler              (v1/core/sched/{scheduler,prefill_bias}.py,
#                                            config/scheduler.py, engine/arg_utils.py,
#                                            v1/core/kv_cache_manager.py)
#     + AIR_* stages via sitecustomize.py   (warmup / prime-pack / gc-freeze / lm_head fp8)
#
# HOW THE PORT WAS DONE (2026-07-14): rather than replay the 0.24 lineage's
# stack of anchor-based patch scripts against drifted 0.25 source, every
# patched file was 3-way merged (base = stock 0.24.0 dist-packages, ours =
# r12-prefillbias-v2 dist-packages, theirs = stock 0.25.0 dist-packages). Six
# of eight files merged clean; two conflicts in v1/core/sched/scheduler.py were
# resolved by KEEPING upstream 0.25.0 semantics and re-grafting our logic:
#   1. max_num_running_reqs check -- 0.25 now counts paused streaming sessions
#      (num_waiting_for_streaming_input) toward the running-slot cap. Took the
#      new condition, kept our _prefill_blocked_max_num_seqs accounting.
#   2. defer_prefills condition -- 0.25 changed it from `num_computed_tokens
#      == 0` to `num_computed_tokens < request.num_tokens - 1`. Took the new
#      condition, kept our batch-budget branch ahead of it.
#
# TWO DELIBERATE DROPS vs the 0.24 image:
#   - TOKCACHE (renderers/base.py): upstream 0.25 deleted BaseRenderer._encode
#     outright (tokenize path refactored around offset mappings), so the anchor
#     no longer exists. It was already inert in this project's shipped config --
#     VLLM_USE_RUST_FRONTEND=1 means vllm-rs owns tokenize + chat-template
#     render (fastokens), so the Python renderer never runs. Dropped rather
#     than re-implemented against the new API: zero benefit here, real risk.
#     AIR_RENDER_CACHE (the sitecustomize-level render memo, one layer above
#     TOKCACHE) is untouched and still hooks HfRenderer.render_messages[_async],
#     which DOES still exist in 0.25.
#   - champ's vllm-rs binary: the stock 0.25.0 base ships its OWN version-matched
#     vllm-rs (38MB, dist-packages/vllm/vllm-rs). We use that instead of carrying
#     the 0.24-era binary forward -- the Rust frontend talks to the Python engine
#     over a version-coupled interface, so the matched binary is the safe choice.
#
# Everything else ported unchanged: v1/engine/core.py and qwen_gdn_linear_attn.py
# were byte-identical between stock 0.24.0 and 0.25.0, so ENGPACE and GDN-SCANFUSE
# are exact drop-ins; api_server.py's __main__ tail was also unchanged, so
# P36-RUST-DISPATCH applies verbatim; logits_processor.py (AIR_LMHEAD_FP8's
# monkeypatch target) is likewise identical, so that hook needs no changes.

# Warmup assets are lifted byte-for-byte from the 0.24 baseline image rather than
# regenerated, so cache/pack priming behaviour is identical across the A/B.
FROM dngback/vllm-v0.24.0-ubuntu2404-opt-2:r12-prefillbias-v2 AS legacy

FROM vllm/vllm-openai:v0.25.0-ubuntu2404

COPY --from=legacy /app/warmup_prime_pack.jsonl \
                   /app/warmup_prime_pack_full.jsonl \
                   /app/warmup_system_prompt.txt \
                   /app/

# AIR_* deferred-import hooks. Copied to every path Python may auto-import
# sitecustomize from, so it arms in EngineCore/dispatch/spawned children alike.
COPY sitecustomize.py /etc/python3.12/sitecustomize.py
COPY sitecustomize.py /usr/lib/python3.12/sitecustomize.py
COPY sitecustomize.py /usr/local/lib/python3.12/dist-packages/sitecustomize.py
COPY sitecustomize.py /app/sitecustomize.py

# Ported source patches, overlaid onto the stock 0.25.0 package tree.
COPY vllm_patched/ /usr/local/lib/python3.12/dist-packages/vllm/

# Fail the build loudly if any patch failed to land, rather than shipping an
# image that silently lost a lever (the exact failure mode that made
# AIR_WARMUP/PRIME_PACK inert for weeks under the Rust frontend).
RUN set -eu; \
    DP=/usr/local/lib/python3.12/dist-packages; \
    grep -q "v5-latch"           "$DP/vllm/v1/engine/core.py"; \
    grep -q "ENGPACE"            "$DP/vllm/v1/engine/core.py"; \
    grep -q "P36-RUST-DISPATCH"  "$DP/vllm/entrypoints/openai/api_server.py"; \
    grep -q "P26-GDN-SCANFUSE"   "$DP/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"; \
    grep -q "P26_DIRECT_SCAN"    "$DP/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"; \
    grep -q "VLLM_PREFILL_BIAS_PATCH" "$DP/vllm/v1/core/sched/scheduler.py"; \
    grep -q "VLLM_PREFILL_BIAS_PATCH" "$DP/vllm/config/scheduler.py"; \
    grep -q "VLLM_PREFILL_BIAS_PATCH" "$DP/vllm/engine/arg_utils.py"; \
    grep -q "peek_num_computed_tokens" "$DP/vllm/v1/core/kv_cache_manager.py"; \
    grep -q "_patch_lmhead_fp8"   "$DP/sitecustomize.py"; \
    grep -q "_patch_cli_serve"    "$DP/sitecustomize.py"; \
    test -x "$DP/vllm/vllm-rs"; \
    test -s /app/warmup_prime_pack_full.jsonl; \
    python3 -c "import sitecustomize"; \
    python3 -c "import vllm; assert vllm.__version__.startswith('0.25'), vllm.__version__; print('vllm', vllm.__version__)"; \
    python3 -c "import vllm.v1.core.sched.prefill_bias, vllm.v1.core.sched.scheduler"; \
    python3 -c "import vllm.v1.engine.core, vllm.config.scheduler, vllm.engine.arg_utils, vllm.v1.core.kv_cache_manager"; \
    python3 -c "import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"; \
    python3 -c "from vllm.entrypoints.cli.serve import run_multi_api_server"; \
    echo "R15-V0250-PATCHES-OK"

ENV PYTHONPATH=/app
WORKDIR /app

# The grading harness locks this entrypoint -- P36-RUST-DISPATCH exists precisely
# so the Rust frontend still dispatches under it. Don't change to `vllm serve`.
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

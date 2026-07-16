# r14-flashqla-fix-v1 = r11-rustwarm-v1 + a one-line fix to a stock vLLM
# 0.24.0 bug that crashes the EngineCore on every request when
# --gdn-prefill-backend=flashqla is selected (missing g_already_exp param on
# ChunkGatedDeltaRule.forward_flashqla). See patch_flashqla_sig.py.
FROM dngback/vllm-v0.24.0-ubuntu2404-opt-2:r11-rustwarm-v1

COPY patch_flashqla_sig.py /app/patch_flashqla_sig.py
RUN python3 /app/patch_flashqla_sig.py /usr/local/lib/python3.12/dist-packages \
    && python3 -c "import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"

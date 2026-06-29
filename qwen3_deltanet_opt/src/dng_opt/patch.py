"""
Monkey-patch entrypoint — swaps vLLM's QwenGatedDeltaNetAttention for
one of our subclasses without touching any vLLM source file.

Usage
-----
Baseline with per-stage timing:

    DNGOPT_MODE=instrument python -c "
        from dng_opt.patch import apply_patch
        apply_patch()
        # ... then start vLLM engine as normal
    "

Fused Triton kernel:

    DNGOPT_MODE=fused python -c "
        from dng_opt.patch import apply_patch
        apply_patch()
        # ... then start vLLM engine as normal
    "

Or in a startup script:

    import os
    os.environ['DNGOPT_MODE'] = 'fused'   # or 'instrument'
    from dng_opt.patch import apply_patch
    apply_patch()

The patch must be applied BEFORE the vLLM model is instantiated (i.e.,
before LLMEngine / AsyncLLMEngine loads the model weights).  Calling
apply_patch() after the model is loaded has no effect.

How it works
------------
``Qwen3_5DecoderLayer.__init__`` (in vllm/.../qwen3_5.py) instantiates
``QwenGatedDeltaNetAttention`` by name from its own module globals.
Replacing that name before model instantiation is sufficient to make every
GDN layer use our subclass.  The PluggableLayer string-registry is also
updated so that torch.compile's no_compile_layers dict still resolves the
prefix → layer object correctly.
"""

from __future__ import annotations

import os
import importlib
import logging

logger = logging.getLogger(__name__)

_ORIGINAL_CLASS = None
_PATCHED = False


def apply_patch(mode: str | None = None) -> None:
    """
    Replace QwenGatedDeltaNetAttention with the requested variant.

    Parameters
    ----------
    mode : {"fused", "instrument"} or None
        If None, reads the ``DNGOPT_MODE`` environment variable.
        Defaults to ``"fused"`` if the variable is not set.
    """
    global _ORIGINAL_CLASS, _PATCHED

    if _PATCHED:
        logger.warning("dng_opt patch already applied; ignoring duplicate call.")
        return

    if mode is None:
        mode = os.environ.get("DNGOPT_MODE", "fused").lower()

    if mode == "fused":
        from dng_opt.models.qwen35_fused import FusedQwenGDNAttention as NewClass
    elif mode == "instrument":
        from dng_opt.models.qwen35_fused import InstrumentedQwenGDNAttention as NewClass
    else:
        raise ValueError(f"Unknown DNGOPT_MODE={mode!r}.  Use 'fused' or 'instrument'.")

    # ------------------------------------------------------------------
    # 1. Patch the name inside qwen3_5.py (where Qwen3_5DecoderLayer
    #    looks it up at model-load time).
    # ------------------------------------------------------------------
    qwen35_mod = importlib.import_module(
        "vllm.model_executor.models.qwen3_5"
    )
    _ORIGINAL_CLASS = getattr(qwen35_mod, "QwenGatedDeltaNetAttention")
    setattr(qwen35_mod, "QwenGatedDeltaNetAttention", NewClass)

    # ------------------------------------------------------------------
    # 2. Patch the source module as well, so any code that imports
    #    directly from qwen_gdn_linear_attn also gets our class.
    # ------------------------------------------------------------------
    gdn_mod = importlib.import_module(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    setattr(gdn_mod, "QwenGatedDeltaNetAttention", NewClass)

    # ------------------------------------------------------------------
    # 3. Update the PluggableLayer registry (used by torch.compile to
    #    resolve layer names in the static forward context).
    # ------------------------------------------------------------------
    try:
        from vllm.model_executor.custom_op import PluggableLayer
        registry = getattr(PluggableLayer, "_registry", None)
        if isinstance(registry, dict):
            registry["qwen_gated_delta_net_attention"] = NewClass
            logger.debug("dng_opt: PluggableLayer registry updated.")
    except Exception as exc:  # noqa: BLE001
        logger.debug("dng_opt: Could not update PluggableLayer registry: %s", exc)

    _PATCHED = True
    logger.info("dng_opt: patch applied (mode=%s, class=%s).", mode, NewClass.__name__)


def remove_patch() -> None:
    """Restore the original QwenGatedDeltaNetAttention (useful in tests)."""
    global _ORIGINAL_CLASS, _PATCHED

    if not _PATCHED or _ORIGINAL_CLASS is None:
        return

    qwen35_mod = importlib.import_module("vllm.model_executor.models.qwen3_5")
    gdn_mod = importlib.import_module(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    setattr(qwen35_mod, "QwenGatedDeltaNetAttention", _ORIGINAL_CLASS)
    setattr(gdn_mod, "QwenGatedDeltaNetAttention", _ORIGINAL_CLASS)

    try:
        from vllm.model_executor.custom_op import PluggableLayer
        registry = getattr(PluggableLayer, "_registry", None)
        if isinstance(registry, dict):
            registry["qwen_gated_delta_net_attention"] = _ORIGINAL_CLASS
    except Exception:  # noqa: BLE001
        pass

    _PATCHED = False
    _ORIGINAL_CLASS = None
    logger.info("dng_opt: patch removed, original class restored.")

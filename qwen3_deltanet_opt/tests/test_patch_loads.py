"""
Smoke tests for the monkey-patch mechanism.

These tests verify that:
  - apply_patch() replaces the class in the right module namespaces
  - remove_patch() fully restores the original
  - Both modes ("fused", "instrument") are accepted
  - Patching twice is idempotent (just warns)

No GPU or model weights required.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_patch():
    """Ensure patch is removed after each test."""
    yield
    try:
        from dng_opt.patch import remove_patch
        remove_patch()
    except Exception:
        pass


def _original_class():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    return QwenGatedDeltaNetAttention


def _qwen35_class():
    import importlib
    mod = importlib.import_module("vllm.model_executor.models.qwen3_5")
    return getattr(mod, "QwenGatedDeltaNetAttention")


def _gdn_mod_class():
    import importlib
    mod = importlib.import_module(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    return getattr(mod, "QwenGatedDeltaNetAttention")


class TestPatchApply:
    def test_fused_mode_replaces_class_in_qwen35(self):
        from dng_opt.patch import apply_patch
        from dng_opt.models.qwen35_fused import FusedQwenGDNAttention

        apply_patch("fused")
        assert _qwen35_class() is FusedQwenGDNAttention

    def test_fused_mode_replaces_class_in_gdn_mod(self):
        from dng_opt.patch import apply_patch
        from dng_opt.models.qwen35_fused import FusedQwenGDNAttention

        apply_patch("fused")
        assert _gdn_mod_class() is FusedQwenGDNAttention

    def test_instrument_mode_replaces_class(self):
        from dng_opt.patch import apply_patch
        from dng_opt.models.qwen35_fused import InstrumentedQwenGDNAttention

        apply_patch("instrument")
        assert _qwen35_class() is InstrumentedQwenGDNAttention

    def test_unknown_mode_raises(self):
        from dng_opt.patch import apply_patch

        with pytest.raises(ValueError, match="Unknown DNGOPT_MODE"):
            apply_patch("unknown_mode")


class TestPatchRemove:
    def test_remove_restores_original_in_qwen35(self):
        from dng_opt.patch import apply_patch, remove_patch

        original = _original_class()
        apply_patch("fused")
        remove_patch()
        assert _qwen35_class() is original

    def test_remove_restores_original_in_gdn_mod(self):
        from dng_opt.patch import apply_patch, remove_patch

        original = _original_class()
        apply_patch("fused")
        remove_patch()
        assert _gdn_mod_class() is original

    def test_remove_without_apply_is_noop(self):
        from dng_opt.patch import remove_patch
        remove_patch()  # should not raise


class TestPatchIdempotent:
    def test_double_apply_does_not_crash(self):
        from dng_opt.patch import apply_patch

        apply_patch("fused")
        apply_patch("fused")  # second call should just warn, not crash


class TestSubclassInheritance:
    """The patched classes must be proper subclasses of the original."""

    def test_fused_is_subclass(self):
        from dng_opt.models.qwen35_fused import FusedQwenGDNAttention

        base = _original_class()
        assert issubclass(FusedQwenGDNAttention, base)

    def test_instrumented_is_subclass(self):
        from dng_opt.models.qwen35_fused import InstrumentedQwenGDNAttention

        base = _original_class()
        assert issubclass(InstrumentedQwenGDNAttention, base)

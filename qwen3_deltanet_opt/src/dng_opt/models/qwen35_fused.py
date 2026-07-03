"""
Subclasses of ``QwenGatedDeltaNetAttention`` that inject the fused kernel.

Two variants
------------
InstrumentedQwenGDNAttention
    Drop-in replacement for the stock class that wraps each stage of
    ``_forward_core_decode_non_spec`` in CUDA events so we can measure how
    much wall-clock time each stage spends.  Use this with the **baseline**
    server to identify the bottleneck before optimising.

FusedQwenGDNAttention
    Replaces ``fused_recurrent_gated_delta_rule_packed_decode`` (the FLA
    packed-decode kernel) with our single fused Triton kernel from
    ``dng_opt.kernels.fused_gdn_decode``.  The causal conv1d step is kept
    unchanged because it touches a separate piece of state with its own
    layout constraints.

Neither class edits any vLLM source file — they are pure subclasses.
``patch.py`` monkey-patches the vLLM module namespace to make the model
loader pick them up.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import torch

try:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as _GDNBase,
    )
except ModuleNotFoundError:
    # Upstream vLLM ≥0.20 uses the flat layout without a Qwen-specific subclass.
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (  # type: ignore[no-redef]
        GatedDeltaNetAttention as _GDNBase,
    )
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

if TYPE_CHECKING:
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


# ---------------------------------------------------------------------------
# Optional proof-of-execution counter.
#
# The vLLM engine runs in a spawned worker process, so an in-memory counter is
# invisible from outside.  When DNGOPT_COUNTER_FILE is set, the fused override
# below records how many times it actually ran by writing the count to that
# file (and prints a one-time marker to the worker's stdout).  This makes it
# possible to *prove* the fused decode path is on the live hot path rather than
# being silently bypassed by a compiled custom op.
# ---------------------------------------------------------------------------

_FUSED_CALL_COUNT = 0
_FUSED_COUNTER_FILE = os.environ.get("DNGOPT_COUNTER_FILE")


def _record_fused_call() -> None:
    global _FUSED_CALL_COUNT
    _FUSED_CALL_COUNT += 1
    if _FUSED_CALL_COUNT == 1:
        print(
            f"[dng_opt] FUSED decode path HIT (pid={os.getpid()}) — "
            f"FusedQwenGDNAttention._forward_core_decode_non_spec is live.",
            flush=True,
        )
    if _FUSED_COUNTER_FILE:
        try:
            with open(_FUSED_COUNTER_FILE, "w") as f:
                f.write(str(_FUSED_CALL_COUNT))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Instrumented variant  (adds per-stage CUDA event timing)
# ---------------------------------------------------------------------------

class InstrumentedQwenGDNAttention(_GDNBase):
    """
    Wraps each stage of the decode path in CUDA events.

    After inference, ``self.stage_times`` is a ``dict[str, list[float]]``
    mapping stage name → list of elapsed milliseconds (one entry per decode
    step).  Reset with ``reset_stage_times()``.

    Stages measured
    ---------------
    conv1d        causal_conv1d_update call
    recurrent     fused_recurrent_gated_delta_rule_packed_decode call
    total         full _forward_core_decode_non_spec call
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_times: dict[str, list[float]] = defaultdict(list)
        self._cuda_events: list[torch.cuda.Event] = []

    def reset_stage_times(self) -> None:
        self.stage_times = defaultdict(list)

    def _event(self) -> torch.cuda.Event:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._cuda_events.append(e)
        return e

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: "GDNAttentionMetadata",
    ) -> None:
        from vllm.model_executor.layers.fla.ops import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )

        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        self_kv_cache = self.kv_cache
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        t0_total = time.perf_counter()

        # ---- stage: conv1d ----
        e_conv_start = self._event()
        mixed_qkv_conv = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            validate_data=False,
        )
        e_conv_end = self._event()
        torch.cuda.synchronize()
        self.stage_times["conv1d"].append(e_conv_start.elapsed_time(e_conv_end))

        # ---- stage: recurrent ----
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        e_rec_start = self._event()
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_conv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim ** -0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            use_qk_l2norm_in_kernel=True,
        )
        e_rec_end = self._event()
        torch.cuda.synchronize()
        self.stage_times["recurrent"].append(e_rec_start.elapsed_time(e_rec_end))

        t1_total = time.perf_counter()
        self.stage_times["total_ms"].append((t1_total - t0_total) * 1000)


# ---------------------------------------------------------------------------
# Fused variant  (replaces recurrent step with single Triton kernel)
# ---------------------------------------------------------------------------

class FusedQwenGDNAttention(_GDNBase):
    """
    Replaces the FLA packed-decode kernel with the fused Triton kernel from
    ``dng_opt.kernels.fused_gdn_decode``.

    The fused kernel combines:
        gate computation  (g, beta)
        L2-normalisation  (q, k)
        recurrent delta-rule update
        shortcut output   (avoids re-reading the updated state)

    into a single Triton program per (sequence, v-head, v-tile), eliminating
    intermediate global-memory writes between these steps.
    """

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: "GDNAttentionMetadata",
    ) -> None:
        from dng_opt.kernels.fused_gdn_decode import fused_gdn_decode

        _record_fused_call()

        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        self_kv_cache = self.kv_cache
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        # ---- step 1: causal conv1d (unchanged) ----
        mixed_qkv_conv = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            validate_data=False,
        )

        # ---- step 2: fused gate + norm + recurrent + output ----
        nk = self.num_k_heads // self.tp_size
        nv = self.num_v_heads // self.tp_size
        DK = self.head_k_dim
        DV = self.head_v_dim

        out = fused_gdn_decode(
            mixed_qkv=mixed_qkv_conv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            ssm_state=ssm_state,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            nk=nk,
            nv=nv,
            DK=DK,
            DV=DV,
            scale=self.head_k_dim ** -0.5,
        )

        # Write into pre-allocated output buffer  [T, nv, DV]
        core_attn_out[:num_actual_tokens] = out

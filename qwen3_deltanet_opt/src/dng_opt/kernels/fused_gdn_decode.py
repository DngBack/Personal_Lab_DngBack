"""
Fused Gated DeltaNet decode kernel for Qwen3.5.

What this fuses
---------------
In the stock vLLM decode path (``_forward_core_decode_non_spec``), after the
causal conv1d update the pipeline executes these steps as separate kernels:

    1. Split mixed_qkv → q, k, v
    2. L2-normalise q and k, then scale q by ``head_k_dim ** -0.5``
    3. Gate computation  g = -exp(A_log) * softplus(a + dt_bias)
                         beta = sigmoid(b)
    4. Recurrent gated-delta-rule state update
    5. Output projection

This kernel fuses steps 2–5 into one Triton program per (sequence, v-head,
v-tile) triple.

Exact recurrence (matches vLLM's ``fused_recurrent_gated_delta_rule_packed_decode``)
------------------------------------------------------------------------------------
State ``S`` has shape ``[DV, DK]`` (value-major; this is vLLM's temporal-state
layout ``(num_v_heads, head_v_dim, head_k_dim)``).  For one decode step:

    S'      = exp(g) * S                        # decay FIRST
    r       = beta * (v - S' @ k)               # residual read by k, off decayed state
    S_new   = S' + outer(r, k)                  # rank-1 update with k
    out     = S_new @ q                         # output read by q

The output is computed from ``S_new`` while it is still in registers, avoiding
a global-memory re-read while keeping the same arithmetic order as vLLM's FLA
packed decode kernel.

Layout assumptions (match vLLM Qwen3.5 defaults)
-------------------------------------------------
* mixed_qkv  : [T, q_dim + k_dim + v_dim]
                  q_dim = nk * DK,  k_dim = nk * DK,  v_dim = nv * DV
* a, b       : [T, nv]
* A_log, dt_bias : [nv]
* ssm_state  : [max_B, nv, DV, DK]   (value-major: S[v, k])
* out        : [T, nv, DV]
* q is L2-normalised then multiplied by ``scale`` (= DK ** -0.5); k is only
  L2-normalised.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _fused_gdn_decode_kernel(
        # -- packed QKV (after conv1d, before split) --
        mixed_qkv_ptr,
        stride_qkv_t,   # stride along T  (= q_dim + k_dim + v_dim)
        q_dim,          # nk * DK
        k_dim,          # nk * DK  (same as q_dim for symmetric GQA)
        # -- gate inputs --
        a_ptr,
        b_ptr,
        stride_gate_t,  # stride along T for a / b
        # -- per-head fixed parameters --
        A_log_ptr,
        dt_bias_ptr,
        # -- SSM recurrent state  [max_B, NV, DV, DK]  (value-major) --
        state_ptr,
        stride_state_b,  # stride along max-batch slot dim
        stride_state_h,  # stride along head dim
        stride_state_v,  # stride along DV  (value axis)
        stride_state_k,  # stride along DK  (key axis, innermost = 1 for row-major)
        state_idx_ptr,   # [T]  batch-slot index for each sequence
        # -- output tensor --
        out_ptr,
        stride_out_t,
        stride_out_h,
        stride_out_v,
        # -- scalar applied to q after L2-norm (= head_k_dim ** -0.5) --
        scale,
        # -- compile-time constants --
        NK: tl.constexpr,    # num k-heads per TP rank
        NV: tl.constexpr,    # num v-heads per TP rank
        DK: tl.constexpr,    # head_k_dim
        DV: tl.constexpr,    # head_v_dim
        RATIO: tl.constexpr, # NV // NK  (GQA expansion factor)
        BLOCK_V: tl.constexpr,  # tile width along DV axis
    ):
        """
        Grid: (T * NV,  ceil(DV / BLOCK_V))
        Each program handles one (sequence t, head h, v-tile pid_v).
        """
        pid_th = tl.program_id(0)
        pid_v  = tl.program_id(1)

        t  = pid_th // NV
        h  = pid_th % NV
        hk = h // RATIO  # which k-head drives this v-head

        v_start = pid_v * BLOCK_V

        # index vectors (compile-time shapes)
        i = tl.arange(0, DK)                 # [DK]      — key dimension
        j = v_start + tl.arange(0, BLOCK_V)  # [BLOCK_V] — value dimension
        v_mask = j < DV

        # ----------------------------------------------------------------
        # Gate scalars
        # ----------------------------------------------------------------
        A_log_h  = tl.load(A_log_ptr  + h).to(tl.float32)
        dt_bias_h = tl.load(dt_bias_ptr + h).to(tl.float32)
        a_th = tl.load(a_ptr + t * stride_gate_t + h).to(tl.float32)
        b_th = tl.load(b_ptr + t * stride_gate_t + h).to(tl.float32)

        # softplus(x) — numerically stable
        x = a_th + dt_bias_h
        softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)

        # g = -exp(A_log) * softplus(...) → gate_decay = exp(g)
        gate  = tl.exp(-tl.exp(A_log_h) * softplus_x)
        beta = tl.sigmoid(b_th)  # stay in float32 — no bf16 round-trip

        # ----------------------------------------------------------------
        # Load q, k  (full DK vector for the matching k-head)
        # ----------------------------------------------------------------
        qkv_base = t * stride_qkv_t
        q_vec = tl.load(mixed_qkv_ptr + qkv_base          + hk * DK + i).to(tl.float32)
        k_vec = tl.load(mixed_qkv_ptr + qkv_base + q_dim  + hk * DK + i).to(tl.float32)

        # L2 normalise; then scale q (matches vLLM: b_q = l2norm(b_q) * scale,
        # b_k = l2norm(b_k)).  Only q is scaled.
        q_vec = q_vec / tl.sqrt(tl.sum(q_vec * q_vec) + 1e-6)
        k_vec = k_vec / tl.sqrt(tl.sum(k_vec * k_vec) + 1e-6)
        q_vec = q_vec * scale

        # ----------------------------------------------------------------
        # Load v tile  [BLOCK_V]
        # ----------------------------------------------------------------
        v_tile = tl.load(
            mixed_qkv_ptr + qkv_base + q_dim + k_dim + h * DV + j,
            mask=v_mask, other=0.0,
        ).to(tl.float32)

        # ----------------------------------------------------------------
        # Load state tile  S[v_start:v_start+BLOCK_V, 0:DK]
        # State layout: [max_B, NV, DV, DK]   →   S[v, k]
        # ----------------------------------------------------------------
        sidx = tl.load(state_idx_ptr + t)
        state_base = sidx * stride_state_b + h * stride_state_h

        # Match vLLM's packed decode behavior for NULL_BLOCK_ID=0.
        if sidx <= 0:
            zero = tl.zeros([BLOCK_V], dtype=tl.float32).to(out_ptr.dtype.element_ty)
            tl.store(
                out_ptr + t * stride_out_t + h * stride_out_h + j,
                zero,
                mask=v_mask,
            )
            return

        # pointer array [BLOCK_V, DK]
        S_tile = tl.load(
            state_ptr + state_base
                + j[:, None] * stride_state_v
                + i[None, :] * stride_state_k,
            mask=v_mask[:, None],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_V, DK]

        # ----------------------------------------------------------------
        # Decay state FIRST:  S' = exp(g) * S
        # ----------------------------------------------------------------
        S_dec = gate * S_tile  # [BLOCK_V, DK]

        # o_pre_k[v] = Σ_k S'[v,k] * k[k]      (read decayed state with k)
        o_pre_k = tl.sum(S_dec * k_vec[None, :], axis=1)  # [BLOCK_V]
        # residual  r = beta * (v - S'·k)
        r_tile = beta * (v_tile - o_pre_k)  # [BLOCK_V]

        # ----------------------------------------------------------------
        # State update  S_new = S' + outer(r, k)
        # S_new[v, k] = S'[v, k] + r[v] * k[k]
        # ----------------------------------------------------------------
        S_new = S_dec + r_tile[:, None] * k_vec[None, :]  # [BLOCK_V, DK]
        out_tile = tl.sum(S_new * q_vec[None, :], axis=1)  # [BLOCK_V]

        # ----------------------------------------------------------------
        # Write back state tile and output
        # ----------------------------------------------------------------
        tl.store(
            state_ptr + state_base
                + j[:, None] * stride_state_v
                + i[None, :] * stride_state_k,
            S_new.to(state_ptr.dtype.element_ty),
            mask=v_mask[:, None],
        )

        tl.store(
            out_ptr + t * stride_out_t + h * stride_out_h + j,
            out_tile.to(out_ptr.dtype.element_ty),
            mask=v_mask,
        )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def fused_gdn_decode(
    mixed_qkv: torch.Tensor,         # [T, q_dim+k_dim+v_dim]  float16/bf16
    a: torch.Tensor,                  # [T, nv]
    b: torch.Tensor,                  # [T, nv]
    A_log: torch.Tensor,              # [nv]
    dt_bias: torch.Tensor,            # [nv]
    ssm_state: torch.Tensor,          # [max_B, nv, DV, DK]  mutated in-place
    ssm_state_indices: torch.Tensor,  # [T]  int32/int64
    nk: int,
    nv: int,
    DK: int,
    DV: int,
    scale: float | None = None,       # query scale; defaults to DK ** -0.5
) -> torch.Tensor:
    """
    Fused gate + L2-norm + q-scale + gated-delta-rule update for one decode step.

    Replaces the gate-computation → normalise → recurrent-update → output
    sequence run by ``fused_recurrent_gated_delta_rule_packed_decode`` from FLA.

    Returns
    -------
    out : [T, nv, DV]  same dtype as mixed_qkv.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed; cannot run fused_gdn_decode")

    T = mixed_qkv.shape[0]
    ratio = nv // nk
    q_dim = nk * DK
    k_dim = nk * DK
    v_dim = nv * DV
    if scale is None:
        scale = DK ** -0.5

    assert mixed_qkv.shape == (T, q_dim + k_dim + v_dim), (
        f"mixed_qkv shape mismatch: expected ({T}, {q_dim+k_dim+v_dim}), "
        f"got {tuple(mixed_qkv.shape)}"
    )
    assert a.shape == (T, nv) and b.shape == (T, nv)
    # state is value-major: [max_B, nv, DV, DK]
    assert ssm_state.shape[1] == nv and ssm_state.shape[2] == DV and ssm_state.shape[3] == DK

    out = torch.empty(T, nv, DV, dtype=mixed_qkv.dtype, device=mixed_qkv.device)

    BLOCK_V = min(triton.next_power_of_2(DV), 32)
    grid = (T * nv, triton.cdiv(DV, BLOCK_V))

    _fused_gdn_decode_kernel[grid](
        mixed_qkv, mixed_qkv.stride(0), q_dim, k_dim,
        a, b, a.stride(0),
        A_log, dt_bias,
        ssm_state,
        ssm_state.stride(0), ssm_state.stride(1),
        ssm_state.stride(2), ssm_state.stride(3),
        ssm_state_indices,
        out, out.stride(0), out.stride(1), out.stride(2),
        scale,
        NK=nk, NV=nv, DK=DK, DV=DV, RATIO=ratio, BLOCK_V=BLOCK_V,
    )
    return out


# ---------------------------------------------------------------------------
# Pure-PyTorch reference  (used for correctness tests)
# ---------------------------------------------------------------------------

def ref_gdn_decode(
    mixed_qkv: torch.Tensor,         # [T, q_dim+k_dim+v_dim]
    a: torch.Tensor,                  # [T, nv]
    b: torch.Tensor,                  # [T, nv]
    A_log: torch.Tensor,              # [nv]
    dt_bias: torch.Tensor,            # [nv]
    ssm_state: torch.Tensor,          # [max_B, nv, DV, DK]  mutated in-place
    ssm_state_indices: torch.Tensor,  # [T]
    nk: int,
    nv: int,
    DK: int,
    DV: int,
    scale: float | None = None,       # query scale; defaults to DK ** -0.5
) -> torch.Tensor:
    """
    Single-step reference implementation of the fused kernel.
    Mathematically equivalent but uses plain PyTorch.  State is value-major
    ``[max_B, nv, DV, DK]`` — i.e. ``S[v, k]`` — matching vLLM.
    """
    T = mixed_qkv.shape[0]
    ratio = nv // nk
    q_dim = nk * DK
    k_dim = nk * DK
    if scale is None:
        scale = DK ** -0.5

    orig_dtype = mixed_qkv.dtype
    mv = mixed_qkv.float()

    # split + reshape
    q = mv[:, :q_dim].view(T, nk, DK)
    k = mv[:, q_dim : q_dim + k_dim].view(T, nk, DK)
    v = mv[:, q_dim + k_dim :].view(T, nv, DV)

    # L2 normalise; scale q only (matches vLLM: l2norm then *scale)
    q = F.normalize(q, dim=-1) * scale
    k = F.normalize(k, dim=-1)

    # expand k-heads → v-heads (GQA)
    q = q.repeat_interleave(ratio, dim=1)  # [T, nv, DK]
    k = k.repeat_interleave(ratio, dim=1)  # [T, nv, DK]

    # gate computation
    x = a.float() + dt_bias.float().unsqueeze(0)  # [T, nv]
    softplus_x = F.softplus(x)
    gate = (-A_log.float().exp().unsqueeze(0) * softplus_x).exp()  # [T, nv]
    beta = b.float().sigmoid()                                       # [T, nv]

    out = torch.empty(T, nv, DV, device=mixed_qkv.device, dtype=torch.float32)

    for t in range(T):
        sidx = ssm_state_indices[t].item()
        if sidx <= 0:
            out[t].zero_()
            continue
        S = ssm_state[sidx].float()  # [nv, DV, DK]

        for h in range(nv):
            S_h    = S[h]          # [DV, DK]
            q_h    = q[t, h]       # [DK]
            k_h    = k[t, h]       # [DK]
            v_h    = v[t, h]       # [DV]
            gate_h = gate[t, h]
            beta_h = beta[t, h]

            S_dec   = gate_h * S_h          # decay first   [DV, DK]
            o_pre_k = S_dec @ k_h           # [DV]
            r       = beta_h * (v_h - o_pre_k)  # [DV]

            # state update  S_new = S' + outer(r, k)
            S_new = S_dec + torch.outer(r, k_h)
            out[t, h] = S_new @ q_h
            ssm_state[sidx, h] = S_new.to(ssm_state.dtype)

    return out.to(orig_dtype)

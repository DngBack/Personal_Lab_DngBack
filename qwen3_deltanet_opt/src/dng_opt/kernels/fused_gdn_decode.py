"""
Fused Gated DeltaNet decode kernel for Qwen3.5.

What this fuses
---------------
In the stock vLLM decode path (``_forward_core_decode_non_spec``), after the
causal conv1d update the pipeline executes these steps as separate kernels:

    1. Split mixed_qkv → q, k, v
    2. L2-normalise q and k
    3. Gate computation  g = -exp(A_log) * softplus(a + dt_bias)
                         beta = sigmoid(b)
    4. Recurrent delta-rule state update
           S_new = exp(g) * S + beta * outer(k, v - q@S)
    5. Output projection   out = q @ S_new

Each step writes intermediate tensors to global memory.  At batch=1 (single-
token decode) the arithmetic intensity is low, so kernel-launch overhead and
memory-round-trip cost dominate.

This kernel fuses steps 2–5 into one Triton program per (sequence, v-head,
v-tile) triple using the algebraic shortcut:

    out = q @ S_new
        = q @ (exp(g)*S + beta*outer(k, r))      where r = v - q@S
        = exp(g) * (q@S) + beta * (q·k) * r
        = exp(g) * o_pre  +  beta * qk * r        (*)

The shortcut (*) avoids writing S_new to global memory before computing
the output — one less full-state round-trip per head per decode step.

Layout assumptions (match vLLM Qwen3.5 defaults)
-------------------------------------------------
* mixed_qkv  : [T, q_dim + k_dim + v_dim]
                  q_dim = nk * DK,  k_dim = nk * DK,  v_dim = nv * DV
* a, b       : [T, nv]
* A_log, dt_bias : [nv]
* ssm_state  : [max_B, nv, DK, DV]   (state is k-major inside each head)
* out        : [T, nv, DV]
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
        # -- SSM recurrent state --
        state_ptr,
        stride_state_b,  # stride along max-batch slot dim
        stride_state_h,  # stride along head dim
        stride_state_k,  # stride along DK
        stride_state_v,  # stride along DV  (innermost, =1 for row-major)
        state_idx_ptr,   # [T]  batch-slot index for each sequence
        # -- output tensor --
        out_ptr,
        stride_out_t,
        stride_out_h,
        stride_out_v,
        # -- compile-time constants --
        NK: tl.constexpr,    # num k-heads per TP rank
        NV: tl.constexpr,    # num v-heads per TP rank
        DK: tl.constexpr,    # head_k_dim
        DV: tl.constexpr,    # head_v_dim  (must equal BLOCK_V for a single-tile launch)
        RATIO: tl.constexpr, # NV // NK  (GQA expansion factor)
        BLOCK_V: tl.constexpr,  # tile width along DV axis
    ):
        """
        Grid: (T * NV,  DV // BLOCK_V)
        Each program handles one (sequence t, head h, v-tile pid_v).
        """
        pid_th = tl.program_id(0)
        pid_v  = tl.program_id(1)

        t  = pid_th // NV
        h  = pid_th % NV
        hk = h // RATIO  # which k-head drives this v-head

        v_start = pid_v * BLOCK_V

        # index vectors (compile-time shapes)
        i = tl.arange(0, DK)           # [DK]  — k dimension
        j = v_start + tl.arange(0, BLOCK_V)  # [BLOCK_V] — v dimension
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
        beta  = tl.sigmoid(b_th)

        # ----------------------------------------------------------------
        # Load q, k  (full DK vector for the matching k-head)
        # ----------------------------------------------------------------
        qkv_base = t * stride_qkv_t
        q_vec = tl.load(mixed_qkv_ptr + qkv_base          + hk * DK + i).to(tl.float32)
        k_vec = tl.load(mixed_qkv_ptr + qkv_base + q_dim  + hk * DK + i).to(tl.float32)

        # L2 normalise in place
        q_norm = tl.math.sqrt(tl.sum(q_vec * q_vec) + 1e-6)
        q_vec  = q_vec / q_norm
        k_norm = tl.math.sqrt(tl.sum(k_vec * k_vec) + 1e-6)
        k_vec  = k_vec / k_norm

        # qk scalar  (used in shortcut output formula)
        qk = tl.sum(q_vec * k_vec)

        # ----------------------------------------------------------------
        # Load v tile
        # ----------------------------------------------------------------
        v_tile = tl.load(
            mixed_qkv_ptr + qkv_base + q_dim + k_dim + h * DV + j,
            mask=v_mask, other=0.0,
        ).to(tl.float32)

        # ----------------------------------------------------------------
        # Load state tile  S[0:DK, v_start : v_start+BLOCK_V]
        # State layout: [max_B, NV, DK, DV]
        # ----------------------------------------------------------------
        sidx = tl.load(state_idx_ptr + t)
        state_base = sidx * stride_state_b + h * stride_state_h

        # pointer array [DK, BLOCK_V]
        S_tile = tl.load(
            state_ptr + state_base
                + i[:, None] * stride_state_k
                + j[None, :] * stride_state_v,
            mask=v_mask[None, :],
            other=0.0,
        ).to(tl.float32)  # [DK, BLOCK_V]

        # ----------------------------------------------------------------
        # o_pre = q @ S_tile   →   [BLOCK_V]
        # o_pre[jj] = Σ_i  q[i] * S[i, jj]
        # ----------------------------------------------------------------
        o_pre_tile = tl.sum(q_vec[:, None] * S_tile, axis=0)  # [BLOCK_V]

        # ----------------------------------------------------------------
        # Residual  r = v - o_pre
        # ----------------------------------------------------------------
        r_tile = v_tile - o_pre_tile  # [BLOCK_V]

        # ----------------------------------------------------------------
        # Shortcut output  (never re-reads S_new from global memory):
        #   out = gate * o_pre + beta * qk * r
        # ----------------------------------------------------------------
        out_tile = gate * o_pre_tile + beta * qk * r_tile  # [BLOCK_V]

        # ----------------------------------------------------------------
        # State update  S_new = gate * S + beta * outer(k, r)
        # outer(k, r)[i, jj] = k[i] * r[jj]
        # ----------------------------------------------------------------
        S_new = gate * S_tile + beta * k_vec[:, None] * r_tile[None, :]  # [DK, BLOCK_V]

        # ----------------------------------------------------------------
        # Write back state tile and output
        # ----------------------------------------------------------------
        tl.store(
            state_ptr + state_base
                + i[:, None] * stride_state_k
                + j[None, :] * stride_state_v,
            S_new.to(state_ptr.dtype.element_ty),
            mask=v_mask[None, :],
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
    ssm_state: torch.Tensor,          # [max_B, nv, DK, DV]  mutated in-place
    ssm_state_indices: torch.Tensor,  # [T]  int32/int64
    nk: int,
    nv: int,
    DK: int,
    DV: int,
) -> torch.Tensor:
    """
    Fused gate + L2-norm + delta-rule recurrent update for a single decode step.

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

    assert mixed_qkv.shape == (T, q_dim + k_dim + v_dim), (
        f"mixed_qkv shape mismatch: expected ({T}, {q_dim+k_dim+v_dim}), "
        f"got {tuple(mixed_qkv.shape)}"
    )
    assert a.shape == (T, nv) and b.shape == (T, nv)
    assert ssm_state.shape[1] == nv and ssm_state.shape[2] == DK and ssm_state.shape[3] == DV

    out = torch.empty(T, nv, DV, dtype=mixed_qkv.dtype, device=mixed_qkv.device)

    BLOCK_V = min(DV, 128)
    grid = (T * nv, DV // BLOCK_V)

    _fused_gdn_decode_kernel[grid](
        mixed_qkv, mixed_qkv.stride(0), q_dim, k_dim,
        a, b, a.stride(0),
        A_log, dt_bias,
        ssm_state,
        ssm_state.stride(0), ssm_state.stride(1),
        ssm_state.stride(2), ssm_state.stride(3),
        ssm_state_indices,
        out, out.stride(0), out.stride(1), out.stride(2),
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
    ssm_state: torch.Tensor,          # [max_B, nv, DK, DV]  mutated in-place
    ssm_state_indices: torch.Tensor,  # [T]
    nk: int,
    nv: int,
    DK: int,
    DV: int,
) -> torch.Tensor:
    """
    Single-step reference implementation of the fused kernel.
    Mathematically equivalent but uses plain PyTorch.
    """
    T = mixed_qkv.shape[0]
    ratio = nv // nk
    q_dim = nk * DK
    k_dim = nk * DK

    orig_dtype = mixed_qkv.dtype
    mv = mixed_qkv.float()

    # split + reshape
    q = mv[:, :q_dim].view(T, nk, DK)
    k = mv[:, q_dim : q_dim + k_dim].view(T, nk, DK)
    v = mv[:, q_dim + k_dim :].view(T, nv, DV)

    # L2 normalise
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    # expand k-heads → v-heads (GQA)
    q = q.repeat_interleave(ratio, dim=1)  # [T, nv, DK]
    k = k.repeat_interleave(ratio, dim=1)  # [T, nv, DK]

    # gate computation
    x = a.float() + dt_bias.float().unsqueeze(0)  # [T, nv]
    softplus_x = F.softplus(x)
    gate = (-A_log.float().exp().unsqueeze(0) * softplus_x).exp()  # [T, nv]
    beta = b.float().sigmoid()                                       # [T, nv]

    # qk scalar per (t, h)
    qk = (q * k).sum(-1)  # [T, nv]

    out = torch.empty(T, nv, DV, device=mixed_qkv.device, dtype=torch.float32)

    for t in range(T):
        sidx = ssm_state_indices[t].item()
        S = ssm_state[sidx].float()  # [nv, DK, DV]

        for h in range(nv):
            S_h    = S[h]          # [DK, DV]
            q_h    = q[t, h]       # [DK]
            k_h    = k[t, h]       # [DK]
            v_h    = v[t, h]       # [DV]
            gate_h = gate[t, h]
            beta_h = beta[t, h]
            qk_h   = qk[t, h]

            o_pre = q_h @ S_h            # [DV]
            r     = v_h - o_pre          # [DV]

            # shortcut output (same formula as the Triton kernel)
            out[t, h] = gate_h * o_pre + beta_h * qk_h * r

            # state update  (mutates ssm_state in-place per-slot)
            S_new = gate_h * S_h + beta_h * torch.outer(k_h, r)
            ssm_state[sidx, h] = S_new.to(ssm_state.dtype)

    return out.to(orig_dtype)

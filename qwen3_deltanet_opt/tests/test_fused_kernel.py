"""
Numerical correctness tests for the fused GDN decode kernel.

Runs the Triton kernel and the pure-PyTorch reference on the same random
inputs and checks that outputs and updated states match within a tolerance.

Run with:
    pytest tests/test_fused_kernel.py -v
or without GPU (reference only):
    pytest tests/test_fused_kernel.py -v -k "ref"
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_inputs(
    T: int,
    nk: int,
    nv: int,
    DK: int,
    DV: int,
    device: str,
    dtype: torch.dtype,
    seed: int = 42,
):
    """Return consistent random tensors for one decode step."""
    torch.manual_seed(seed)
    ratio = nv // nk
    q_dim = nk * DK
    k_dim = nk * DK
    v_dim = nv * DV

    mixed_qkv = torch.randn(T, q_dim + k_dim + v_dim, device=device, dtype=dtype)
    a = torch.randn(T, nv, device=device, dtype=dtype) * 0.1
    b = torch.randn(T, nv, device=device, dtype=dtype)
    A_log = torch.full((nv,), -1.0, device=device, dtype=dtype)
    dt_bias = torch.zeros(nv, device=device, dtype=dtype)

    max_B = T + 4  # a bit larger than T
    # value-major state layout [max_B, nv, DV, DK] — matches vLLM
    ssm_state = torch.randn(max_B, nv, DV, DK, device=device, dtype=torch.float32) * 0.01
    ssm_state_indices = torch.arange(T, device=device, dtype=torch.int32)

    return mixed_qkv, a, b, A_log, dt_bias, ssm_state, ssm_state_indices


# --------------------------------------------------------------------------
# Reference self-consistency test  (no GPU required)
# --------------------------------------------------------------------------

class TestRefGdnDecode:
    """The PyTorch reference should match a manual loop."""

    @pytest.mark.parametrize("T,nk,nv,DK,DV", [
        (1, 2, 4, 8, 8),
        (4, 2, 4, 16, 16),
    ])
    def test_ref_output_shape(self, T, nk, nv, DK, DV):
        from dng_opt.kernels.fused_gdn_decode import ref_gdn_decode

        device = "cpu"
        dtype = torch.float32
        mq, a, b, A_log, dt_bias, state, idx = _make_inputs(T, nk, nv, DK, DV, device, dtype)
        out = ref_gdn_decode(mq, a, b, A_log, dt_bias, state.clone(), idx, nk, nv, DK, DV)
        assert out.shape == (T, nv, DV)
        assert out.dtype == dtype

    @pytest.mark.parametrize("T,nk,nv,DK,DV", [
        (1, 1, 2, 8, 8),
        (2, 2, 4, 16, 16),
    ])
    def test_ref_matches_manual_loop(self, T, nk, nv, DK, DV):
        """Reference impl should match the explicit loop from the docstring."""
        from dng_opt.kernels.fused_gdn_decode import ref_gdn_decode

        device = "cpu"
        dtype = torch.float32
        mq, a, b, A_log, dt_bias, state_ref, idx = _make_inputs(T, nk, nv, DK, DV, device, dtype)
        state_manual = state_ref.clone()
        state_ref2   = state_ref.clone()

        # ---- manual loop ----
        ratio = nv // nk
        q_dim = nk * DK
        k_dim = nk * DK
        scale = DK ** -0.5
        q = (F.normalize(mq[:, :q_dim].view(T, nk, DK), dim=-1) * scale).repeat_interleave(ratio, 1)
        k = F.normalize(mq[:, q_dim:q_dim+k_dim].view(T, nk, DK), dim=-1).repeat_interleave(ratio, 1)
        v = mq[:, q_dim+k_dim:].view(T, nv, DV)
        x = a.float() + dt_bias.float()
        g_decay = (-A_log.float().exp() * F.softplus(x)).exp()  # [T, nv]
        beta = b.float().sigmoid()
        qk = (q * k).sum(-1)

        out_manual = torch.zeros(T, nv, DV)
        for t in range(T):
            for h in range(nv):
                S = state_manual[idx[t], h].float()   # [DV, DK]
                S_dec = g_decay[t, h] * S              # decay first
                o_pre_k = S_dec @ k[t, h]              # [DV]
                o_pre_q = S_dec @ q[t, h]              # [DV]
                r = beta[t, h] * (v[t, h].float() - o_pre_k)
                out_manual[t, h] = o_pre_q + r * qk[t, h]
                state_manual[idx[t], h] = (S_dec + torch.outer(r, k[t, h])).to(state_manual.dtype)

        # ---- reference function ----
        out_ref = ref_gdn_decode(mq, a, b, A_log, dt_bias, state_ref2, idx, nk, nv, DK, DV)

        torch.testing.assert_close(out_ref, out_manual.to(dtype), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(state_ref2, state_manual, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------
# Triton kernel vs reference  (requires CUDA)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFusedKernelCuda:
    """The Triton kernel must produce results matching the PyTorch reference."""

    @pytest.mark.parametrize("T,nk,nv,DK,DV", [
        (1, 2, 4, 32, 32),
        (4, 2, 4, 64, 64),
        (8, 4, 8, 64, 64),
    ])
    def test_kernel_matches_reference(self, T, nk, nv, DK, DV):
        from dng_opt.kernels.fused_gdn_decode import fused_gdn_decode, ref_gdn_decode

        device = "cuda"
        dtype = torch.bfloat16

        mq, a, b, A_log, dt_bias, state_base, idx = _make_inputs(
            T, nk, nv, DK, DV, device, dtype, seed=7
        )
        state_ref    = state_base.clone()
        state_fused  = state_base.clone()

        # Reference (CPU-backed for precision).  Bind the CPU copy so we
        # compare against the SAME tensor that ref_gdn_decode mutates in-place
        # (state_ref.cpu() would create a throwaway copy and leave state_ref
        # unmodified, making the state assertion below meaningless).
        state_ref_cpu = state_ref.cpu()
        out_ref = ref_gdn_decode(
            mq.cpu().float(), a.cpu().float(), b.cpu().float(),
            A_log.cpu().float(), dt_bias.cpu().float(),
            state_ref_cpu, idx.cpu(),
            nk, nv, DK, DV,
        ).to(device=device, dtype=dtype)
        state_ref_gpu = state_ref_cpu.cuda()

        # Fused Triton kernel
        out_fused = fused_gdn_decode(mq, a, b, A_log, dt_bias, state_fused, idx, nk, nv, DK, DV)

        torch.testing.assert_close(out_fused, out_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            state_fused, state_ref_gpu.to(state_fused.dtype),
            rtol=1e-2, atol=1e-2,
        )

    @pytest.mark.parametrize("T", [1, 4, 16])
    def test_kernel_large_dim(self, T):
        """Test with Qwen3.5-typical dimensions: DK=DV=128."""
        from dng_opt.kernels.fused_gdn_decode import fused_gdn_decode, ref_gdn_decode

        nk, nv, DK, DV = 2, 4, 128, 128
        device = "cuda"
        dtype = torch.bfloat16

        mq, a, b, A_log, dt_bias, state_base, idx = _make_inputs(
            T, nk, nv, DK, DV, device, dtype, seed=99
        )
        state_ref   = state_base.clone()
        state_fused = state_base.clone()

        out_ref = ref_gdn_decode(
            mq.cpu().float(), a.cpu().float(), b.cpu().float(),
            A_log.cpu().float(), dt_bias.cpu().float(),
            state_ref.cpu(), idx.cpu(),
            nk, nv, DK, DV,
        ).to(device=device, dtype=dtype)

        out_fused = fused_gdn_decode(mq, a, b, A_log, dt_bias, state_fused, idx, nk, nv, DK, DV)

        # Looser tolerance for large dims (bf16 accumulation)
        torch.testing.assert_close(out_fused, out_ref, rtol=2e-2, atol=2e-2)

    def test_state_updated_inplace(self):
        """Kernel must modify ssm_state in-place (same storage)."""
        from dng_opt.kernels.fused_gdn_decode import fused_gdn_decode

        T, nk, nv, DK, DV = 2, 2, 4, 32, 32
        device, dtype = "cuda", torch.bfloat16
        mq, a, b, A_log, dt_bias, state, idx = _make_inputs(T, nk, nv, DK, DV, device, dtype)
        state_before = state.clone()
        _ = fused_gdn_decode(mq, a, b, A_log, dt_bias, state, idx, nk, nv, DK, DV)
        # state must differ from initial value (it was updated)
        assert not torch.allclose(state, state_before), "State was not updated in-place"

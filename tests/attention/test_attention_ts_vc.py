"""VC-Attention on the prims_ts context kernel: E4M3 Q/K with per-block scales and
E4M3 V tile residuals whose bf16 tile means the kernel restores in the online
softmax. Compared against the fp32 attention over the dequantized operands."""

import math

import pytest
import torch

from flashinfer.attention.prims_ts import vc_attention as vca
from flashinfer.attention.prims_ts.context import BatchPrefillTSWrapper

_REQUIRES_SM100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="VC-Attention context requires an SM100-family GPU",
)


@pytest.fixture(autouse=True)
def _exact_fp32_matmul():
    """The torch reference needs true fp32 matmuls; NGC containers export
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which would turn them into TF32."""
    prev_prec = torch.get_float32_matmul_precision()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(prev_prec)
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def _structured_values(shape, device):
    """Values with a strong per-token component so tile means matter."""
    b, s, h, d = shape
    return torch.randn(shape, device=device) * 0.3 + torch.randn(
        b, s, h, 1, device=device
    ) * torch.randn(b, 1, h, d, device=device)


def _run_vc(batch, seq_len, heads, *, q_block_size=128, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    head_dim = 128
    q = torch.randn(batch, seq_len, heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = _structured_values(q.shape, device)
    ops = vca.vc_quantize(q, k, v, q_block_size=q_block_size)
    sm_scale = 1.0 / math.sqrt(head_dim)
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=device,
        batch_size=batch,
        max_seq_len_q=seq_len,
        max_kv_len=seq_len,
        num_qo_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
        q_dtype=torch.float8_e4m3fn,
        k_dtype=torch.float8_e4m3fn,
        v_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        packed=False,
        mask_type="dense",
        sm_scale=sm_scale,
        vc_attention=True,
        vc_q_block_size=q_block_size,
    )
    out = wrapper.run(
        ops.q,
        ops.k,
        ops.v,
        output_scale=ops.v_scale.reshape(-1).contiguous(),
        vc_mu=ops.mu,
        vc_q_scale=ops.q_scale,
        vc_k_scale=ops.k_scale,
    )
    reference = vca.vc_reference(ops, sm_scale=sm_scale)
    exact = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    return out.float(), reference, exact


@pytest.mark.arch_blackwell
@_REQUIRES_SM100
@pytest.mark.parametrize(
    "batch,seq_len,heads,q_block_size",
    [
        (2, 2048, 2, 128),
        (1, 2000, 2, 128),  # partial final K/V tile
        (1, 1024, 1, 64),  # Q scale blocks narrower than the Q tile
    ],
)
def test_vc_attention_matches_dequantized_reference(
    batch, seq_len, heads, q_block_size
):
    out, reference, exact = _run_vc(batch, seq_len, heads, q_block_size=q_block_size)
    assert torch.isfinite(out).all()
    # ExpCast-FP8 codes P with Mitchell's log2 approximation (<= 7.5% per
    # element, bounded total variation), plus bf16 output rounding.
    rel_l2 = torch.linalg.vector_norm(out - reference) / torch.linalg.vector_norm(
        reference
    )
    assert float(rel_l2) < 6e-2
    torch.testing.assert_close(out, reference, rtol=1e-1, atol=1e-1)
    # The recipe itself stays close to unquantized attention.
    rel_exact = torch.linalg.vector_norm(out - exact) / torch.linalg.vector_norm(exact)
    assert float(rel_exact) < 1e-1


@pytest.mark.arch_blackwell
@_REQUIRES_SM100
def test_vc_attention_tile_mean_restored_exactly():
    """A constant tile mean must shift every output by exactly that constant."""
    from dataclasses import replace

    torch.manual_seed(1)
    device = torch.device("cuda")
    b, s, h, d = 1, 1024, 1, 128
    q = torch.randn(b, s, h, d, device=device)
    k = torch.randn_like(q)
    v = _structured_values(q.shape, device)
    ops = vca.vc_quantize(q, k, v)
    zero = torch.zeros_like(ops.mean)
    ops_zero = replace(ops, mean=zero, mu=vca.pack_vc_tile_means(zero, ops.v_scale))
    one = torch.ones_like(ops.mean)
    ops_one = replace(ops, mean=one, mu=vca.pack_vc_tile_means(one, ops.v_scale))
    sm_scale = 1.0 / math.sqrt(d)
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=device,
        batch_size=b,
        max_seq_len_q=s,
        max_kv_len=s,
        num_qo_heads=h,
        num_kv_heads=h,
        head_dim=d,
        q_dtype=torch.float8_e4m3fn,
        k_dtype=torch.float8_e4m3fn,
        v_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        packed=False,
        mask_type="dense",
        sm_scale=sm_scale,
        vc_attention=True,
    )
    outs = [
        wrapper.run(
            o.q,
            o.k,
            o.v,
            output_scale=o.v_scale.reshape(-1).contiguous(),
            vc_mu=o.mu,
            vc_q_scale=o.q_scale,
            vc_k_scale=o.k_scale,
        ).float()
        for o in (ops_zero, ops_one)
    ]
    torch.testing.assert_close(
        outs[1] - outs[0], torch.ones_like(outs[0]), rtol=0, atol=1.6e-2
    )


def test_vc_plan_rejects_unsupported_recipes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wrapper = BatchPrefillTSWrapper()
    common = dict(
        device="cuda",
        batch_size=1,
        max_seq_len_q=256,
        max_kv_len=256,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=128,
        out_dtype=torch.bfloat16,
        vc_attention=True,
    )
    with pytest.raises(NotImplementedError):
        wrapper.plan(
            q_dtype=torch.bfloat16,
            k_dtype=torch.bfloat16,
            v_dtype=torch.float8_e4m3fn,
            **common,
        )
    with pytest.raises(NotImplementedError):
        wrapper.plan(
            q_dtype=torch.float8_e4m3fn,
            k_dtype=torch.float8_e4m3fn,
            v_dtype=torch.float8_e4m3fn,
            mask_type="causal",
            **common,
        )
    with pytest.raises(ValueError):
        wrapper.plan(
            q_dtype=torch.float8_e4m3fn,
            k_dtype=torch.float8_e4m3fn,
            v_dtype=torch.float8_e4m3fn,
            vc_q_block_size=96,
            **common,
        )


def test_pack_vc_tile_means_layout():
    mean = torch.arange(128, dtype=torch.float32).view(1, 1, 1, 128) * 0.25
    packed = vca.pack_vc_tile_means(mean, 0.5).view(1, 1, 1, 16, 2, 8, 8)
    expected = (mean.view(16, 8) / 0.5).to(torch.bfloat16)
    assert torch.equal(packed[0, 0, 0, :, 0, :, 0], expected)
    assert torch.equal(packed[0, 0, 0, :, 0, :, 1], expected)
    assert packed[0, 0, 0, :, 0, :, 2:].eq(0).all()
    assert packed[0, 0, 0, :, 1].eq(0).all()


@_REQUIRES_SM100
@pytest.mark.parametrize("batch,seq_len,heads", [(1, 2000, 2), (2, 4096, 4)])
def test_vc_quantize_fused_matches_reference(batch, seq_len, heads):
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(batch, seq_len, heads, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = _structured_values(q.shape, device).to(torch.bfloat16)
    perm = vca.vc_token_permutation(v)
    ref = vca.vc_quantize(q, k, v, perm=perm)
    fused = vca.vc_quantize_fused(q, k, v, perm=perm)
    torch.testing.assert_close(fused.q_scale, ref.q_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(fused.k_scale, ref.k_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(fused.v_scale, ref.v_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(fused.mean, ref.mean, rtol=1e-5, atol=1e-6)
    assert torch.equal(fused.mu, ref.mu)
    for name in ("q", "k", "v"):
        a = getattr(fused, name).float()
        b = getattr(ref, name).float()
        # E4M3 rounding ties may differ by one code between the two paths.
        assert (a != b).float().mean().item() < 1e-3, name


@_REQUIRES_SM100
def test_vc_wrapper_quantizes_bf16_inputs():
    """run() under a VC plan accepts bf16 Q/K/V and quantizes internally."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seq_len, heads, head_dim = 1, 2048, 2, 128
    q = torch.randn(
        batch, seq_len, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = _structured_values(q.shape, device).to(torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(head_dim)
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=device,
        batch_size=batch,
        max_seq_len_q=seq_len,
        max_kv_len=seq_len,
        num_qo_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
        q_dtype=torch.float8_e4m3fn,
        k_dtype=torch.float8_e4m3fn,
        v_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        sm_scale=sm_scale,
        vc_attention=True,
    )
    out = wrapper.run(q, k, v)
    ops = vca.vc_quantize_fused(
        q, k, v, perm=wrapper._vc_perms[(batch, seq_len, heads)]["perm"]
    )
    ref = vca.vc_reference(ops, sm_scale=sm_scale)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 6e-2, rel


@_REQUIRES_SM100
def test_vc_hadamard_is_orthonormal_and_qk_invariant():
    device = torch.device("cuda")
    hm = vca.hadamard_matrix(128, device)
    torch.testing.assert_close(
        hm @ hm.T, torch.eye(128, device=device), atol=1e-5, rtol=0
    )
    torch.manual_seed(0)
    q = torch.randn(1, 512, 2, 128, device=device)
    k = torch.randn_like(q)
    a = torch.einsum("bqhd,bkhd->bhqk", q, k)
    b = torch.einsum("bqhd,bkhd->bhqk", q @ hm, k @ hm)
    torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-4)


@_REQUIRES_SM100
def test_vc_wrapper_step_schedule():
    """First 25% of steps: grouping + demeaning (refreshed every 4 steps); afterwards demeaning
    is off and the kernel keeps the permutation the window left behind (paper Sec. 3.4)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seq_len, heads, head_dim = 1, 1024, 1, 128
    q = torch.randn(
        batch, seq_len, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = _structured_values(q.shape, device).to(torch.bfloat16)
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=device,
        batch_size=batch,
        max_seq_len_q=seq_len,
        max_kv_len=seq_len,
        num_qo_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
        q_dtype=torch.float8_e4m3fn,
        k_dtype=torch.float8_e4m3fn,
        v_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        sm_scale=1.0 / math.sqrt(head_dim),
        vc_attention=True,
    )
    exact = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
    ).transpose(1, 2)
    key = (batch, seq_len, heads)
    wrapper.run(q, k, v, vc_denoise_step=(0, 40))
    p0 = wrapper._vc_perms[key]["perm"].clone()
    wrapper.run(q, k, v, vc_denoise_step=(1, 40))
    assert torch.equal(wrapper._vc_perms[key]["perm"], p0)  # no refresh at step 1
    wrapper.run(q * 1.0, k, v.flip(1), vc_denoise_step=(4, 40))
    assert wrapper._vc_perms[key]["step"] == 4  # refreshed at step 4
    out_late = wrapper.run(q, k, v, vc_denoise_step=(20, 40))  # V-Smooth off
    rel = ((out_late.float() - exact).norm() / exact.norm()).item()
    assert rel < 1e-1, rel

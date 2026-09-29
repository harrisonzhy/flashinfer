"""VC-Attention host-side preprocessing for the prims_ts context kernel.

VC-Attention (arXiv 2609.15810) is SageAttention2-style 8-bit attention with
two changes on the value side: a token permutation that groups similar value
rows into the same 128-token K/V tile, and per-tile value smoothing. Each V
tile is stored as E4M3 residuals around its tile mean; the kernel restores the
mean inside the online softmax as ``O += rowsum(P_tile) * mean_tile`` (one
bf16 K=16 UMMA step per tile), so the residual quantization error no longer
carries the value DC component.

This module provides the reference preprocessing:

* :func:`vc_token_permutation` -- online k-means over value rows, one
  permutation per (batch, head), applied to K and V (attention is invariant to
  a common key/value permutation);
* :func:`vc_quantize` -- Q/K E4M3 quantization with one scale per Q block and
  per K/V tile (after key channel-mean centring and a 128-point Hadamard
  rotation of Q and K), V tile-mean subtraction and per-channel E4M3 residual
  quantization, and the packed kernel operands;
* :func:`pack_vc_tile_means` -- the bf16 mean operand layout the kernel's
  unswizzled K-major descriptor expects;
* :func:`vc_reference` -- the fp32 attention over the dequantized operands.

The scales and means are plain tensors; the kernel takes them through
:meth:`BatchPrefillTSWrapper.run` (``vc_mu``, ``vc_q_scale``, ``vc_k_scale``)
and the per-tensor V scale through ``output_scale``.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch

KV_TILE = 128
E4M3_MAX = 448.0
VC_KMEANS_CLUSTERS = 64
_MEAN_MMA_K = 16


@dataclass(frozen=True)
class VCAttentionOperands:
    """Everything one VC-Attention forward needs, in kernel layout.

    ``q``, ``k``, ``v`` are E4M3 ``[B, S, H, D]``; ``k`` and ``v`` are stored in
    permuted token order. ``v`` holds tile residuals ``(V[perm] - mean) / v_scale``.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor  # [B, H, ceil(S_q / q_block)] fp32
    k_scale: torch.Tensor  # [B, H, num_kv_tiles] fp32
    v_scale: (
        torch.Tensor
    )  # [B, H, D] fp32 per-channel E4M3 residual scale (feeds output_scale)
    mean: torch.Tensor  # [B, H, num_kv_tiles, D] fp32 tile means (permuted order)
    mu: torch.Tensor  # [B, H, num_kv_tiles, 8, 256] bf16 packed kernel operand
    perm: torch.Tensor  # [B, H, S_k] int64 token permutation
    q_block_size: int


def _blocks(length: int, block: int) -> int:
    return (length + block - 1) // block


@functools.lru_cache(maxsize=None)
def _hadamard_matrix_cached(d: int, device: str) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(d)).to(device)


def hadamard_matrix(d: int, device: torch.device) -> torch.Tensor:
    """Normalised Sylvester Hadamard matrix ``[d, d]`` (``d`` a power of two)."""
    if d & (d - 1):
        raise ValueError("head_dim must be a power of two for the Hadamard rotation")
    return _hadamard_matrix_cached(d, str(device))


def _pad_tokens(x: torch.Tensor, block: int) -> torch.Tensor:
    """Zero-pad the token axis (dim 1) of ``[B, S, H, D]`` up to a block multiple."""
    pad = _blocks(x.shape[1], block) * block - x.shape[1]
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))


def _block_amax_scale(x: torch.Tensor, block: int) -> torch.Tensor:
    """Per-(batch, head, token block) E4M3 scale of ``[B, S, H, D]``: amax / 448."""
    b, s, h, d = x.shape
    xp = _pad_tokens(x.float().abs(), block)
    nb = xp.shape[1] // block
    amax = xp.view(b, nb, block, h, d).amax(dim=(2, 4))  # [B, nb, H]
    return (
        (amax / E4M3_MAX).clamp_min(1e-12).permute(0, 2, 1).contiguous()
    )  # [B, H, nb]


def _quantize_blocks(x: torch.Tensor, scale: torch.Tensor, block: int) -> torch.Tensor:
    """Divide ``[B, S, H, D]`` by its ``[B, H, nb]`` block scale and round to E4M3."""
    b, s, h, d = x.shape
    per_token = scale.permute(0, 2, 1).repeat_interleave(block, dim=1)[
        :, :s
    ]  # [B, S, H]
    return (x.float() / per_token.unsqueeze(-1)).to(torch.float8_e4m3fn)


@torch.no_grad()
def vc_token_permutation_with_centroids(
    v: torch.Tensor,
    *,
    num_clusters: int | None = None,
    iters: int = 3,
    generator: torch.Generator | None = None,
    init_centroids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k-means per (batch, head) over value rows; returns ``(perm, centroids)``.

    ``init_centroids`` (``[B*H, k, D]`` from a previous call) warm-starts the
    clustering, as the paper does when the grouping is refreshed a few
    denoising steps later.
    """
    b, s, h, d = v.shape
    if num_clusters is None:
        # The paper does not fix k; 64 clusters keep the online k-means at
        # ~0.1 s per layer call at 147.6k tokens while still grouping the
        # 128-token tiles by value similarity.
        num_clusters = VC_KMEANS_CLUSTERS
    n_groups = b * h
    # Memory: work on a few heads of one batch item at a time so the fp32 copy of V
    # ([G, S, D]) and the distance matrix ([G, S, k]) each stay under ~0.75 GB.
    per_group = 4 * s * max(d, num_clusters)
    chunk_h = max(1, min(h, int(0.75e9 // per_group)))
    perm_out = torch.empty((b, h, s), dtype=torch.int64, device=v.device)
    cent_out = torch.empty(
        (n_groups, num_clusters, d), dtype=torch.float32, device=v.device
    )
    warm = init_centroids is not None and tuple(init_centroids.shape) == (
        n_groups,
        num_clusters,
        d,
    )
    for bi in range(b):
        for h0 in range(0, h, chunk_h):
            h1 = min(h, h0 + chunk_h)
            gsz = h1 - h0
            g0 = bi * h + h0
            x = v[bi, :, h0:h1, :].permute(1, 0, 2).float()  # [G, S, D]
            if warm:
                centroids = init_centroids[g0 : g0 + gsz].to(x.dtype)
            else:
                idx = torch.stack(
                    [
                        torch.randperm(s, device=v.device, generator=generator)[
                            :num_clusters
                        ]
                        for _ in range(gsz)
                    ]
                )  # [G, k]
                centroids = torch.gather(
                    x, 1, idx.unsqueeze(-1).expand(-1, -1, d)
                )  # [G, k, D]
            x_sq = (x * x).sum(-1, keepdim=True)  # [G, S, 1]
            labels = None
            for _ in range(iters):
                c_sq = (centroids * centroids).sum(-1).unsqueeze(1)  # [G, 1, k]
                dist = x_sq + c_sq - 2.0 * torch.bmm(x, centroids.transpose(1, 2))
                labels = dist.argmin(dim=-1)  # [G, S]
                del dist
                counts = torch.zeros(
                    gsz, num_clusters, device=v.device, dtype=torch.float32
                )
                counts.scatter_add_(
                    1, labels, torch.ones_like(labels, dtype=torch.float32)
                )
                sums = torch.zeros_like(centroids)
                sums.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, d), x)
                nonempty = counts > 0
                centroids = torch.where(
                    nonempty.unsqueeze(-1),
                    sums / counts.clamp_min(1.0).unsqueeze(-1),
                    centroids,
                )
            assert labels is not None
            perm_out[bi, h0:h1] = torch.argsort(labels, dim=-1, stable=True)
            cent_out[g0 : g0 + gsz] = centroids
            del x, x_sq
    return perm_out, cent_out


@torch.no_grad()
def vc_token_permutation(
    v: torch.Tensor,
    *,
    num_clusters: int | None = None,
    iters: int = 3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Group similar value rows: plain k-means per (batch, head), then argsort labels.

    Returns an int64 permutation ``[B, H, S]`` such that ``V[b, perm[b, h], h]``
    lists the tokens cluster by cluster, so 128-token K/V tiles hold similar
    values and their tile means remove most of the value energy.
    """
    return vc_token_permutation_with_centroids(
        v, num_clusters=num_clusters, iters=iters, generator=generator
    )[0]


def pack_vc_tile_means(
    mean: torch.Tensor, v_scale: torch.Tensor | float
) -> torch.Tensor:
    """Pack ``[B, H, T, D]`` tile means into the kernel's bf16 UMMA operand.

    ``v_scale`` is the per-channel residual scale ``[B, H, D]`` (or a scalar);
    the means are stored divided by it.

    The kernel multiplies a ``[128 x 16]`` row-sum operand (``k=0`` bf16 high
    part, ``k=1`` bf16 remainder of each row's tile sum) by this ``[D x 16]``
    K-major operand, so rows ``k=0`` and ``k=1`` both carry ``mean / v_scale``
    and the other 14 K rows are zero. The tile is laid out as the unswizzled
    tcgen05 K-major core-matrix order: 8-row groups of 256 bytes, two 128-byte
    core matrices (k 0-7, k 8-15) of 8 rows x 16 bytes.
    """
    b, h, t, d = mean.shape
    if d % 8 != 0:
        raise ValueError("head_dim must be a multiple of 8")
    vs = (
        v_scale.reshape(b, h, 1, d)
        if isinstance(v_scale, torch.Tensor) and v_scale.dim() == 3
        else v_scale
    )
    m = (mean.float() / vs).to(torch.bfloat16)
    tile = torch.zeros(
        (b, h, t, d // 8, 2, 8, _MEAN_MMA_K // 2),
        dtype=torch.bfloat16,
        device=mean.device,
    )
    rows = m.view(b, h, t, d // 8, 8)
    tile[..., 0, :, 0] = rows
    tile[..., 0, :, 1] = rows
    return tile.reshape(b, h, t, (d * _MEAN_MMA_K * 2) // 512, 256).contiguous()


@torch.no_grad()
def vc_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_block_size: int = KV_TILE,
    perm: torch.Tensor | None = None,
    kmeans_iters: int = 3,
    generator: torch.Generator | None = None,
    hadamard: bool = True,
    demean: bool = True,
) -> VCAttentionOperands:
    """Quantize ``[B, S, H, D]`` Q/K/V into VC-Attention kernel operands.

    Keys are centred by their per-(batch, head) channel mean over tokens
    (softmax-invariant), permuted with the values, rotated (with Q) by the
    normalised Hadamard matrix when ``hadamard`` is set, and quantized to E4M3
    with one scale per 128-token tile; queries take one E4M3 scale per
    ``q_block_size`` tokens. Values are permuted, split into 128-token tile
    means and residuals (``demean=False`` keeps the means at zero), and the
    residuals are quantized to E4M3 with one scale per (batch, head, channel)
    that the caller passes as ``output_scale``.
    """
    if q.dim() != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("q, k, v must be [B, S, H, D] with matching batch and heads")
    b, s_k, h, d = k.shape
    if perm is None:
        perm = vc_token_permutation(v, iters=kmeans_iters, generator=generator)
    gather_idx = (
        perm.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, d)
    )  # [B, S, H, D]
    k_smooth = k.float() - k.float().mean(dim=1, keepdim=True)
    k_p = torch.gather(k_smooth, 1, gather_idx)
    v_p = torch.gather(v.float(), 1, gather_idx)
    q_f = q.float()
    if hadamard:
        hm = hadamard_matrix(d, q.device)
        q_f = q_f @ hm
        k_p = k_p @ hm

    q_scale = _block_amax_scale(q_f, q_block_size)
    k_scale = _block_amax_scale(k_p, KV_TILE)
    q8 = _quantize_blocks(q_f, q_scale, q_block_size)
    k8 = _quantize_blocks(k_p, k_scale, KV_TILE)

    num_kv_tiles = _blocks(s_k, KV_TILE)
    v_pad = _pad_tokens(v_p, KV_TILE).view(b, num_kv_tiles, KV_TILE, h, d)
    valid = torch.zeros(num_kv_tiles, KV_TILE, device=v.device, dtype=torch.float32)
    valid.view(-1)[:s_k] = 1.0
    counts = valid.sum(dim=1).clamp_min(1.0)  # [T]
    mean = (v_pad * valid.view(1, num_kv_tiles, KV_TILE, 1, 1)).sum(
        dim=2
    ) / counts.view(1, num_kv_tiles, 1, 1)  # [B, T, H, D]
    if not demean:
        mean = torch.zeros_like(mean)
    residual = (v_pad - mean.unsqueeze(2)) * valid.view(1, num_kv_tiles, KV_TILE, 1, 1)
    residual = residual.view(b, num_kv_tiles * KV_TILE, h, d)[:, :s_k]
    # One E4M3 scale per (batch, head, channel) over all tokens (paper).
    v_scale = (
        (residual.abs().amax(dim=1) / E4M3_MAX).clamp_min(1e-12).float()
    )  # [B, H, D]
    v8 = (residual / v_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    mean = mean.permute(0, 2, 1, 3).contiguous()  # [B, H, T, D]
    return VCAttentionOperands(
        q=q8.contiguous(),
        k=k8.contiguous(),
        v=v8.contiguous(),
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        mean=mean,
        mu=pack_vc_tile_means(mean, v_scale),
        perm=perm,
        q_block_size=q_block_size,
    )


# ---------------------------------------------------------------------------
# Fused preprocessing (CuTe DSL kernels in kernels/vc_prepare.py)
# ---------------------------------------------------------------------------
@torch.no_grad()
def vc_quantize_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_block_size: int = KV_TILE,
    perm: torch.Tensor | None = None,
    kmeans_iters: int = 3,
    generator: torch.Generator | None = None,
    hadamard: bool = True,
    demean: bool = True,
) -> VCAttentionOperands:
    """Same contract as :func:`vc_quantize`, computed by three CuTe DSL kernels
    (:mod:`flashinfer.attention.prims_ts.kernels.vc_prepare`).

    Pass 1 gathers each permuted 128-token K/V tile once, smooths and quantizes
    K (per-tile scale), and writes the V tile mean and residual amax; a scalar
    reduction over the tile amaxes gives the per-tensor V scale; pass 2 writes
    the E4M3 residuals and the packed bf16 mean operand; Q blocks are one more
    kernel. Every tensor is read once (V twice) and written once. Requires
    head_dim 128, bf16/fp16 inputs and ``q_block_size == 128``; other cases
    fall back to :func:`vc_quantize`.
    """
    if q.dim() != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("q, k, v must be [B, S, H, D] with matching batch and heads")
    if (
        q.shape[-1] != 128
        or q_block_size != KV_TILE
        or q.dtype not in (torch.bfloat16, torch.float16)
        or not q.is_cuda
    ):
        return vc_quantize(
            q,
            k,
            v,
            q_block_size=q_block_size,
            perm=perm,
            kmeans_iters=kmeans_iters,
            generator=generator,
            hadamard=hadamard,
            demean=demean,
        )
    from .kernels.vc_prepare import vc_prepare

    if perm is None:
        perm = vc_token_permutation(v, iters=kmeans_iters, generator=generator)
    q8, k8, v8, q_scale, k_scale, v_scale, mean, mu = vc_prepare(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        perm,
        hadamard=hadamard,
        demean=demean,
    )
    return VCAttentionOperands(
        q=q8,
        k=k8,
        v=v8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        mean=mean,
        mu=mu,
        perm=perm,
        q_block_size=q_block_size,
    )


@torch.no_grad()
def vc_reference(
    ops: VCAttentionOperands, *, sm_scale: float | None = None
) -> torch.Tensor:  # noqa: D401
    """fp32 attention over the dequantized VC operands (what the kernel computes,
    up to E4M3 P rounding), as ``[B, S_q, H, D]`` fp32."""
    b, s_q, h, d = ops.q.shape
    s_k = ops.k.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    q = ops.q.float() * ops.q_scale.permute(0, 2, 1).repeat_interleave(
        ops.q_block_size, dim=1
    )[:, :s_q].unsqueeze(-1)
    k = ops.k.float() * ops.k_scale.permute(0, 2, 1).repeat_interleave(KV_TILE, dim=1)[
        :, :s_k
    ].unsqueeze(-1)
    # The kernel restores bf16(mean / v_scale) * v_scale, per channel.
    vs = ops.v_scale.reshape(b, ops.k.shape[2], 1, d)  # [B, H, 1, D]
    mean = (ops.mean / vs).to(torch.bfloat16).float() * vs
    mean_tokens = mean.permute(0, 2, 1, 3).repeat_interleave(KV_TILE, dim=1)[:, :s_k]
    v = ops.v.float() * ops.v_scale.unsqueeze(1) + mean_tokens
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


__all__ = [
    "KV_TILE",
    "VCAttentionOperands",
    "hadamard_matrix",
    "vc_token_permutation_with_centroids",
    "pack_vc_tile_means",
    "vc_quantize",
    "vc_quantize_fused",
    "vc_reference",
    "vc_token_permutation",
]

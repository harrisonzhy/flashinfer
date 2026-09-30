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
from .sage import flat_scale_numel, flat_scale_slot, log2_block_size

_MEAN_MMA_K = 16
# The kernel restores one V tile mean per 128-token K/V tile; K scales share it.
VC_K_BLOCK_SIZE = KV_TILE
_VC_Q_BLOCK_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class VCAttentionConfig:
    """Compile-time VC-Attention recipe of one plan (arXiv 2609.15810).

    ``q_block_size`` tokens share one Q scale (a power of two up to 256);
    ``k_block_size`` is fixed at 128, the K/V tile whose bf16 value mean the
    kernel restores. ``smooth_step_fraction`` and ``perm_refresh_every`` are
    the V-Smooth schedule of :meth:`BatchPrefillTSWrapper.run` with
    ``vc_denoise_step``: grouping and demeaning run on the first fraction of
    the denoising steps, the token permutation is recomputed every
    ``perm_refresh_every`` steps inside that window and kept afterwards.
    ``kmeans_clusters`` (``None`` = 64) and ``kmeans_iters`` parametrize the
    online k-means grouping. The defaults are the paper's Wan2.2 settings.
    """

    q_block_size: int = KV_TILE
    k_block_size: int = VC_K_BLOCK_SIZE
    smooth_step_fraction: float = 0.25
    perm_refresh_every: int = 4
    kmeans_clusters: int | None = None
    kmeans_iters: int = 3

    def __post_init__(self) -> None:
        if self.q_block_size not in _VC_Q_BLOCK_SIZES:
            raise ValueError(
                f"q_block_size must be a power of two in [1, 256], got {self.q_block_size}"
            )
        if self.k_block_size != VC_K_BLOCK_SIZE:
            raise ValueError(f"k_block_size must be {VC_K_BLOCK_SIZE}")
        if not (0.0 <= self.smooth_step_fraction <= 1.0):
            raise ValueError("smooth_step_fraction must be in [0, 1]")
        if self.perm_refresh_every < 1:
            raise ValueError("perm_refresh_every must be at least 1")
        if self.kmeans_clusters is not None and self.kmeans_clusters < 1:
            raise ValueError("kmeans_clusters must be at least 1")
        if self.kmeans_iters < 1:
            raise ValueError("kmeans_iters must be at least 1")

    @property
    def q_block_log2(self) -> int:
        return log2_block_size(self.q_block_size)


@dataclass(frozen=True)
class VCAttentionParams:
    """Per-run VC-Attention operands beyond the E4M3 ``q``, ``k``, ``v``.

    ``q_scale`` is ``[Hq, flat_scale_numel(B, Sq, q_block_size)]`` and
    ``k_scale`` ``[Hkv, flat_scale_numel(B, Skv, 128)]`` fp32 in the flat
    scale layout of :mod:`flashinfer.attention.prims_ts.sage`; ``v_scale`` is
    the ``[B, Hkv, D]`` fp32 per-channel E4M3 residual scale; ``tile_means``
    is the packed bf16 mean operand ``[B, Hkv, num_kv_tiles, 8, 256]`` from
    :func:`pack_vc_tile_means` (means already divided by ``v_scale``); ``demean``
    says whether the run restores them (``False`` after the V-Smooth window,
    when they are zero). Scales must be positive and finite; the kernel does
    not check them.
    """

    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    tile_means: torch.Tensor
    # False after the V-Smooth window: the tile means are zero and the kernel
    # runs the plain low-bit recipe without the mean-restore steps.
    demean: bool = True


def vc_scale_shapes(
    config: VCAttentionConfig,
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, tuple[int, ...]]:
    """Return the shape of every :class:`VCAttentionParams` tensor, by field."""
    num_kv_tiles = _blocks(seq_len_kv, KV_TILE)
    return {
        "q_scale": (
            num_qo_heads,
            flat_scale_numel(batch_size, seq_len_q, config.q_block_size),
        ),
        "k_scale": (
            num_kv_heads,
            flat_scale_numel(batch_size, seq_len_kv, config.k_block_size),
        ),
        "v_scale": (batch_size, num_kv_heads, head_dim),
        "tile_means": (batch_size, num_kv_heads, num_kv_tiles, 8, 256),
    }


_VC_PARAM_DTYPES = {
    "q_scale": torch.float32,
    "k_scale": torch.float32,
    "v_scale": torch.float32,
    "tile_means": torch.bfloat16,
}


def validate_vc_params(
    params: VCAttentionParams,
    expected_shapes: dict[str, tuple[int, ...]],
    *,
    device: torch.device,
) -> None:
    """Validate the operands of one run against the plan's expected shapes."""
    if not isinstance(params, VCAttentionParams):
        raise TypeError("vc must be a VCAttentionParams instance")
    if not isinstance(params.demean, bool):
        raise TypeError("vc.demean must be a bool")
    for name, shape in expected_shapes.items():
        tensor = getattr(params, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"vc.{name} must be a torch.Tensor")
        if tensor.device != device:
            raise ValueError(f"vc.{name} must be on {device}, got {tensor.device}")
        if tensor.dtype != _VC_PARAM_DTYPES[name]:
            raise TypeError(
                f"vc.{name} must have dtype {_VC_PARAM_DTYPES[name]}, got {tensor.dtype}"
            )
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"vc.{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"vc.{name} must be contiguous")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"vc.{name} must be 16-byte aligned")


def flat_block_scales(
    scale: torch.Tensor, seq_len: int, block_size: int
) -> torch.Tensor:
    """Pack ``[B, H, ceil(S / blk)]`` block scales into the flat ``[H, numel]`` layout."""
    b, h, nb = scale.shape
    out = torch.ones(
        (h, flat_scale_numel(b, seq_len, block_size)),
        dtype=torch.float32,
        device=scale.device,
    )
    lb = log2_block_size(block_size)
    for bi in range(b):
        base = int(flat_scale_slot(bi, 0, seq_len, lb))
        out[:, base : base + nb] = scale[bi]
    return out


def block_scales_from_flat(
    flat: torch.Tensor, batch_size: int, seq_len: int, block_size: int
) -> torch.Tensor:
    """Inverse of :func:`flat_block_scales`: ``[H, numel]`` -> ``[B, H, ceil(S / blk)]``."""
    nb = _blocks(seq_len, block_size)
    lb = log2_block_size(block_size)
    return torch.stack(
        [
            flat[:, int(flat_scale_slot(bi, 0, seq_len, lb)) :][:, :nb]
            for bi in range(batch_size)
        ]
    )


@dataclass(frozen=True)
class VCAttentionOperands:
    """Everything one VC-Attention forward needs, in kernel layout.

    ``q``, ``k``, ``v`` are E4M3 ``[B, S, H, D]``; ``k`` and ``v`` are stored in
    permuted token order. ``v`` holds tile residuals ``(V[perm] - mean) / v_scale``.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: (
        torch.Tensor
    )  # [H, flat_scale_numel(B, S_q, q_block)] fp32 (sage flat layout)
    k_scale: torch.Tensor  # [H, flat_scale_numel(B, S_k, 128)] fp32 (sage flat layout)
    v_scale: (
        torch.Tensor
    )  # [B, H, D] fp32 per-channel E4M3 residual scale (feeds output_scale)
    mean: torch.Tensor  # [B, H, num_kv_tiles, D] fp32 tile means (permuted order)
    mu: torch.Tensor  # [B, H, num_kv_tiles, 8, 256] bf16 packed kernel operand
    perm: torch.Tensor  # [B, H, S_k] int64 token permutation
    q_block_size: int

    @property
    def params(self) -> "VCAttentionParams":
        """The per-run operands :meth:`BatchPrefillTSWrapper.run` takes as ``vc``."""
        return VCAttentionParams(
            q_scale=self.q_scale,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            tile_means=self.mu,
        )


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


def _per_token_scales(scale: torch.Tensor, s: int, block: int) -> torch.Tensor:
    """Expand ``[B, H, nb]`` block scales to ``[B, S, H]``."""
    return scale.permute(0, 2, 1).repeat_interleave(block, dim=1)[:, :s]


def _quantize_blocks(x: torch.Tensor, scale: torch.Tensor, block: int) -> torch.Tensor:
    """Divide ``[B, S, H, D]`` by its ``[B, H, nb]`` block scale and round to E4M3."""
    b, s, h, d = x.shape
    per_token = _per_token_scales(scale, s, block)  # [B, S, H]
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
        q_scale=flat_block_scales(q_scale, q.shape[1], q_block_size),
        k_scale=flat_block_scales(k_scale, s_k, KV_TILE),
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
    q_scale = block_scales_from_flat(ops.q_scale, b, s_q, ops.q_block_size)
    k_scale = block_scales_from_flat(ops.k_scale, b, s_k, KV_TILE)
    q = ops.q.float() * _per_token_scales(q_scale, s_q, ops.q_block_size).unsqueeze(-1)
    k = ops.k.float() * _per_token_scales(k_scale, s_k, KV_TILE).unsqueeze(-1)
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
    "VC_K_BLOCK_SIZE",
    "VCAttentionConfig",
    "VCAttentionOperands",
    "VCAttentionParams",
    "block_scales_from_flat",
    "flat_block_scales",
    "validate_vc_params",
    "vc_scale_shapes",
    "hadamard_matrix",
    "vc_token_permutation_with_centroids",
    "pack_vc_tile_means",
    "vc_quantize",
    "vc_quantize_fused",
    "vc_reference",
    "vc_token_permutation",
]

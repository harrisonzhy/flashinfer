"""VC-Attention operand preparation kernels.

``_prepare`` turns bf16 or fp16 ``[B, S, H, D]`` K, V and under QK8 Q into the
VC-Attention operands with one CTA per batch, head and 128-token tile. Shapes
are compile-time constants of the kernels and the launches take tensors only.
An operand a recipe does not use is ``None``, and the kernels take their mode
from that.

* ``VcKvPass`` writes the V tile means, the permuted K rows under QK16 and the
  V residual amax of the channel statistics. Given the statistics of a previous
  call it also writes the V scale, the E4M3 V residuals and the tile-mean
  operand, the residual energies under repair, in the same pass. Without them
  the pass runs twice, statistics first.
* ``VcQKPass`` writes the centred, Hadamard-rotated E4M3 Q and K with one scale
  per block under QK8, and the K channel sums of the statistics.
* ``VcSelectPass`` and ``VcRepairRowsPass`` pick the tokens of largest residual
  energy and write their repair rows under repair.

Mean recipes take tokens in ``perm`` order. Repair recipes keep input order and
shift the partial last tile past the repair tiles. Each thread owns 8 columns of
8 rows and issues all its loads before the guarded math, since the kernels are
latency-bound and need the occupancy.
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from ..vc_attention import (
    E4M3_MAX,
    VC_K_BLOCK_SIZE,
    VC_MEAN_GROUP_TILES,
    VC_MEAN_MMA_K,
    VC_MEAN_OPERANDS,
    vc_mean_group_shape,
    vc_mean_operand_shape,
    vc_repair_kv_len,
)
from .fmha_decode.fmha_decode_resources.helpers_common import _pack_float2_to_bf16
from .fmha_decode.fmha_decode_resources.helpers_softmax import _pack_float4_to_fp8_e4m3

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
_D = 128
_THREADS = 256
_TILE = VC_K_BLOCK_SIZE
_COLS = 8
_COL_GROUPS = _D // _COLS
_ROW_GROUPS = _THREADS // _COL_GROUPS
_ROWS = _TILE // _ROW_GROUPS
_WORDS = _COLS // 2
_MU_ELEMS = math.prod(vc_mean_operand_shape(_D))
_HADAMARD_NORM = 1.0 / math.sqrt(_D)
_SELECT_BINS = 256
assert _THREADS == _SELECT_BINS, "VcSelectPass owns one histogram bin per thread"


@cute.jit
def _abs_f32(x: Float32) -> Float32:
    return cute.arch.fmax(x, -x)


@cute.jit
def _hadamard128_row(vals, base: cutlass.Constexpr[int], cg: Int32):
    """Unnormalised 128-point Walsh-Hadamard transform of one row in place.

    This thread holds 8 columns and the 15 lanes of its column group hold the
    rest. The caller folds the normalisation into its scale. The butterflies run
    as packed f32x2 pairs because the pass is issue-bound.
    """
    for h in (1, 2, 4):
        lo = [i for i in range(_COLS) if (i & h) == 0]
        for q in cutlass.range_constexpr(0, len(lo), 2):
            i0, i1 = lo[q], lo[q + 1]
            a = (vals[base + i0], vals[base + i1])
            b = (vals[base + (i0 ^ h)], vals[base + (i1 ^ h)])
            vals[base + i0], vals[base + i1] = cute.arch.add_packed_f32x2(a, b)
            vals[base + (i0 ^ h)], vals[base + (i1 ^ h)] = cute.arch.sub_packed_f32x2(
                a, b
            )
    for sbit in cutlass.range_constexpr(4):
        stride = 1 << sbit
        sign = Float32(1.0) - Float32(2.0) * Float32((cg >> sbit) & 1)
        for j in cutlass.range_constexpr(0, _COLS, 2):
            m0 = vals[base + j]
            m1 = vals[base + j + 1]
            o0 = cute.arch.shuffle_sync_bfly(m0, stride)
            o1 = cute.arch.shuffle_sync_bfly(m1, stride)
            vals[base + j], vals[base + j + 1] = cute.arch.fma_packed_f32x2(
                (m0, m1), (sign, sign), (o0, o1)
            )


@cute.jit
def _block_max(v: Float32, red_smem, tidx: Int32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, 1 << shift))
    if tidx % 32 == 0:
        red_smem[tidx // 32] = v
    cute.arch.sync_threads()
    r = red_smem[0]
    for w in cutlass.range_constexpr(1, _THREADS // 32):
        r = cute.arch.fmax(r, red_smem[w])
    cute.arch.sync_threads()
    return r


@cute.jit
def _load_raw(raw, base: cutlass.Constexpr[int], addr: Int64):
    """Load this thread's 16 bytes at ``addr`` into four words of ``raw``."""
    regs = cutlass.inttoptr(addr, mem_space=1, dtype=Int32).load(
        count=_WORDS, alignment=16
    )
    for q in cutlass.range_constexpr(_WORDS):
        raw[base + q] = regs[q]


@cute.jit
def _row(raw, base: cutlass.Constexpr[int], is_bf16: cutlass.Constexpr[bool]):
    """The 8 bf16 or fp16 values held in four words of ``raw`` as Float32."""
    regs = (raw.data_ptr() + base).load(count=_WORDS, alignment=16)
    if cutlass.const_expr(is_bf16):
        return regs.bitcast(cutlass.BFloat16).to(Float32)
    return regs.bitcast(cutlass.Float16).to(Float32)


@cute.jit
def _load_row_f32(addr: Int64):
    return cutlass.inttoptr(addr, mem_space=1, dtype=Float32).load(
        count=_COLS, alignment=32
    )


@cute.jit
def _store_row_fp8(addr: Int64, vals, base: cutlass.Constexpr[int], scale_inv: Float32):
    """Quantize this thread's 8 values by ``scale_inv`` and store 8 E4M3 bytes."""
    packed = cutlass.Array(Int32, _COLS // 4, space=cutlass.AddressSpace.rmem)
    for j in cutlass.range_constexpr(_COLS // 4):
        packed[j] = _pack_float4_to_fp8_e4m3(
            vals[base + 4 * j] * scale_inv,
            vals[base + 4 * j + 1] * scale_inv,
            vals[base + 4 * j + 2] * scale_inv,
            vals[base + 4 * j + 3] * scale_inv,
        )
    cutlass.inttoptr(addr, mem_space=1, dtype=Int32).store(
        packed.data_ptr().load(count=_COLS // 4, alignment=8), alignment=8
    )


@cute.jit
def _cta_col_reduce(
    col_sums, vals, rg: Int32, col0: Int32, is_max: cutlass.Constexpr[bool]
):
    """Reduce this thread's 8 column values over the row groups through shared memory."""
    cute.arch.sync_threads()
    for j in cutlass.range_constexpr(_COLS):
        col_sums[rg * _TILE + col0 + j] = vals[j]
    cute.arch.sync_threads()
    for j in cutlass.range_constexpr(_COLS):
        acc = col_sums[col0 + j]
        for g in cutlass.range_constexpr(1, _ROW_GROUPS):
            if cutlass.const_expr(is_max):
                acc = cute.arch.fmax(acc, col_sums[g * _TILE + col0 + j])
            else:
                acc = acc + col_sums[g * _TILE + col0 + j]
        vals[j] = acc


@cute.jit
def _v_scale(vmax: Float32) -> Float32:
    return cute.arch.fmax(vmax * Float32(1.0 / E4M3_MAX), Float32(1e-12))


@cute.jit
def _stats_addr(
    mStats: cute.Tensor, bh: Int32, slot: cutlass.Constexpr[int], col0: Int32
) -> Int64:
    """Slot 0 holds the K channel sums and slot 1 the V residual amax."""
    return mStats.iterator.toint() + ((Int64(bh) * 2 + slot) * _D + col0) * 4


@cute.jit
def _scale_slot(
    bh: Int32,
    block: Int32,
    H: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    S: cutlass.Constexpr[int],
) -> Int64:
    """Element of block ``block`` of sequence ``bh`` in the flat scale layout."""
    numel = (B * S + _TILE - 1) // _TILE + B - 1
    return Int64(bh % H) * numel + Int64(((bh // H) * S) // _TILE + bh // H + block)


class _Shape:
    """Compile-time geometry shared by the kernels of one preparation."""

    def __init__(self, is_bf16: bool, B: int, S: int, H: int, repair_tiles: int):
        self.is_bf16 = is_bf16
        self.B = B
        self.S = S
        self.H = H
        self.T = (S + _TILE - 1) // _TILE
        self.R = repair_tiles
        self.repair = repair_tiles > 0
        self.rows = vc_repair_kv_len(S, repair_tiles) if self.repair else S
        self.tail = S // _TILE * _TILE if self.repair else S
        self.shift = repair_tiles * _TILE
        self.row_bytes = H * _D * 2
        self.row_bytes8 = H * _D


@cute.jit
def _write_v_outputs(
    shape,
    t: Int32,
    bh: Int32,
    b: Int32,
    h: Int32,
    tidx: Int32,
    rg: Int32,
    cg: Int32,
    col0: Int32,
    raw,
    mean,
    mean_all,
    energy_part,
    vmax_base: Int64,
    mVScale: cute.Tensor,
    mV8: cute.Tensor,
    mMu: cute.Tensor | None,
    mEnergy: cute.Tensor | None,
):
    """Write this tile's V outputs from the rows in ``raw``.

    The residual scale comes from the amax at ``vmax_base``. Tile 0 stores the
    scale. Each row becomes E4M3 residuals, with its squared rounding error
    under repair. The packed mean operand reads the tile means from shared
    memory, which the caller fills before the call.
    """
    vmax = _load_row_f32(vmax_base + col0 * 4)
    vs = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
    vs_inv = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
    for j in cutlass.range_constexpr(_COLS):
        vs[j] = _v_scale(vmax[j])
        vs_inv[j] = Float32(1.0) / vs[j]
    if (t == 0) & (rg == 0):
        cutlass.inttoptr(
            mVScale.iterator.toint() + (Int64(bh) * _D + col0) * 4,
            mem_space=1,
            dtype=Float32,
        ).store(vs.data_ptr().load(count=_COLS, alignment=32), alignment=32)
    for i in cutlass.range_constexpr(_ROWS):
        pos = t * _TILE + rg + _ROW_GROUPS * i
        energy = Float32(0.0)
        if pos < shape.S:
            row = _row(raw, i * _WORDS, shape.is_bf16)
            res = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
            for j in cutlass.range_constexpr(_COLS):
                res[j] = (row[j] - mean[j]) * vs_inv[j]
            out_row = pos + shape.shift * Int32(pos >= shape.tail)
            out_addr = (
                mV8.iterator.toint()
                + (Int64(b) * shape.rows + out_row) * shape.row_bytes8
                + Int64(h) * _D
                + col0
            )
            if cutlass.const_expr(mEnergy is not None):
                # The squared E4M3 rounding error in value units ranks the
                # tokens for repair.
                packed = cutlass.Array(
                    Int32, _COLS // 4, space=cutlass.AddressSpace.rmem
                )
                for j in cutlass.range_constexpr(_COLS // 4):
                    packed[j] = _pack_float4_to_fp8_e4m3(
                        res[4 * j], res[4 * j + 1], res[4 * j + 2], res[4 * j + 3]
                    )
                words = packed.data_ptr().load(count=_COLS // 4, alignment=8)
                codes = words.bitcast(cutlass.Float8E4M3FN).to(Float32)
                for c in cutlass.range_constexpr(_COLS):
                    err = (res[c] - codes[c]) * vs[c]
                    energy = energy + err * err
                cutlass.inttoptr(out_addr, mem_space=1, dtype=Int32).store(
                    words, alignment=8
                )
            else:
                _store_row_fp8(out_addr, res, 0, Float32(1.0))
        if cutlass.const_expr(mEnergy is not None):
            energy_part[(rg + _ROW_GROUPS * i) * _COL_GROUPS + cg] = energy
    if cutlass.const_expr(mEnergy is not None):
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(_ROWS):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            if (cg == 0) & (pos < shape.S):
                acc = Float32(0.0)
                for c in cutlass.range_constexpr(_COL_GROUPS):
                    acc = acc + energy_part[(rg + _ROW_GROUPS * i) * _COL_GROUPS + c]
                cutlass.inttoptr(
                    mEnergy.iterator.toint() + (Int64(bh) * shape.S + pos) * 4,
                    mem_space=1,
                    dtype=Float32,
                ).store(acc)

    if cutlass.const_expr(mMu is not None):
        # Mean operand of this tile's group in core-matrix order. Tile 8o+i
        # of a group owns K slots 2i and 2i+1 of operand o. The host zeroes
        # the operands and each of the first 128 threads stores its two
        # rows' slots. Means are stored divided by the per-channel scale.
        k_per_core = VC_MEAN_MMA_K // 2
        idx0 = tidx * (2 * k_per_core)
        g = idx0 // (8 * VC_MEAN_MMA_K)
        c = (idx0 // (8 * k_per_core)) % 2
        d0 = g * 8 + (idx0 // k_per_core) % 8
        tile_in_group = t % VC_MEAN_GROUP_TILES
        operand = tile_in_group // k_per_core
        slot = 2 * (tile_in_group % k_per_core)
        num_groups = (shape.T + VC_MEAN_GROUP_TILES - 1) // VC_MEAN_GROUP_TILES
        if (tidx < 128) & (c == slot // k_per_core):
            m0 = mean_all[d0]
            m1 = mean_all[d0 + 1]
            s0 = _v_scale(
                cutlass.inttoptr(
                    vmax_base + Int64(d0) * 4, mem_space=1, dtype=Float32
                ).load()
            )
            s1 = _v_scale(
                cutlass.inttoptr(
                    vmax_base + Int64(d0 + 1) * 4, mem_space=1, dtype=Float32
                ).load()
            )
            dst = (
                mMu.iterator.toint()
                + (
                    (
                        (Int64(bh) * num_groups + t // VC_MEAN_GROUP_TILES)
                        * VC_MEAN_OPERANDS
                        + operand
                    )
                    * _MU_ELEMS
                    + idx0
                    + slot % k_per_core
                )
                * 2
            )
            cutlass.inttoptr(dst, mem_space=1, dtype=Int32).store(
                _pack_float2_to_bf16(m0 / s0, m0 / s0)
            )
            cutlass.inttoptr(dst + k_per_core * 2, mem_space=1, dtype=Int32).store(
                _pack_float2_to_bf16(m1 / s1, m1 / s1)
            )


class VcKvPass(_Shape):
    """One pass over the V tiles, and the K tiles when they are permuted here.

    The pass writes the V tile means and with ``mKPerm`` the permuted K rows.
    With ``mStats`` it accumulates the V residual amax of this call with
    atomics. With ``mPrev`` it takes the statistics of a previous call for the
    V scale and writes the E4M3 residuals and the packed mean operand from the
    rows already in registers. The exact preparation runs the pass twice,
    statistics first.
    """

    def __init__(self, is_bf16, B, S, H, repair_tiles, demean: bool):
        super().__init__(is_bf16, B, S, H, repair_tiles)
        self.demean = demean

    @cute.kernel
    def kernel(
        self,
        mK: cute.Tensor | None,
        mV: cute.Tensor,
        mPerm: cute.Tensor | None,
        mKPerm: cute.Tensor | None,
        mMean: cute.Tensor,
        mStats: cute.Tensor | None,
        mPrev: cute.Tensor | None,
        mVScale: cute.Tensor | None,
        mV8: cute.Tensor | None,
        mMu: cute.Tensor | None,
        mEnergy: cute.Tensor | None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        t, bh, _ = cute.arch.block_idx()
        b = bh // self.H
        h = bh % self.H
        cg = tidx % _COL_GROUPS
        rg = tidx // _COL_GROUPS
        col0 = cg * _COLS
        smem = cutlass.utils.SmemAllocator()
        col_sums = smem.allocate_array(Float32, _ROW_GROUPS * _TILE)
        mean_all = smem.allocate_array(Float32, _D)
        energy_part = smem.allocate_array(Float32, _THREADS * _ROWS)
        v_base = (
            mV.iterator.toint()
            + Int64(b) * self.S * self.row_bytes
            + (Int64(h) * _D + col0) * 2
        )

        # Rows past the sequence load the last token and skip the math, so no
        # load carries a predicate and the loads of a thread issue together.
        toks = cutlass.Array(Int32, _ROWS, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            if pos >= self.S:
                pos = Int32(self.S - 1)
            tok = pos
            if cutlass.const_expr(mPerm is not None):
                tok = cutlass.inttoptr(
                    mPerm.iterator.toint() + (Int64(bh) * self.S + pos) * 4,
                    mem_space=1,
                    dtype=Int32,
                ).load()
            toks[i] = tok
        raw = cutlass.Array(Int32, _ROWS * _WORDS, space=cutlass.AddressSpace.rmem)

        if cutlass.const_expr(mKPerm is not None):
            k_base = (
                mK.iterator.toint()
                + Int64(b) * self.S * self.row_bytes
                + (Int64(h) * _D + col0) * 2
            )
            for i in cutlass.range_constexpr(_ROWS):
                _load_raw(raw, i * _WORDS, k_base + Int64(toks[i]) * self.row_bytes)
            for i in cutlass.range_constexpr(_ROWS):
                pos = t * _TILE + rg + _ROW_GROUPS * i
                if pos < self.S:
                    out_row = pos + self.shift * Int32(pos >= self.tail)
                    cutlass.inttoptr(
                        mKPerm.iterator.toint()
                        + (Int64(b) * self.rows + out_row) * self.row_bytes
                        + (Int64(h) * _D + col0) * 2,
                        mem_space=1,
                        dtype=Int32,
                    ).store(
                        (raw.data_ptr() + i * _WORDS).load(count=_WORDS, alignment=16),
                        alignment=16,
                    )

        # Per-column sum, max and min of the tile's V rows give the mean and the
        # residual amax exactly.
        psum = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        pmax = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        pmin = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS):
            psum[j] = Float32(0.0)
            pmax[j] = -Float32.inf
            pmin[j] = Float32.inf
        for i in cutlass.range_constexpr(_ROWS):
            _load_raw(raw, i * _WORDS, v_base + Int64(toks[i]) * self.row_bytes)
        for i in cutlass.range_constexpr(_ROWS):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            if pos < self.S:
                row = _row(raw, i * _WORDS, self.is_bf16)
                for j in cutlass.range_constexpr(_COLS):
                    psum[j] = psum[j] + row[j]
                    pmax[j] = cute.arch.fmax(pmax[j], row[j])
                    pmin[j] = cute.arch.fmin(pmin[j], row[j])
        valid_rows = self.S - t * _TILE
        if valid_rows > _TILE:
            valid_rows = Int32(_TILE)
        inv_count = Float32(1.0) / Float32(valid_rows)
        _cta_col_reduce(col_sums, psum, rg, col0, False)
        mean = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS):
            mean[j] = psum[j] * inv_count
            if cutlass.const_expr(not self.demean):
                mean[j] = Float32(0.0)
        if rg == 0:
            cutlass.inttoptr(
                mMean.iterator.toint() + ((Int64(bh) * self.T + t) * _D + col0) * 4,
                mem_space=1,
                dtype=Float32,
            ).store(mean.data_ptr().load(count=_COLS, alignment=32), alignment=32)
        cmax = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS):
            cmax[j] = Float32(0.0)
            if pmax[j] >= pmin[j]:
                cmax[j] = cute.arch.fmax(pmax[j] - mean[j], mean[j] - pmin[j])
        _cta_col_reduce(col_sums, cmax, rg, col0, True)
        if cutlass.const_expr(mStats is not None):
            if rg == 0:
                for j in cutlass.range_constexpr(_COLS):
                    cute.arch.atomic_fmax(
                        cutlass.inttoptr(
                            _stats_addr(mStats, bh, 1, col0) + j * 4,
                            mem_space=1,
                            dtype=Float32,
                        ),
                        cmax[j],
                        sign_bit=False,
                    )
        if cutlass.const_expr(mPrev is not None):
            if rg == 0:
                for j in cutlass.range_constexpr(_COLS):
                    mean_all[col0 + j] = mean[j]
            cute.arch.sync_threads()
            _write_v_outputs(
                self,
                t,
                bh,
                b,
                h,
                tidx,
                rg,
                cg,
                col0,
                raw,
                mean,
                mean_all,
                energy_part,
                _stats_addr(mPrev, bh, 1, 0),
                mVScale,
                mV8,
                mMu,
                mEnergy,
            )

    @cute.jit
    def __call__(
        self,
        mK,
        mV,
        mPerm,
        mKPerm,
        mMean,
        mStats,
        mPrev,
        mVScale,
        mV8,
        mMu,
        mEnergy,
        stream,
    ):
        self.kernel(
            mK, mV, mPerm, mKPerm, mMean, mStats, mPrev, mVScale, mV8, mMu, mEnergy
        ).launch(
            grid=[self.T, self.B * self.H, 1],
            block=[_THREADS, 1, 1],
            smem=(_ROW_GROUPS * _TILE + _D + _THREADS * _ROWS) * 4,
            stream=stream,
        )


class VcQKPass(_Shape):
    """QK8 quantization of Q or K, one CTA per 128-token block.

    ``kind`` is ``q`` for Q, ``k`` for K through ``perm`` or the repair row map,
    and ``k_repair`` for the repair tiles through the selection index. K is
    centred with the channel sums of ``mStats`` and the ``k`` pass accumulates
    the sums of this call into ``mSums`` with atomics. Without ``mX8`` the pass
    only accumulates. The rotated block stays in registers between the amax
    and the store.
    """

    def __init__(
        self, is_bf16, B, S, H, repair_tiles, kind: str, SQ: int = 0, SEL: int = 0
    ):
        super().__init__(is_bf16, B, S, H, repair_tiles)
        self.kind = kind
        self.SEL = SEL
        if kind == "q":
            self.rows_out = self.src = SQ
        else:
            self.rows_out, self.src = self.rows, S
        self.block0 = self.tail // _TILE if kind == "k_repair" else 0
        self.blocks = (
            self.R if kind == "k_repair" else (self.rows_out + _TILE - 1) // _TILE
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mPerm: cute.Tensor | None,
        mIndex: cute.Tensor | None,
        mStats: cute.Tensor | None,
        mSums: cute.Tensor | None,
        mX8: cute.Tensor | None,
        mScale: cute.Tensor | None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        nb, bh, _ = cute.arch.block_idx()
        nb = nb + self.block0
        b = bh // self.H
        h = bh % self.H
        cg = tidx % _COL_GROUPS
        rg = tidx // _COL_GROUPS
        col0 = cg * _COLS
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_array(Float32, _THREADS // 32)
        x_base = (
            mX.iterator.toint()
            + Int64(b) * self.src * self.row_bytes
            + (Int64(h) * _D + col0) * 2
        )
        centre = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS):
            centre[j] = Float32(0.0)
        if cutlass.const_expr(mStats is not None):
            c = _load_row_f32(_stats_addr(mStats, bh, 0, col0))
            for j in cutlass.range_constexpr(_COLS):
                centre[j] = c[j] * Float32(1.0 / self.src)

        # Source token per row, with -1 for rows outside the sequence and for
        # zero rows. Loads read token 0 for those and the math is skipped.
        toks = cutlass.Array(Int32, _ROWS, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS):
            pos = nb * _TILE + rg + _ROW_GROUPS * i
            tok = Int32(-1)
            if pos < self.rows_out:
                tok = pos
                if cutlass.const_expr(mIndex is not None):
                    tok = Int32(-1)
                    sel = pos - self.tail
                    if (sel >= 0) & (sel < self.SEL):
                        tok = cutlass.inttoptr(
                            mIndex.iterator.toint() + (Int64(bh) * self.SEL + sel) * 4,
                            mem_space=1,
                            dtype=Int32,
                        ).load()
                elif cutlass.const_expr(mPerm is not None):
                    tok = cutlass.inttoptr(
                        mPerm.iterator.toint() + (Int64(bh) * self.S + pos) * 4,
                        mem_space=1,
                        dtype=Int32,
                    ).load()
                elif cutlass.const_expr(self.kind == "k" and self.repair):
                    if pos >= self.tail:
                        tok = Int32(-1)
                        if (pos >= self.tail + self.shift) & (
                            pos < self.S + self.shift
                        ):
                            tok = pos - self.shift
            toks[i] = tok
        raw = cutlass.Array(Int32, _ROWS * _WORDS, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS):
            tok = toks[i]
            if tok < 0:
                tok = Int32(0)
            _load_raw(raw, i * _WORDS, x_base + Int64(tok) * self.row_bytes)
        if cutlass.const_expr(mSums is not None):
            col_sums = smem.allocate_array(Float32, _ROW_GROUPS * _TILE)
            ksum = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
            for j in cutlass.range_constexpr(_COLS):
                ksum[j] = Float32(0.0)
            for i in cutlass.range_constexpr(_ROWS):
                if toks[i] >= 0:
                    row = _row(raw, i * _WORDS, self.is_bf16)
                    for j in cutlass.range_constexpr(_COLS):
                        ksum[j] = ksum[j] + row[j]
            _cta_col_reduce(col_sums, ksum, rg, col0, False)
            if rg == 0:
                for j in cutlass.range_constexpr(_COLS):
                    cute.arch.atomic_add(
                        cutlass.inttoptr(
                            _stats_addr(mSums, bh, 0, col0) + j * 4,
                            mem_space=1,
                            dtype=Float32,
                        ),
                        ksum[j],
                    )
        if cutlass.const_expr(mX8 is None):
            return
        vals = cutlass.Array(Float32, _ROWS * _COLS, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS):
            if toks[i] >= 0:
                row = _row(raw, i * _WORDS, self.is_bf16)
                for j in cutlass.range_constexpr(_COLS):
                    vals[i * _COLS + j] = row[j] - centre[j]
            else:
                # The rotation shuffles need all 16 lanes of a row.
                for j in cutlass.range_constexpr(_COLS):
                    vals[i * _COLS + j] = Float32(0.0)
        for i in cutlass.range_constexpr(_ROWS):
            _hadamard128_row(vals, i * _COLS, cg)
        amax = Float32(0.0)
        for i in cutlass.range_constexpr(_ROWS):
            pos = nb * _TILE + rg + _ROW_GROUPS * i
            if pos < self.rows_out:
                for j in cutlass.range_constexpr(_COLS):
                    amax = cute.arch.fmax(amax, _abs_f32(vals[i * _COLS + j]))
        amax = _block_max(amax, red, tidx)
        scale = cute.arch.fmax(
            amax * Float32(_HADAMARD_NORM / E4M3_MAX), Float32(1e-12)
        )
        if tidx == 0:
            cutlass.inttoptr(
                mScale.iterator.toint()
                + _scale_slot(bh, nb, self.H, self.B, self.rows_out) * 4,
                mem_space=1,
                dtype=Float32,
            ).store(scale)
        scale_inv = Float32(_HADAMARD_NORM) / scale
        x8_base = (
            mX8.iterator.toint()
            + Int64(b) * self.rows_out * self.row_bytes8
            + Int64(h) * _D
            + col0
        )
        for i in cutlass.range_constexpr(_ROWS):
            pos = nb * _TILE + rg + _ROW_GROUPS * i
            if pos < self.rows_out:
                _store_row_fp8(
                    x8_base + Int64(pos) * self.row_bytes8, vals, i * _COLS, scale_inv
                )

    @cute.jit
    def __call__(self, mX, mPerm, mIndex, mStats, mSums, mX8, mScale, stream):
        smem = (_THREADS // 32) * 4
        if cutlass.const_expr(mSums is not None):
            smem = smem + _ROW_GROUPS * _TILE * 4
        self.kernel(mX, mPerm, mIndex, mStats, mSums, mX8, mScale).launch(
            grid=[self.blocks, self.B * self.H, 1],
            block=[_THREADS, 1, 1],
            smem=smem,
            stream=stream,
        )


class VcSelectPass(_Shape):
    """The ``SEL`` tokens of largest residual energy per sequence.

    A radix select over the four bytes of the non-negative fp32 bits with one
    histogram per warp, then the indices above the threshold and the first
    ones at it, in arbitrary order.
    """

    def __init__(self, is_bf16, B, S, H, repair_tiles, SEL: int):
        super().__init__(is_bf16, B, S, H, repair_tiles)
        self.SEL = SEL

    @cute.kernel
    def kernel(self, mEnergy: cute.Tensor, mIndex: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bh, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        smem = cutlass.utils.SmemAllocator()
        hist = smem.allocate_array(Int32, (_THREADS // 32) * _SELECT_BINS)
        warp_tot = smem.allocate_array(Int32, _THREADS // 32)
        # The threshold prefix, the count still wanted at it, the count stored
        # above it and the count seen at it.
        state = smem.allocate_array(Int32, 4)
        e_base = mEnergy.iterator.toint() + Int64(bh) * self.S * 4
        out_base = mIndex.iterator.toint() + Int64(bh) * self.SEL * 4
        # One CTA streams the whole row, so four loads per thread and
        # iteration set the memory parallelism.
        chunks = (self.S + 4 * _THREADS - 1) // (4 * _THREADS)
        if tidx == 0:
            state[0] = Int32(0)
            state[1] = Int32(self.SEL)
            state[2] = Int32(0)
            state[3] = Int32(0)
        cute.arch.sync_threads()
        for p in cutlass.range_constexpr(4):
            shift = 24 - 8 * p
            for w in cutlass.range_constexpr(_THREADS // 32):
                hist[w * _SELECT_BINS + tidx] = Int32(0)
            prefix = state[0]
            rem = state[1]
            cute.arch.sync_threads()
            for it in cutlass.range(chunks):
                vals = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
                for u in cutlass.range_constexpr(4):
                    i = tidx + (it * 4 + u) * _THREADS
                    vals[u] = Int32(-1)
                    if i < self.S:
                        vals[u] = cutlass.inttoptr(
                            e_base + Int64(i) * 4, mem_space=1, dtype=Int32
                        ).load()
                for u in cutlass.range_constexpr(4):
                    bits = vals[u]
                    hit = bits >= 0
                    if cutlass.const_expr(p > 0):
                        hit = hit & ((bits >> (shift + 8)) == prefix)
                    if hit:
                        cute.arch.atomic_add(
                            hist
                            + (
                                warp * _SELECT_BINS
                                + ((bits >> shift) & (_SELECT_BINS - 1))
                            ),
                            Int32(1),
                        )
            cute.arch.sync_threads()
            # Thread tidx owns bin 255 - tidx and the scan runs in descending bin order.
            mybin = _SELECT_BINS - 1 - tidx
            cnt = Int32(0)
            for w in cutlass.range_constexpr(_THREADS // 32):
                cnt = cnt + hist[w * _SELECT_BINS + mybin]
            incl = cnt
            for sbit in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_up(incl, 1 << sbit)
                if lane >= (1 << sbit):
                    incl = incl + other
            if lane == 31:
                warp_tot[warp] = incl
            cute.arch.sync_threads()
            excl = incl - cnt
            for w in cutlass.range_constexpr(_THREADS // 32):
                if w < warp:
                    excl = excl + warp_tot[w]
            if (excl < rem) & (rem <= excl + cnt):
                state[0] = (prefix << 8) | mybin
                state[1] = rem - excl
            cute.arch.sync_threads()
        threshold = state[0]
        rem = state[1]
        n_above = self.SEL - rem
        for it in cutlass.range(chunks):
            vals = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
            for u in cutlass.range_constexpr(4):
                i = tidx + (it * 4 + u) * _THREADS
                vals[u] = Int32(-1)
                if i < self.S:
                    vals[u] = cutlass.inttoptr(
                        e_base + Int64(i) * 4, mem_space=1, dtype=Int32
                    ).load()
            for u in cutlass.range_constexpr(4):
                i = tidx + (it * 4 + u) * _THREADS
                bits = vals[u]
                if bits > threshold:
                    slot = cute.arch.atomic_add(state + 2, Int32(1))
                    cutlass.inttoptr(
                        out_base + Int64(slot) * 4, mem_space=1, dtype=Int32
                    ).store(i)
                elif bits == threshold:
                    slot = cute.arch.atomic_add(state + 3, Int32(1))
                    if slot < rem:
                        cutlass.inttoptr(
                            out_base + Int64(n_above + slot) * 4,
                            mem_space=1,
                            dtype=Int32,
                        ).store(i)

    @cute.jit
    def __call__(self, mEnergy, mIndex, stream):
        self.kernel(mEnergy, mIndex).launch(
            grid=[self.B * self.H, 1, 1],
            block=[_THREADS, 1, 1],
            smem=((_THREADS // 32) * _SELECT_BINS + _THREADS // 32 + 4) * 4,
            stream=stream,
        )


class VcRepairRowsPass(_Shape):
    """The repair tiles and the padding after the shifted tail.

    One CTA per repair tile writes the selected tokens' E4M3 value residuals
    and, when ``mKOut`` is given, their keys, with zero rows for unused slots.
    One more CTA zeroes the padding rows.
    """

    def __init__(self, is_bf16, B, S, H, repair_tiles, SEL: int):
        super().__init__(is_bf16, B, S, H, repair_tiles)
        self.SEL = SEL

    @cute.kernel
    def kernel(
        self,
        mK: cute.Tensor | None,
        mV: cute.Tensor,
        mIndex: cute.Tensor,
        mStats: cute.Tensor,
        mKOut: cute.Tensor | None,
        mV8: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        j, bh, _ = cute.arch.block_idx()
        b = bh // self.H
        h = bh % self.H
        cg = tidx % _COL_GROUPS
        rg = tidx // _COL_GROUPS
        col0 = cg * _COLS
        vmax = _load_row_f32(_stats_addr(mStats, bh, 1, col0))
        vs_inv = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        for c in cutlass.range_constexpr(_COLS):
            vs_inv[c] = Float32(1.0) / _v_scale(vmax[c])
        zeros = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
        for c in cutlass.range_constexpr(4):
            zeros[c] = Int32(0)
        raw = cutlass.Array(Int32, _WORDS, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS):
            rr = rg + _ROW_GROUPS * i
            tok = Int32(-1)
            out_row = self.S + self.shift + rr
            if j < self.R:
                out_row = self.tail + j * _TILE + rr
                sel = j * _TILE + rr
                if sel < self.SEL:
                    tok = cutlass.inttoptr(
                        mIndex.iterator.toint() + (Int64(bh) * self.SEL + sel) * 4,
                        mem_space=1,
                        dtype=Int32,
                    ).load()
            if out_row < self.rows:
                row_off = Int64(b) * self.rows + out_row
                v_dst = (
                    mV8.iterator.toint()
                    + row_off * self.row_bytes8
                    + Int64(h) * _D
                    + col0
                )
                if tok >= 0:
                    src_off = (Int64(b) * self.S + tok) * self.row_bytes + (
                        Int64(h) * _D + col0
                    ) * 2
                    _load_raw(raw, 0, mV.iterator.toint() + src_off)
                    row = _row(raw, 0, self.is_bf16)
                    # The residual of the E4M3 code pass 2 stored for this token.
                    res = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
                    for c in cutlass.range_constexpr(_COLS):
                        res[c] = row[c] * vs_inv[c]
                    packed = cutlass.Array(
                        Int32, _COLS // 4, space=cutlass.AddressSpace.rmem
                    )
                    for c in cutlass.range_constexpr(_COLS // 4):
                        packed[c] = _pack_float4_to_fp8_e4m3(
                            res[4 * c], res[4 * c + 1], res[4 * c + 2], res[4 * c + 3]
                        )
                    codes = (
                        packed.data_ptr()
                        .load(count=_COLS // 4, alignment=8)
                        .bitcast(cutlass.Float8E4M3FN)
                        .to(Float32)
                    )
                    for c in cutlass.range_constexpr(_COLS):
                        res[c] = res[c] - codes[c]
                    _store_row_fp8(v_dst, res, 0, Float32(1.0))
                    if cutlass.const_expr(mKOut is not None):
                        _load_raw(raw, 0, mK.iterator.toint() + src_off)
                        cutlass.inttoptr(
                            mKOut.iterator.toint()
                            + row_off * self.row_bytes
                            + (Int64(h) * _D + col0) * 2,
                            mem_space=1,
                            dtype=Int32,
                        ).store(
                            raw.data_ptr().load(count=_WORDS, alignment=16),
                            alignment=16,
                        )
                else:
                    cutlass.inttoptr(v_dst, mem_space=1, dtype=Int32).store(
                        zeros.data_ptr().load(count=2, alignment=8), alignment=8
                    )
                    if cutlass.const_expr(mKOut is not None):
                        cutlass.inttoptr(
                            mKOut.iterator.toint()
                            + row_off * self.row_bytes
                            + (Int64(h) * _D + col0) * 2,
                            mem_space=1,
                            dtype=Int32,
                        ).store(
                            zeros.data_ptr().load(count=4, alignment=16), alignment=16
                        )

    @cute.jit
    def __call__(self, mK, mV, mIndex, mStats, mKOut, mV8, stream):
        self.kernel(mK, mV, mIndex, mStats, mKOut, mV8).launch(
            grid=[self.R + 1, self.B * self.H, 1],
            block=[_THREADS, 1, 1],
            smem=0,
            stream=stream,
        )


def _fake1d(dtype, align=32):
    return cute.runtime.make_fake_compact_tensor(
        dtype, (cute.sym_int(),), assumed_align=align
    )


def _in_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16, True
    if dtype == torch.float16:
        return cutlass.Float16, False
    raise TypeError(f"VC fused preparation needs bf16/fp16 inputs, got {dtype}")


@functools.lru_cache(maxsize=None)
def _compiled(
    dtype: torch.dtype,
    demean: bool,
    B: int,
    S: int,
    H: int,
    repair_tiles: int,
    SQ: int,
    SEL: int,
):
    """The kernels of one geometry. ``SQ`` is 0 under QK16 and ``SEL`` is 0
    without repair. The K/V pass and the K pass come in three forms each.
    ``stats`` and ``ksum`` collect the statistics, ``kv`` and ``k`` write the
    outputs from given statistics and collect the next ones, ``kv_only`` and
    ``k_only`` write without collecting. Absent operands are ``None`` at
    compile time as at run time."""
    cdt, is_bf16 = _in_dtype(dtype)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    f32 = lambda: _fake1d(Float32, 64)  # noqa: E731
    i32 = lambda: _fake1d(Int32, 16)  # noqa: E731
    e4m3 = lambda: _fake1d(cutlass.Float8E4M3FN, 16)  # noqa: E731
    x16 = lambda: _fake1d(cdt)  # noqa: E731
    qk8, repair = SQ > 0, repair_tiles > 0
    shape = (is_bf16, B, S, H, repair_tiles)
    perm = None if repair else i32()
    kperm = None if qk8 else x16()
    mu = None if repair else _fake1d(cutlass.BFloat16)
    energy = f32() if repair else None

    def kv_pass(kperm, stats, prev, *outs):
        return cute.compile(
            VcKvPass(*shape, demean),
            x16(),
            x16(),
            perm,
            kperm,
            f32(),
            stats,
            prev,
            *outs,
            stream,
            options=_COMPILE_OPTIONS,
        )

    k = {
        "stats": kv_pass(None, f32(), None, None, None, None, None),
        "kv": kv_pass(kperm, f32(), f32(), f32(), e4m3(), mu, energy),
        "kv_only": kv_pass(kperm, None, f32(), f32(), e4m3(), mu, energy),
    }
    if repair:
        k["select"] = cute.compile(
            VcSelectPass(*shape, SEL), f32(), i32(), stream, options=_COMPILE_OPTIONS
        )
        k["rows"] = cute.compile(
            VcRepairRowsPass(*shape, SEL),
            None if qk8 else x16(),
            x16(),
            i32(),
            f32(),
            None if qk8 else x16(),
            e4m3(),
            stream,
            options=_COMPILE_OPTIONS,
        )
    if qk8:
        k["q"] = cute.compile(
            VcQKPass(*shape, "q", SQ=SQ),
            x16(),
            None,
            None,
            None,
            None,
            e4m3(),
            f32(),
            stream,
            options=_COMPILE_OPTIONS,
        )

        def k_pass(stats, sums, *outs):
            return cute.compile(
                VcQKPass(*shape, "k"),
                x16(),
                perm,
                None,
                stats,
                sums,
                *outs,
                stream,
                options=_COMPILE_OPTIONS,
            )

        k["ksum"] = k_pass(None, f32(), None, None)
        k["k"] = k_pass(f32(), f32(), e4m3(), f32())
        k["k_only"] = k_pass(f32(), None, e4m3(), f32())
        if repair:
            k["k_repair"] = cute.compile(
                VcQKPass(*shape, "k_repair", SEL=SEL),
                x16(),
                None,
                i32(),
                f32(),
                None,
                e4m3(),
                f32(),
                stream,
                options=_COMPILE_OPTIONS,
            )
    return k


@torch.no_grad()
def vc_prepare(
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor,
    *,
    demean: bool = True,
    prev_stats: torch.Tensor | None = None,
):
    """VC-Attention-QK16 operands ``(k_perm, v8, v_scale, mean, mu, stats)`` of
    contiguous ``[B, S, H, D]`` K and V in ``perm`` order.

    ``stats`` holds this call's channel statistics, the K channel sums under
    QK8 and the V residual amax, as ``[B, H, 2, D]`` fp32. Given the previous
    call's as ``prev_stats``, the V scale comes from them and the preparation
    runs in one pass over V.
    """
    out = _prepare(
        None,
        k,
        v,
        perm,
        demean=demean,
        repair_tiles=0,
        selected=0,
        prev_stats=prev_stats,
    )
    return out[:5] + out[8:]


@torch.no_grad()
def vc_prepare_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor,
    *,
    demean: bool = True,
    prev_stats: torch.Tensor | None = None,
):
    """VC-Attention-QK8 operands ``(q8, k8, v8, q_scale, k_scale, v_scale, mean,
    mu, stats)`` of contiguous ``[B, S, H, D]`` Q, K and V, K and V in ``perm``
    order. ``stats`` holds this call's K channel sums and V residual amax as
    ``[B, H, 2, D]`` fp32. Given the previous call's as ``prev_stats``, the K
    centre and the V scale come from them and each pass runs once."""
    if q.shape[-1] != _D or not q.is_contiguous():
        raise ValueError(f"VC fused preparation needs contiguous head_dim {_D} Q")
    k8, v8, v_scale, mean, mu, q8, q_scale, k_scale, stats = _prepare(
        q, k, v, perm, demean=demean, repair_tiles=0, selected=0, prev_stats=prev_stats
    )
    return q8, k8, v8, q_scale, k_scale, v_scale, mean, mu, stats


@torch.no_grad()
def vc_prepare_repair(
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    repair_tiles: int,
    selected: int,
    prev_stats: torch.Tensor | None = None,
):
    """V repair operands ``(q8, k_all, v_all, q_scale, k_scale, v_scale, stats)``
    in input order plus ``repair_tiles`` tiles of the ``selected``
    largest-residual tokens. The QK8 entries are ``None`` under QK16.
    ``stats`` holds this call's channel statistics as ``[B, H, 2, D]`` fp32
    and ``prev_stats`` of a previous call makes each pass run once."""
    if q is not None and (q.shape[-1] != _D or not q.is_contiguous()):
        raise ValueError(f"VC fused preparation needs contiguous head_dim {_D} Q")
    if not 0 < selected <= min(repair_tiles * _TILE, k.shape[1]):
        raise ValueError("selected must fit the repair tiles and the sequence")
    k_all, v_all, v_scale, _, _, q8, q_scale, k_scale, stats = _prepare(
        q,
        k,
        v,
        None,
        demean=False,
        repair_tiles=repair_tiles,
        selected=selected,
        prev_stats=prev_stats,
    )
    return q8, k_all, v_all, q_scale, k_scale, v_scale, stats


def _prepare(
    q, k, v, perm, *, demean: bool, repair_tiles: int, selected: int, prev_stats
):
    """Run the passes and return ``(k_out, v8, v_scale, mean, mu, q8, q_scale,
    k_scale, stats)``. ``q`` selects QK8 and the QK8 outputs are ``None``
    without it. The quantizers read the channel statistics of ``prev_stats``
    when given, otherwise the ones a first pass over V, and under QK8 over K,
    collects."""
    b, s_k, h, d = k.shape
    if d != _D:
        raise ValueError(f"VC fused preparation supports head_dim {_D} only")
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("k and v must be contiguous")
    dev = k.device
    t = (s_k + _TILE - 1) // _TILE
    repair = repair_tiles > 0
    rows = vc_repair_kv_len(s_k, repair_tiles) if repair else s_k
    s_q = 0 if q is None else q.shape[1]
    if prev_stats is not None and (
        prev_stats.shape != (b, h, 2, d) or not prev_stats.is_contiguous()
    ):
        raise ValueError("prev_stats must be contiguous [B, H, 2, D]")
    ker = _compiled(k.dtype, bool(demean), b, s_k, h, repair_tiles, s_q, selected)
    perm32 = None if repair else perm.to(torch.int32).contiguous()
    stats = torch.zeros((b, h, 2, d), dtype=torch.float32, device=dev)
    mean = torch.empty((b, h, t, d), dtype=torch.float32, device=dev)
    v8 = torch.empty((b, rows, h, d), dtype=torch.float8_e4m3fn, device=dev)
    v_scale = torch.empty((b, h, d), dtype=torch.float32, device=dev)
    k_perm = mu = energy = index = None
    if q is None:
        k_perm = torch.empty((b, rows, h, d), dtype=k.dtype, device=dev)
    if repair:
        energy = torch.empty((b, h, s_k), dtype=torch.float32, device=dev)
        index = torch.empty((b, h, selected), dtype=torch.int32, device=dev)
    else:
        groups = (t + VC_MEAN_GROUP_TILES - 1) // VC_MEAN_GROUP_TILES
        mu = torch.zeros(
            (b, h, groups, *vc_mean_group_shape(d)), dtype=torch.bfloat16, device=dev
        )
    flat = lambda x: None if x is None else x.view(-1)  # noqa: E731
    # Without previous statistics a first pass collects this call's own and the
    # second pass reads them. Otherwise one pass does both.
    read_stats = prev_stats
    fresh = stats
    write = ker["kv"]
    if prev_stats is None:
        ker["stats"](
            flat(k),
            flat(v),
            flat(perm32),
            None,
            flat(mean),
            flat(stats),
            None,
            None,
            None,
            None,
            None,
        )
        read_stats, fresh, write = stats, None, ker["kv_only"]
    write(
        flat(k),
        flat(v),
        flat(perm32),
        flat(k_perm),
        flat(mean),
        flat(fresh),
        flat(read_stats),
        flat(v_scale),
        flat(v8),
        flat(mu),
        flat(energy),
    )
    if repair:
        ker["select"](flat(energy), flat(index))
        ker["rows"](
            flat(k) if q is None else None,
            flat(v),
            flat(index),
            flat(read_stats),
            flat(k_perm),
            flat(v8),
        )
    if q is None:
        return k_perm, v8, v_scale, mean, mu, None, None, None, stats
    k8 = torch.empty((b, rows, h, d), dtype=torch.float8_e4m3fn, device=dev)
    q8 = torch.empty((b, s_q, h, d), dtype=torch.float8_e4m3fn, device=dev)
    # The flat scale layout holds one entry per 128-token block of the whole batch.
    k_scale = torch.ones(
        (h, (b * rows + _TILE - 1) // _TILE + b - 1), dtype=torch.float32, device=dev
    )
    q_scale = torch.ones(
        (h, (b * s_q + _TILE - 1) // _TILE + b - 1), dtype=torch.float32, device=dev
    )
    write = ker["k"]
    if prev_stats is None:
        ker["ksum"](flat(k), flat(perm32), None, None, flat(stats), None, None)
        write = ker["k_only"]
    write(
        flat(k),
        flat(perm32),
        None,
        flat(read_stats),
        flat(fresh),
        flat(k8),
        flat(k_scale),
    )
    if repair:
        ker["k_repair"](
            flat(k), None, flat(index), flat(read_stats), None, flat(k8), flat(k_scale)
        )
    ker["q"](flat(q), None, None, None, None, flat(q8), flat(q_scale))
    return k8, v8, v_scale, mean, mu, q8, q_scale, k_scale, stats

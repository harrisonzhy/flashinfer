"""VC-Attention operand preparation kernels (CuTe DSL).

Three small kernels turn bf16/fp16 ``[B, S, H, D]`` Q/K/V into the
prims_ts VC-Attention operands with each tensor read once (V twice) and
written once:

* :class:`VcKvPass1`: per (batch*head, 128-token K/V tile) gathers the permuted
  tokens, subtracts the per-(batch, head) key channel mean, quantizes K to E4M3
  with one scale per tile, and writes the V tile mean and the residual amax;
* :class:`VcKvPass2`: after the per-tensor V residual scale is known, writes
  the E4M3 V residuals and the packed bf16 tile-mean UMMA operand;
* :class:`VcQPass`: per (batch*head, 128-token Q block) amax scale + E4M3.

Following the paper (arXiv 2609.15810): keys are centred by their per-(batch,
head) channel mean, Q and (centred) K rows are rotated by the normalised
128-point Walsh-Hadamard transform before per-128-token-block E4M3
quantization, V residuals get one E4M3 scale per (batch, head, channel), and
the block means are stored divided by that per-channel scale.

Thread mapping in every kernel: 128 threads, thread ``t`` owns head_dim
columns ``(t % 8) * 16 .. + 16`` (one 32-byte vector) of rows
``t // 8 + 16 * i`` for ``i < 8``.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from .fmha_decode.fmha_decode_resources.helpers_common import _pack_float2_to_bf16
from .fmha_decode.fmha_decode_resources.helpers_softmax import _pack_float4_to_fp8_e4m3

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
THREADS = 128
TILE = 128
COLS_PER_THREAD = 16
ROW_GROUPS = THREADS // 8  # 16
ROWS_PER_THREAD = TILE // ROW_GROUPS  # 8
E4M3_MAX = 448.0
MU_ELEMS = 2048  # d=128 x k=16 bf16 per tile
HADAMARD_NORM = 0.08838834764831845  # 1 / sqrt(128)


@cute.jit
def _hadamard128_row(vals, base: cutlass.Constexpr[int], cg: Int32):
    """In-place normalised 128-point Walsh-Hadamard transform of one row whose
    16 columns ``vals[base : base + 16]`` live in this thread and whose other
    112 columns live in the 7 neighbouring lanes ``cg ^ {1, 2, 4}``."""
    for h in (1, 2, 4, 8):
        for i in cutlass.range_constexpr(COLS_PER_THREAD):
            if (i & h) == 0:
                a = vals[base + i]
                b = vals[base + (i ^ h)]
                vals[base + i] = a + b
                vals[base + (i ^ h)] = a - b
    for sbit in cutlass.range_constexpr(3):
        stride = 1 << sbit
        # lanes with bit clear take (mine + other); lanes with bit set take (other - mine)
        sign = Float32(1.0) - Float32(2.0) * Float32((cg >> sbit) & 1)
        for j in cutlass.range_constexpr(COLS_PER_THREAD):
            mine = vals[base + j]
            other = cute.arch.shuffle_sync_bfly(mine, stride)
            vals[base + j] = sign * mine + other
    for j in cutlass.range_constexpr(COLS_PER_THREAD):
        vals[base + j] = vals[base + j] * Float32(HADAMARD_NORM)


@cute.jit
def _abs_f32(x: Float32) -> Float32:
    return cute.arch.fmax(x, -x)


@cute.jit
def _block_max(v: Float32, red_smem, tidx: Int32) -> Float32:
    """Max over the 128 threads of the CTA (4 warps)."""
    for shift in cutlass.range_constexpr(5):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, 1 << shift))
    if tidx % 32 == 0:
        red_smem[tidx // 32] = v
    cute.arch.sync_threads()
    r = red_smem[0]
    for w in cutlass.range_constexpr(1, 4):
        r = cute.arch.fmax(r, red_smem[w])
    cute.arch.sync_threads()
    return r


@cute.jit
def _load_row16(base_addr: Int64, is_bf16: cutlass.Constexpr[bool]):
    """16 bf16/fp16 elements at ``base_addr`` as a Vector of 16 Float32."""
    regs = cutlass.inttoptr(base_addr, mem_space=1, dtype=Int32).load(
        count=8, alignment=32
    )
    if cutlass.const_expr(is_bf16):
        return regs.bitcast(cutlass.BFloat16).to(Float32)
    return regs.bitcast(cutlass.Float16).to(Float32)


@cute.jit
def _store_row16_fp8(base_addr: Int64, vals, scale_inv: Float32):
    """Quantize 16 Float32 values (Array or Vector) by ``scale_inv`` and store 16 E4M3 bytes."""
    packed = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
    for j in cutlass.range_constexpr(4):
        packed[j] = _pack_float4_to_fp8_e4m3(
            vals[4 * j] * scale_inv,
            vals[4 * j + 1] * scale_inv,
            vals[4 * j + 2] * scale_inv,
            vals[4 * j + 3] * scale_inv,
        )
    cutlass.inttoptr(base_addr, mem_space=1, dtype=Int32).store(
        packed.data_ptr().load(count=4, alignment=16), alignment=16
    )


class VcKvPass1:
    def __init__(self, is_bf16: bool):
        self.is_bf16 = is_bf16

    @cute.kernel
    def kernel(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mKMean: cute.Tensor,
        mK8: cute.Tensor,
        mKScale: cute.Tensor,
        mMean: cute.Tensor,
        mVAmax: cute.Tensor,
        S: Int32,
        H: Int32,
        T: Int32,
        BH: Int32,
        HADAMARD: cutlass.Constexpr[bool],
        DEMEAN: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        t, bh, _ = cute.arch.block_idx()
        b = bh // H
        h = bh % H
        cg = tidx % 8
        rg = tidx // 8
        col0 = cg * COLS_PER_THREAD

        smem = cutlass.utils.SmemAllocator()
        col_sums = smem.allocate_array(Float32, ROW_GROUPS * TILE)
        red = smem.allocate_array(Float32, 4)

        k_base = mK.iterator.toint()
        v_base = mV.iterator.toint()
        k8_base = mK8.iterator.toint()
        perm_base = mPerm.iterator.toint() + Int64(bh) * Int64(S) * 4
        row_bytes = Int64(H) * 256  # H * D * 2
        k8_row_bytes = Int64(H) * 128

        toks = cutlass.Array(Int32, ROWS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            pos = t * TILE + rg + 16 * i
            tok = Int32(0)
            if pos < S:
                tok = cutlass.inttoptr(
                    perm_base + Int64(pos) * 4, mem_space=1, dtype=Int32
                ).load()
            toks[i] = tok

        # ---- K: smooth, tile amax, quantize -------------------------------
        kmean = _load_row16_f32(mKMean.iterator.toint() + (Int64(bh) * 128 + col0) * 4)
        kvals = cutlass.Array(
            Float32, ROWS_PER_THREAD * COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
        )
        kmax = Float32(0.0)
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            pos = t * TILE + rg + 16 * i
            if pos < S:
                addr = (
                    k_base
                    + (Int64(b) * Int64(S) + Int64(toks[i])) * row_bytes
                    + (Int64(h) * 128 + col0) * 2
                )
                row = _load_row16(addr, self.is_bf16)
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    kvals[i * COLS_PER_THREAD + j] = row[j] - kmean[j]
            else:
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    kvals[i * COLS_PER_THREAD + j] = Float32(0.0)
        # The rotation needs all 8 lanes of a row (also for padded rows).
        if cutlass.const_expr(HADAMARD):
            for i in cutlass.range_constexpr(ROWS_PER_THREAD):
                _hadamard128_row(kvals, i * COLS_PER_THREAD, cg)
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            pos = t * TILE + rg + 16 * i
            if pos < S:
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    kmax = cute.arch.fmax(
                        kmax, _abs_f32(kvals[i * COLS_PER_THREAD + j])
                    )
        kmax = _block_max(kmax, red, tidx)
        kscale = cute.arch.fmax(kmax / E4M3_MAX, Float32(1e-12))
        kscale_inv = Float32(1.0) / kscale
        if tidx == 0:
            # Flat scale layout (sage.flat_scale_slot): sequence b of head h starts
            # at slot (b*S >> 7) + b; numel per head = ceil(B*S / 128) + B - 1.
            b_idx = bh // H
            h_idx = bh % H
            n_batch = BH // H
            numel = ((n_batch * S + TILE - 1) // TILE) + n_batch - 1
            slot = ((b_idx * S) // TILE) + b_idx + t
            cutlass.inttoptr(
                mKScale.iterator.toint()
                + (Int64(h_idx) * Int64(numel) + Int64(slot)) * 4,
                mem_space=1,
                dtype=Float32,
            ).store(kscale)
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            pos = t * TILE + rg + 16 * i
            if pos < S:
                addr = (
                    k8_base
                    + (Int64(b) * Int64(S) + Int64(pos)) * k8_row_bytes
                    + Int64(h) * 128
                    + col0
                )
                sub = cutlass.Array(
                    Float32, COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
                )
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    sub[j] = kvals[i * COLS_PER_THREAD + j]
                _store_row16_fp8(addr, sub, kscale_inv)

        # ---- V: tile mean and residual amax --------------------------------
        vvals = cutlass.Array(
            Float32, ROWS_PER_THREAD * COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
        )
        psum = cutlass.Array(Float32, COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(COLS_PER_THREAD):
            psum[j] = Float32(0.0)
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            pos = t * TILE + rg + 16 * i
            if pos < S:
                addr = (
                    v_base
                    + (Int64(b) * Int64(S) + Int64(toks[i])) * row_bytes
                    + (Int64(h) * 128 + col0) * 2
                )
                row = _load_row16(addr, self.is_bf16)
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    vvals[i * COLS_PER_THREAD + j] = row[j]
                    psum[j] = psum[j] + row[j]
            else:
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    vvals[i * COLS_PER_THREAD + j] = Float32(0.0)
        for j in cutlass.range_constexpr(COLS_PER_THREAD):
            col_sums[rg * TILE + col0 + j] = psum[j]
        cute.arch.sync_threads()
        valid_rows = S - t * TILE
        if valid_rows > TILE:
            valid_rows = Int32(TILE)
        inv_count = Float32(1.0) / Float32(valid_rows)
        mean = cutlass.Array(Float32, COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(COLS_PER_THREAD):
            acc = Float32(0.0)
            for g in cutlass.range_constexpr(ROW_GROUPS):
                acc = acc + col_sums[g * TILE + col0 + j]
            mean[j] = acc * inv_count
            if cutlass.const_expr(not DEMEAN):
                mean[j] = Float32(0.0)
        if rg == 0:
            mean_addr = (
                mMean.iterator.toint() + ((Int64(bh) * Int64(T) + t) * 128 + col0) * 4
            )
            cutlass.inttoptr(mean_addr, mem_space=1, dtype=Float32).store(
                mean.data_ptr().load(count=16, alignment=64), alignment=64
            )
        # Per-channel residual amax of this tile (reduced over tiles on the host).
        cmax = cutlass.Array(Float32, COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(COLS_PER_THREAD):
            cmax[j] = Float32(0.0)
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            pos = t * TILE + rg + 16 * i
            if pos < S:
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    cmax[j] = cute.arch.fmax(
                        cmax[j], _abs_f32(vvals[i * COLS_PER_THREAD + j] - mean[j])
                    )
        cute.arch.sync_threads()  # everyone is done reading col_sums as sums
        for j in cutlass.range_constexpr(COLS_PER_THREAD):
            col_sums[rg * TILE + col0 + j] = cmax[j]
        cute.arch.sync_threads()
        if rg == 0:
            for j in cutlass.range_constexpr(COLS_PER_THREAD):
                acc = Float32(0.0)
                for g in cutlass.range_constexpr(ROW_GROUPS):
                    acc = cute.arch.fmax(acc, col_sums[g * TILE + col0 + j])
                cmax[j] = acc
            amax_addr = (
                mVAmax.iterator.toint() + ((Int64(bh) * Int64(T) + t) * 128 + col0) * 4
            )
            cutlass.inttoptr(amax_addr, mem_space=1, dtype=Float32).store(
                cmax.data_ptr().load(count=16, alignment=64), alignment=64
            )

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mKMean: cute.Tensor,
        mK8: cute.Tensor,
        mKScale: cute.Tensor,
        mMean: cute.Tensor,
        mVAmax: cute.Tensor,
        S: Int32,
        H: Int32,
        T: Int32,
        BH: Int32,
        hadamard: cutlass.Constexpr[bool],
        demean: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(
            mK,
            mV,
            mPerm,
            mKMean,
            mK8,
            mKScale,
            mMean,
            mVAmax,
            S,
            H,
            T,
            BH,
            hadamard,
            demean,
        ).launch(
            grid=[T, BH, 1],
            block=[THREADS, 1, 1],
            smem=(ROW_GROUPS * TILE + 4) * 4,
            stream=stream,
        )


@cute.jit
def _load_row16_f32(base_addr: Int64):
    return cutlass.inttoptr(base_addr, mem_space=1, dtype=Float32).load(
        count=16, alignment=64
    )


class VcKvPass2:
    def __init__(self, is_bf16: bool):
        self.is_bf16 = is_bf16

    @cute.kernel
    def kernel(
        self,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVScale: cute.Tensor,
        mV8: cute.Tensor,
        mMu: cute.Tensor,
        S: Int32,
        H: Int32,
        T: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        t, bh, _ = cute.arch.block_idx()
        b = bh // H
        h = bh % H
        cg = tidx % 8
        rg = tidx // 8
        col0 = cg * COLS_PER_THREAD
        v_base = mV.iterator.toint()
        v8_base = mV8.iterator.toint()
        perm_base = mPerm.iterator.toint() + Int64(bh) * Int64(S) * 4
        row_bytes = Int64(H) * 256
        v8_row_bytes = Int64(H) * 128
        mean_base = mMean.iterator.toint() + (Int64(bh) * Int64(T) + t) * 512

        vscale_base = mVScale.iterator.toint() + Int64(bh) * 512  # [B, H, 128] fp32
        vs = _load_row16_f32(vscale_base + col0 * 4)
        vs_inv = cutlass.Array(
            Float32, COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
        )
        for j in cutlass.range_constexpr(COLS_PER_THREAD):
            vs_inv[j] = Float32(1.0) / vs[j]
        mean = _load_row16_f32(mean_base + col0 * 4)
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            pos = t * TILE + rg + 16 * i
            if pos < S:
                tok = cutlass.inttoptr(
                    perm_base + Int64(pos) * 4, mem_space=1, dtype=Int32
                ).load()
                addr = (
                    v_base
                    + (Int64(b) * Int64(S) + Int64(tok)) * row_bytes
                    + (Int64(h) * 128 + col0) * 2
                )
                row = _load_row16(addr, self.is_bf16)
                res = cutlass.Array(
                    Float32, COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
                )
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    res[j] = (row[j] - mean[j]) * vs_inv[j]
                out_addr = (
                    v8_base
                    + (Int64(b) * Int64(S) + Int64(pos)) * v8_row_bytes
                    + Int64(h) * 128
                    + col0
                )
                _store_row16_fp8(out_addr, res, Float32(1.0))

        # Packed mean operand tile: index idx in [0, 2048): g = idx // 128,
        # c = (idx // 64) % 2, r = (idx // 8) % 8, k = idx % 8; value is
        # mean[g * 8 + r] / v_scale when c == 0 and k < 2, else 0.
        packed = cutlass.Array(Int32, 8, space=cutlass.AddressSpace.rmem)
        idx0 = tidx * 16
        g = idx0 // 128
        c = (idx0 // 64) % 2
        r0 = (idx0 // 8) % 8  # rows r0 (k 0..7) and r0 + 1 (k 8..15)
        d0 = g * 8 + r0
        m0 = cutlass.inttoptr(
            mean_base + Int64(d0) * 4, mem_space=1, dtype=Float32
        ).load()
        m1 = cutlass.inttoptr(
            mean_base + Int64(d0 + 1) * 4, mem_space=1, dtype=Float32
        ).load()
        s0 = cutlass.inttoptr(
            vscale_base + Int64(d0) * 4, mem_space=1, dtype=Float32
        ).load()
        s1 = cutlass.inttoptr(
            vscale_base + Int64(d0 + 1) * 4, mem_space=1, dtype=Float32
        ).load()
        keep = Float32(0.0)
        if c == 0:
            keep = Float32(1.0)
        # Means are stored divided by the per-channel value scale (paper, App. B).
        m0 = m0 / s0 * keep
        m1 = m1 / s1 * keep
        for j in cutlass.range_constexpr(8):
            packed[j] = Int32(0)
        packed[0] = _pack_float2_to_bf16(m0, m0)
        packed[4] = _pack_float2_to_bf16(m1, m1)
        cutlass.inttoptr(
            mMu.iterator.toint() + ((Int64(bh) * Int64(T) + t) * MU_ELEMS + idx0) * 2,
            mem_space=1,
            dtype=Int32,
        ).store(packed.data_ptr().load(count=8, alignment=32), alignment=32)

    @cute.jit
    def __call__(
        self,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVScale: cute.Tensor,
        mV8: cute.Tensor,
        mMu: cute.Tensor,
        S: Int32,
        H: Int32,
        T: Int32,
        BH: Int32,
        stream,
    ):
        self.kernel(mV, mPerm, mMean, mVScale, mV8, mMu, S, H, T).launch(
            grid=[T, BH, 1], block=[THREADS, 1, 1], stream=stream
        )


class VcQPass:
    def __init__(self, is_bf16: bool):
        self.is_bf16 = is_bf16

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQ8: cute.Tensor,
        mQScale: cute.Tensor,
        S: Int32,
        H: Int32,
        NB: Int32,
        BH: Int32,
        HADAMARD: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        nb, bh, _ = cute.arch.block_idx()
        b = bh // H
        h = bh % H
        cg = tidx % 8
        rg = tidx // 8
        col0 = cg * COLS_PER_THREAD
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_array(Float32, 4)
        q_base = mQ.iterator.toint()
        q8_base = mQ8.iterator.toint()
        row_bytes = Int64(H) * 256
        q8_row_bytes = Int64(H) * 128
        qvals = cutlass.Array(
            Float32, ROWS_PER_THREAD * COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
        )
        qmax = Float32(0.0)
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            pos = nb * TILE + rg + 16 * i
            if pos < S:
                addr = (
                    q_base
                    + (Int64(b) * Int64(S) + Int64(pos)) * row_bytes
                    + (Int64(h) * 128 + col0) * 2
                )
                row = _load_row16(addr, self.is_bf16)
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    qvals[i * COLS_PER_THREAD + j] = row[j]
            else:
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    qvals[i * COLS_PER_THREAD + j] = Float32(0.0)
        if cutlass.const_expr(HADAMARD):
            for i in cutlass.range_constexpr(ROWS_PER_THREAD):
                _hadamard128_row(qvals, i * COLS_PER_THREAD, cg)
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            pos = nb * TILE + rg + 16 * i
            if pos < S:
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    qmax = cute.arch.fmax(
                        qmax, _abs_f32(qvals[i * COLS_PER_THREAD + j])
                    )
        qmax = _block_max(qmax, red, tidx)
        qscale = cute.arch.fmax(qmax / E4M3_MAX, Float32(1e-12))
        qscale_inv = Float32(1.0) / qscale
        if tidx == 0:
            # Flat scale layout (sage.flat_scale_slot), see the K pass.
            b_idx = bh // H
            h_idx = bh % H
            n_batch = BH // H
            numel = ((n_batch * S + TILE - 1) // TILE) + n_batch - 1
            slot = ((b_idx * S) // TILE) + b_idx + nb
            cutlass.inttoptr(
                mQScale.iterator.toint()
                + (Int64(h_idx) * Int64(numel) + Int64(slot)) * 4,
                mem_space=1,
                dtype=Float32,
            ).store(qscale)
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            pos = nb * TILE + rg + 16 * i
            if pos < S:
                addr = (
                    q8_base
                    + (Int64(b) * Int64(S) + Int64(pos)) * q8_row_bytes
                    + Int64(h) * 128
                    + col0
                )
                sub = cutlass.Array(
                    Float32, COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
                )
                for j in cutlass.range_constexpr(COLS_PER_THREAD):
                    sub[j] = qvals[i * COLS_PER_THREAD + j]
                _store_row16_fp8(addr, sub, qscale_inv)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mQ8: cute.Tensor,
        mQScale: cute.Tensor,
        S: Int32,
        H: Int32,
        NB: Int32,
        BH: Int32,
        hadamard: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(mQ, mQ8, mQScale, S, H, NB, BH, hadamard).launch(
            grid=[NB, BH, 1], block=[THREADS, 1, 1], smem=16, stream=stream
        )


# ---------------------------------------------------------------------------
# Compilation cache (TVM-FFI entry points taking torch tensors directly)
# ---------------------------------------------------------------------------
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
def _compiled(dtype: torch.dtype, hadamard: bool, demean: bool):
    cdt, is_bf16 = _in_dtype(dtype)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    f32 = lambda: _fake1d(Float32, 64)  # noqa: E731
    i32 = lambda: _fake1d(Int32, 16)  # noqa: E731
    e4m3 = lambda: _fake1d(cutlass.Float8E4M3FN, 16)  # noqa: E731
    p1 = cute.compile(
        VcKvPass1(is_bf16),
        _fake1d(cdt),
        _fake1d(cdt),
        i32(),
        f32(),
        e4m3(),
        f32(),
        f32(),
        f32(),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        hadamard,
        demean,
        stream,
        options=_COMPILE_OPTIONS,
    )
    p2 = cute.compile(
        VcKvPass2(is_bf16),
        _fake1d(cdt),
        i32(),
        f32(),
        f32(),
        e4m3(),
        _fake1d(cutlass.BFloat16),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        stream,
        options=_COMPILE_OPTIONS,
    )
    pq = cute.compile(
        VcQPass(is_bf16),
        _fake1d(cdt),
        e4m3(),
        f32(),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        hadamard,
        stream,
        options=_COMPILE_OPTIONS,
    )
    return p1, p2, pq


@torch.no_grad()
def vc_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor,
    *,
    hadamard: bool = True,
    demean: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Run the three preparation kernels.

    ``q``, ``k``, ``v``: contiguous ``[B, S, H, 128]`` bf16/fp16; ``perm``:
    ``[B, H, S_k]`` int32/int64. ``hadamard`` rotates Q and centred K rows by
    the normalised 128-point Hadamard before quantization; ``demean=False``
    keeps the tile means at zero (V-Smooth off, plain per-channel E4M3 V).
    Returns ``(q8, k8, v8, q_scale, k_scale, v_scale, mean, mu)`` in the
    :class:`VCAttentionOperands` layouts (Q/K scales in the sage flat layout);
    ``v_scale`` is ``[B, H, 128]``.
    """
    b, s_k, h, d = k.shape
    s_q = q.shape[1]
    if d != 128:
        raise ValueError("VC fused preparation supports head_dim 128 only")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k, v must be contiguous")
    dev = q.device
    t = (s_k + TILE - 1) // TILE
    nbq = (s_q + TILE - 1) // TILE
    bh = b * h
    p1, p2, pq = _compiled(q.dtype, bool(hadamard), bool(demean))
    perm32 = perm.to(torch.int32).contiguous()
    k_mean = k.float().mean(dim=1).contiguous()  # [B, H, D]
    k8 = torch.empty((b, s_k, h, d), dtype=torch.float8_e4m3fn, device=dev)
    v8 = torch.empty_like(k8)
    q8 = torch.empty((b, s_q, h, d), dtype=torch.float8_e4m3fn, device=dev)
    # Flat scale layout (sage.flat_scale_numel): [H, ceil(B*S/blk) + B - 1].
    k_scale = torch.ones(
        (h, (b * s_k + TILE - 1) // TILE + b - 1), dtype=torch.float32, device=dev
    )
    q_scale = torch.ones(
        (h, (b * s_q + TILE - 1) // TILE + b - 1), dtype=torch.float32, device=dev
    )
    mean = torch.empty((b, h, t, d), dtype=torch.float32, device=dev)
    vamax = torch.empty((b, h, t, d), dtype=torch.float32, device=dev)
    mu = torch.empty((b, h, t, 8, 256), dtype=torch.bfloat16, device=dev)
    p1(
        k.view(-1),
        v.view(-1),
        perm32.view(-1),
        k_mean.view(-1),
        k8.view(-1),
        k_scale.view(-1),
        mean.view(-1),
        vamax.view(-1),
        s_k,
        h,
        t,
        bh,
    )
    v_scale = (vamax.amax(dim=2) / E4M3_MAX).clamp_min(1e-12).contiguous()  # [B, H, D]
    p2(
        v.view(-1),
        perm32.view(-1),
        mean.view(-1),
        v_scale.view(-1),
        v8.view(-1),
        mu.view(-1),
        s_k,
        h,
        t,
        bh,
    )
    pq(q.view(-1), q8.view(-1), q_scale.view(-1), s_q, h, nbq, bh)
    return q8, k8, v8, q_scale, k_scale, v_scale, mean, mu

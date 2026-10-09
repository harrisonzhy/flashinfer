# Experimental Task-Scheduled Attention

`flashinfer.attention.prims_ts` exposes experimental CuTe DSL attention
kernels for NVIDIA Blackwell GPUs. Scheduling, tile selection, and split-KV
reduction are implementation details; the public interfaces expose attention
and cache semantics without tuning knobs.

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but is not yet signoff-qualified.

## Guides and public APIs

Import all entries below from `flashinfer.attention.prims_ts`.

| Kernel | Guide | Public APIs |
| --- | --- | --- |
| FMHA context/prefill | [Task-Scheduled FMHA Context](kernels/fmha_context/README.md), [VC-Attention-QK16](#vc-attention-qk16) below | `BatchPrefillTSWrapper`, `batch_prefill`, `VCAttentionConfig`, `VCAttentionParams`, `VCAttentionPreprocessor`, `BatchPrefillPagedTSWrapper`, `batch_prefill_with_paged_kv_cache` |
| FMHA decode | [Task-Scheduled FMHA Decode](kernels/fmha_decode/README.md) | `BatchDecodePagedTSWrapper`, `batch_decode_with_paged_kv_cache`, `get_prims_ts_batch_decode_workspace_size`, `prims_ts_batch_decode_with_kv_cache` |
| Block-sparse FMHA | — | `BlockSparseTSWrapper`, `block_sparse_attention`; fixed-Q paged KV: `BlockSparsePagedTSWrapper`, `block_sparse_attention_with_paged_kv_cache` |
| MLA decode | [Task-Scheduled MLA Decode](kernels/mla_decode/README.md) | `BatchMLADecodePagedTSWrapper`, `batch_mla_decode_with_paged_kv_cache`, `get_prims_ts_batch_mla_decode_workspace_size`, `prims_ts_batch_mla_decode_with_kv_cache` |

The component guides define supported shapes, layouts, metadata lifetime,
output/workspace ownership, examples, limitations, and validation commands.

The contiguous and paged context, FMHA decode, and MLA decode wrappers separate
reusable static state from per-run request state. By default, `plan()` compiles
a static capacity, shape, dtype, and storage-mode specialization without
retaining request tensors or metadata. FMHA decode plans may instead own fixed
K/V lengths when `plan(seq_lens=...)` is supplied; those runs must pass
`seq_lens=None`, and changing lengths requires replanning. Paged context plans
may additionally freeze
explicit exact-uniform-length, zero-causal-offset, or zeroed-V-tail promises.
Context `run()` receives current packed offsets or fixed-table paged metadata;
per-token variable-window bounds for fixed-shape inputs are also per-run, with
optional caller-precomputed per-CTA start minima. Both context wrappers own
their default scale tensors. Contiguous variable-window plans additionally own
mutable fallback scratch that derives CTA minima only when the caller omits
them, while paged context owns no other workspace. With `validate=False`,
supplying CTA minima avoids that repeated preprocessing and leaves all
variable-window metadata caller-owned. Runtime validation is enabled by
default; callers that have already validated their inputs may use
`validate=False` for steady-state timing or CUDA Graph capture and then own
every dtype, device, shape, stride, alignment, value, aliasing, and lifetime
obligation.

For `BlockSparsePagedTSWrapper`, `plan` freezes only the compact fixed-Q
geometry, dtypes, sparse-route capacity, and `max_seq_len_kv`; it retains no
request metadata. Every `run` reads live page tables, per-request K/V lengths,
per-KV-head sparse routes, and optional token bits from device tensors.
`block_tables` is Int32 `[B, C]`, contiguous within each row and free to use a
padded outer row stride; `C * page_size` must cover `max_seq_len_kv`, and only
the first `ceil(seq_lens_kv[b] / page_size)` entries of each row are read. The
caller owns every live value contract: dense K/V lengths must be in
`[1, max_seq_len_kv]`, causal lengths must be in `[Sq, max_seq_len_kv]`, and
every live physical page ID must lie in `[0, P)`. Every BSR row must have
bounded offsets, strictly increasing unique block IDs, and at most the planned
`max_blocks_per_row` entries. Contiguous IDs must lie below
`ceil(seq_len_kv / kv_block_size)`; paged IDs must start below the owning
request's live K/V length.

Reusable wrappers validate tensor structure and aliasing by default and read
values directly without host synchronization; `validate=False` skips the
structural checks as well. Invalid values therefore have undefined behavior and
may access out of bounds. Set `CUTE_DSL_ENABLE_ASSERTIONS=1` before the process
first compiles these kernels to diagnose violations encountered while preparing
selected routes; such assertions report asynchronously and leave the CUDA
context unusable. The one-shot APIs instead synchronize once to validate all
live values, including the complete physical-page-ID prefix, before creating
their temporary plans and cannot run during CUDA Graph capture.

The one-shot `block_sparse_attention_with_paged_kv_cache` API takes
`max_seq_len_kv` as the static capacity and requires `seq_lens_kv` with the
live per-request logical lengths. Paged PrimTS does not support packed or
mixed/variable Q lengths.
Eager launches retain all launch tensors on the run stream; CUDA Graph users
must keep the wrapper and Q/cache/output/runtime-metadata tensors alive and
unmodified until replay completes. Values may change between completed replays
while tensor addresses, shapes, dtypes, and strides remain stable.

Qualified Q64/coarse-KV profiles retain KV256 routes for page sizes 64 and
128. Optional `kv_valid_bits` is a `torch.uint32` per-request bitset with shape
`[B, ceil(max_seq_len_kv / 32)]` over logical KV tokens; it is shared by all KV
heads and independent of the physical page mapping.

For contiguous block-sparse attention, both `BlockSparseTSWrapper.plan` and
the `block_sparse_attention` one-shot API can opt into
`sparse_format="bitmask"` and/or `use_proxy_routes=True`. BSR and packed
exact-block bitmaps are alternative frontends; both are prepared into the same
route stream before attention. The bitmask one-shot uses the full structural
KV-block count as its temporary plan capacity, while reusable plans accept a
tighter caller-provided bound. Proxy routes are supported across the existing
contiguous block-sparse profiles and preserve the profile's Q tile, KV route,
and KeepsAB/SWAPAB geometry. Proxy routes currently require
`mask_type="dense"`; paged K/V proxy execution remains unsupported. Route rows
are owned by `(batch, KV head, Q block)`, so all Q heads in one GQA/MQA group
share sparsity. A proxy run supplies one K arithmetic mean and one V sum per
semantic KV block. The final partial block uses only its structural tokens.
Optional `kv_valid_bits` filters exact K/V tokens only and does not change
proxy summaries or their represented mass.

## VC-Attention-QK16

VC-Attention-QK16, a bf16 Q/K adaptation of
[Li et al., 2026](https://arxiv.org/html/2609.15810v1), runs the dense
fixed-length context kernel (`BatchPrefillTSWrapper` and `batch_prefill`) with
bf16 Q and K, E4M3 V with value smoothing, and a direct probability cast.
The plan selects the recipe at compile time with
`vc_config=VCAttentionConfig(...)`. Every run passes the permuted K, the E4M3 V
residuals and `vc=VCAttentionParams(...)`:

```text
S[r][c] = (Q . K[perm]^T)[r][c]
P8      = softmax_row(S), quantized to E4M3
O[r][d] = sfV[b, h, d] * (sum_c P8[r][c] * V8[perm(c)][d] + sum_t rowsum_t(P8[r]) * mu_t[d]) / l[r]
```

`V8` holds the E4M3 residuals of the k-means-permuted values around their
128-token tile means `mu_t` (stored in bf16, divided by `sfV`); the kernel adds
the mean terms back inside the online softmax recurrence, two K=16 UMMA steps per
group of 16 tiles, so the row-max correction covers them. `VCAttentionPreprocessor` turns bf16 K/V into the
operands with the paper's V-Smooth schedule: grouping and demeaning run on the
first `smooth_step_fraction` of the denoising steps, the permutation is
refreshed every `perm_refresh_every` steps inside that window and kept
afterwards. `vc_quantize` is the torch reference of the same preparation.

| Input | Supported values |
| --- | --- |
| Q/K dtype | `torch.bfloat16` |
| V dtype | `torch.float8_e4m3fn` |
| Output dtype | `torch.bfloat16` or `torch.float16` |
| `k_block_size` | 128 (the K/V tile whose mean is restored) |
| Geometry | `head_dim=128`, `num_qo_heads == num_kv_heads`, `packed=False`, `mask_type="dense"`; runs the two-CTA UMMA form |

### Operand tensors

`v_scale` is the `[B, Hkv, D]` fp32 per-channel residual scale and
`tile_means` the packed bf16 `[B, Hkv, ceil(Skv / 2048), 16, 256]` mean operands
from `pack_vc_tile_means`; both are contiguous and 16-byte aligned on the run
device, and a validating `run()` checks them against the plan. `demean` says
whether the run restores the means (`False` after the V-Smooth window, when
they are zero). `VCAttentionPreprocessor.prepare`, `vc_quantize_fused` (CuTe
DSL) and `vc_quantize` (torch) return them with the permuted K and the E4M3 V
as `VCAttentionOperands`, whose `.params` is the run-time object.

### Example

```python
from flashinfer.attention.prims_ts import (
    BatchPrefillTSWrapper, VCAttentionConfig, VCAttentionPreprocessor,
)

wrapper = BatchPrefillTSWrapper()
wrapper.plan(
    device="cuda", batch_size=1, max_seq_len_q=S, max_kv_len=S,
    num_qo_heads=H, num_kv_heads=H, head_dim=128,
    q_dtype=torch.bfloat16, k_dtype=torch.bfloat16, v_dtype=torch.float8_e4m3fn,
    out_dtype=torch.bfloat16, vc_config=VCAttentionConfig(),
)
prep = VCAttentionPreprocessor()
ops = prep.prepare(k_bf16, v_bf16, denoise_step=(step, num_steps))
out = wrapper.run(q_bf16, ops.k, ops.v, vc=ops.params)
```

The one-shot `batch_prefill(..., vc=ops.params)` plans the recipe from the
operands; `vc_config` without `vc` is rejected.

## Validation

Run the numerical, graph, scheduler/resource, alias-safety, and public-surface
contracts:

```bash
pytest -q \
  tests/attention/test_attention_ts_context.py \
  tests/attention/test_attention_ts_decode.py \
  tests/attention/test_attention_ts_block_sparse.py \
  tests/attention/test_attention_ts_mask.py \
  tests/attention/test_attention_ts_mla_decode.py
```

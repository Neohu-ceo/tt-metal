# PLAN: lift the DST-based H cap in sparse_sdpa via query sub-blocking

## Problem

The compute kernel processes **all** `Sqt = H/TILE_HEIGHT` query tile-rows as a *single* DST
sub-block in every DST-bound primitive: the QK matmul (`subblock_h=Sqt`), `reduce_c_row_group`
(`sbh=Sqt`), `sub_exp_block_bcast_cols` (`sbh=Sqt`), the PV matmul (`subblock_h=Sqt`),
`salad_correct_fused<Sqt,…>`, and `sub_exp_first_col_blocks` (`sbh=Sqt`). Each holds `sbh` (or
`sbh*sbw`) output tiles live in the DEST register file, so they require **Sqt ≤ dst_size**.

`dst_size` (half-sync) = **8** without fp32-dest-acc, **4** with. Hence the current hard cap
H ≤ 256 (or 128 under fp32-acc), enforced today by a `TT_FATAL` in `validate_on_program_cache_miss`.

This cap is **not fundamental** — DST only bounds the matmul/SFPU sub-block dims. Iterating the
work in query-row groups of `qsb ≤ dst_size` removes it; H is then bounded only by L1.

## The proven pattern (already in-tree)

`compute_streaming.hpp:1227` (the dense streaming SDPA) does exactly this:

```
constexpr uint32_t q_num_subblocks = Sq_chunk_t / qkt_subblock_h;   // qkt_subblock_h <= dst_size
for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) { … QK, mask, reduce, exp, PV … }
```

Every primitive we call **already takes the group index** (`row_group_index` / `q_subblock` /
`ob_q_subblock` / `row_subblock_idx`) and uses cumulative CB indexing `(idx+1)*group_size`, so
they are built to be called group-by-group against a full-size CB. We only need to add the loop
and thread `qg`/`qsb` through; no primitive signature changes.

## Design

- `qsb` = query tile-rows per group = **largest divisor of `Sqt` with `qsb ≤ dst_size`**
  (mirrors `determine_largest_subblock_size`). For `Sqt ≤ dst_size` this is `qsb = Sqt`, one
  group → byte-identical to today. `qsb` divides `Sqt`, so all groups are equal-height (required:
  `salad_correct_fused`'s `sbh_t` is a **template/compile-time** param).
- `q_groups = Sqt / qsb`.
- Both are **compile-time** (host derives `qsb` from `Sqt` and `dst_size`, passes `qsb` as a new
  compute compile arg; kernel computes `q_groups = Sqt/qsb`). `dst_size` on host = `fp32_acc ? 4 : 8`.
- CBs stay full-size (`cb_qk_im` = `Sqt*Skt`, `cb_out_*` = `Sqt*vDHt`, `cb_max/sum` = `Sqt`). Each
  group writes/reads its own row band `[qg*qsb, (qg+1)*qsb)`.
- `cb_k_in` / `cb_q_in` are **shared** across groups → pop once *after* the group loop (not per group).

## Call-site changes (compute kernel, per K-chunk body)

Wrap the existing per-chunk body in `for (uint32_t qg = 0; qg < q_groups; ++qg)`, with
`row_base = qg*qsb`. Threading (all indices that are `0`/`Sqt` today):

| primitive | today | becomes |
|---|---|---|
| QK `blocked_matmul_and_pack` | `in0_index_start=0, row_subblock_idx=0, subblock_h=Sqt` | `in0_index_start=row_base*DHt, row_subblock_idx=qg, subblock_h=qsb` |
| mask loop | `for r in [0,Sqt)` add at `r*Skt+t` | `for r in [0,qsb)` add at `(row_base+r)*Skt+t` |
| `reduce_c_row_group` | `row_group_index=0, sbh=Sqt` | `row_group_index=qg, sbh=qsb` |
| `sub_exp_block_bcast_cols` | `q_subblock=0, sbh=Sqt` | `q_subblock=qg, sbh=qsb` |
| PV `blocked_matmul_and_pack` | `in0_index_start=0, row_subblock_idx=0, subblock_h=Sqt` | `in0_index_start=row_base*Skt, row_subblock_idx=qg, subblock_h=qsb` |
| `sub_exp_first_col_blocks` (salad corr) | `q_subblock=0, Sqt` | `q_subblock=qg, qsb` |
| `salad_correct_fused<Sqt,vDHt,dst>` | `<Sqt,…>(…,0,0,0)` | `<qsb,vDHt,dst>(…,qg,qg,qg)` |
| `normalize_row_streaming(…,Sqt)` | `sbh=Sqt` | **No change.** Verified DST-safe for any `sbh`: it loops one row at a time (`:693`), the matmul_reduce/recip use ≤1 DST, and the output normalize batches `head_dim` by `dst_size` (`:751`). Keep `sbh=Sqt` and call it **once after** the group loop (on the last chunk), as today. |

### Ordering / state subtleties to preserve
- **Per-group max/sum reserve**: `cb_reserve_back(cb_qk_im, Sqt*KT_stride)` and `cb_reserve_back(sum_cur,Sqt)`
  stay once per chunk (full size). `cb_reserve_back(max_cur,Sqt)` likewise. The group loop packs into
  bands; `cb_push_back(max_cur,Sqt)` / `cb_push_back(sum_cur,Sqt)` / `cb_push_back(out_cur,Sqt*vDHt)`
  fire **once after** the group loop (the cumulative `cb_wait_front` inside primitives expects the band
  filled incrementally — mirror the reference, which pushes after all subblocks).
- **In-place QK held wr-ptr**: `cb_push_back_hold_wr_ptr(cb_qk_im, Sqt*KT_stride)` stays once. Keep the QK
  matmul for *all* groups before the mask/softmax, OR (simpler, sequential) do full per-group pipeline —
  pick the sequential form first (QK→mask→reduce→exp→PV for group qg) and confirm the `cb_qk_im`
  visibility (`cb_wait_front` cumulative) holds; the reference interleaves for perf but that's optional.
- **pack-config / l1-acc**: the `configure_row_pack_width` + `skip_pack_configure` dance and the
  `llk_pack_reconfig_l1_acc` in sub_exp/salad are per-call; they already work per-subblock in the
  reference. Re-init exp scale (`exp_packthread_tile_init`) placement: keep per-chunk (once), as the
  reference does (`:1212`, before the q_subblock loop).
- **salad pack_init restore** (`llk_pack_init<Default>` before `salad_correct_fused`): keep before the
  salad call; with the group loop it runs per group — fine (it's idempotent geometry restore).

## Host changes (`sparse_sdpa_program_factory.cpp`)
- Compute `dst_size = fp32_acc ? 4 : 8`; `qsb = largest divisor of Sqt ≤ dst_size`; append `qsb` to
  `compute_ct` (new compile arg). Kernel reads it and derives `q_groups = Sqt/qsb`.

## Validation change (`sparse_sdpa_device_operation.cpp`)
- **Delete** the `Sqt ≤ dst_size` FATAL added for fp32-acc. After sub-blocking, H is L1-bound only
  (large H fails at CB allocation = OOM, per the existing "let it OOM" decision). Keep `H % TILE_HEIGHT == 0`.

## Test sequence (run each via `scripts/run_safe_pytest.sh`, no `--dev`)
1. **Regression (qsb==Sqt path)**: full suite minus perf — must stay 32/32 (H≤128 cases all have Sqt≤dst_size,
   so `qsb=Sqt`, one group → identical codegen/results). Confirms the loop refactor is a no-op there.
2. **New multi-group correctness**: H=192,256 at k_chunk=32 (Sqt=6,8 with dst_size=8 → still one group at
   bf16; force multi-group by also testing under fp32-acc where dst_size=4 → Sqt=6,8 ⇒ 2 groups). Add
   `H=256, fp32_acc=True` (Sqt=8, qsb=4, 2 groups) and `H=192, fp32_acc=True` (Sqt=6, qsb=3 or 2 groups).
   PCC ≥ 0.99 vs `sparse_mla` golden.
3. **fp32-acc large-H now allowed**: the H=160 fp32=True case that the FATAL rejects today should pass
   (Sqt=5, qsb=5 if 5≤? no, dst_size=4 → qsb=1, 5 groups — slow but correct). Confirms cap lifted.
4. **Perf**: prod-dense (H=32) unchanged (Sqt=1, one group — no loop overhead). Re-profile to confirm ~4.0 ms.

## Risks
- **TR-thread stack / code size**: the group loop + threaded indices grow the compute kernel; watch the
  watcher kernel-config-buffer overflow (prior issue — outlined pack_tile). Re-check after build.
- **qsb=1 inefficiency** for `Sqt` whose only divisor ≤dst_size is 1 (e.g. Sqt=5,7 under fp32 dst_size=4).
  Acceptable (correctness over speed for odd large H); `log`/document. A future refine can pad Sqt up to a
  friendlier multiple.
- **In-place cb_qk_im band visibility** across groups — the highest-risk piece; validate with the
  H=256/fp32 (2-group) PCC test, and if PCC drops, suspect a missing `pack_to_unpack_sync` / cumulative
  `cb_wait_front` between groups.

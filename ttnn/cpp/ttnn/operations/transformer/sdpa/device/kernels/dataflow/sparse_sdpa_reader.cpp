// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa reader: per token, gather Q rows + per-chunk indexed K rows (sentinel suffix zero-filled),
// plus the per-token partial-boundary mask tile. (The persistent scaler/col-identity/-inf tiles are built
// by the lighter-loaded writer.) Uses the object-based NoC + circular-buffer APIs.
//
// Sentinels are a contiguous tail (producer contract): nv = first-sentinel index = valid-key count, so only
// ceil(nv/k_chunk) chunks are active; all-sentinel chunks are skipped (no read/fill/compute), compute learns
// nv + the active count via cb_ctrl. The per-chunk K gather is split across BOTH NoCs (the op is NoC-link
// bound): the reader gathers half the rows on its NoC and hands the other half to the writer (cb_kreq/kack),
// which gathers them on its NoC into the same shared cb_k_rm L1.

#include <stdint.h>
#include <tt-metalium/constants.hpp>  // tt::constants::TILE_HEIGHT
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include <tools/profiler/kernel_profiler.hpp>  // DeviceZoneScopedN
#include "sparse_sdpa_gather.hpp"              // dual-NoC trid-ring gather helper (shared with the writer)

constexpr uint32_t sentinel = 0xFFFFFFFFu;
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t topk = get_compile_time_arg_val(2);
    constexpr uint32_t k_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t k_dim = get_compile_time_arg_val(4);  // q/kv head dim (from the tensor)

    // CB ids — passed by the factory (the SparseCB enum is the single source); order must match the
    // factory's reader CB-arg block (compile args 5..9). cb_kreq/cb_kack come as defines (CB_KREQ/CB_KACK).
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(5);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(6);
    constexpr uint32_t cb_mask_part = get_compile_time_arg_val(7);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(8);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(9);

    // Element sizes (from the tensors' dtypes; passed after the CB ids so their indices stay fixed).
    constexpr uint32_t q_elem_bytes = get_compile_time_arg_val(10);    // bf16
    constexpr uint32_t kv_elem_bytes = get_compile_time_arg_val(11);   // bf16 or fp8_e4m3
    constexpr uint32_t idx_elem_bytes = get_compile_time_arg_val(12);  // uint32

    // kv carries a RUNTIME tensor shape (its T dim is common runtime args, not compile-time), so its accessor
    // spans both the compile-time AND common-runtime arg streams. Thread both offsets through all three.
    constexpr auto q_args = TensorAccessorArgs<13, 0>();
    constexpr auto kv_args =
        TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    constexpr auto idx_args =
        TensorAccessorArgs<kv_args.next_compile_time_args_offset(), kv_args.next_common_runtime_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t tok_start = get_arg_val<uint32_t>(3);
    const uint32_t tok_count = get_arg_val<uint32_t>(4);
    // Indexed KV cache: page offset (cache_batch_idx * T) selecting the cache's batch slot; 0 if not indexed.
    const uint32_t kv_batch_page_offset = get_arg_val<uint32_t>(5);

    constexpr uint32_t q_row_bytes = k_dim * q_elem_bytes;     // Q row (bf16)
    constexpr uint32_t k_row_bytes = k_dim * kv_elem_bytes;    // K row (native dtype: fp8 or bf16)
    constexpr uint32_t idx_row_bytes = topk * idx_elem_bytes;  // one index row

    Noc noc;
    experimental::CB q_cb(cb_q_rm), k_cb(cb_k_rm), mask_part_cb(cb_mask_part), idx_cb(cb_idx), ctrl_cb(cb_ctrl);
    experimental::CB kreq_cb(CB_KREQ), kack_cb(CB_KACK);  // dual-NoC K-gather handoff to the writer
    const auto q = TensorAccessor(q_args, q_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    // Reader-internal scratch for one token's index row (reserved once, reused).
    idx_cb.reserve_back(1);
    const uint32_t idx_l1 = idx_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        // Q: H head-rows -> cb_q_rm.  Q is [1,H,S,576] row-major: row (h,tok) = page h*S+tok.
        q_cb.reserve_back(H);
        {
            DeviceZoneScopedN("rdr_Q");
            for (uint32_t h = 0; h < H; ++h) {
                noc.async_read(q, q_cb, q_row_bytes, {.page_id = h * S + tok}, {.offset_bytes = h * q_row_bytes});
            }
            noc.async_read_barrier();
        }
        q_cb.push_back(H);

        // indices row for this token (page = tok) -> idx_cb scratch
        noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = tok}, {.offset_bytes = 0});
        noc.async_read_barrier();

        // binary-search the first sentinel -> nv (valid-key count); only ceil(nv/k_chunk) chunks are active.
        uint32_t nv = topk;
        {
            uint32_t lo = 0, hi = topk;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (idx_ptr[mid] == sentinel) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            nv = lo == 0 ? 1 : lo;  // contract guarantees >=1 valid; guard anyway
        }
        const uint32_t n_active = (nv + k_chunk - 1) / k_chunk;

        // hand the active chunk count + valid-key count to compute (it can't issue DRAM reads to derive
        // them). nv lets compute place the boundary mask (which key tiles of the last chunk to -inf).
        ctrl_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
            cp[0] = n_active;
            cp[1] = nv;
        }
        ctrl_cb.push_back(1);

        for (uint32_t chunk = 0; chunk < n_active; ++chunk) {
            const uint32_t base = chunk * k_chunk;
            const uint32_t valid = (nv - base) < k_chunk ? (nv - base) : k_chunk;  // (0, k_chunk]

            // K chunk rows -> cb_k_rm: read the `valid` real keys, zero-fill the sentinel suffix.
            k_cb.reserve_back(k_chunk);
            {
                DeviceZoneScopedN("rdr_Kchunk");  // the per-chunk indexed K gather (dominant cost)
                // Dual-NoC split gather: the K-row response data is the op's bottleneck (NoC-link bound, not
                // DRAM-controller bound), so we spread it across BOTH NoCs. The reader (this NoC) gathers
                // rows [half,valid); it hands the writer rows [0,half) via cb_kreq, and the writer gathers
                // them on ITS NoC into the same reserved cb_k_rm L1 (shared on-core), then acks via cb_kack.
                // ~13% faster than single-NoC on prod-dense (same-session A/B; magnitude drifts with DRAM
                // contention state) since each link now carries half the gather traffic.
                const uint32_t half = valid >> 1;
                kreq_cb.reserve_back(1);
                {
                    volatile tt_l1_ptr uint32_t* rq =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                    rq[0] = base;
                    rq[1] = half;
                    rq[2] = (chunk == n_active - 1);  // writer ends the token's gather phase on the last chunk
                }
                kreq_cb.push_back(1);
                // Reader gathers its half [half, valid) on its NoC (depth-4 trid ring, shared helper).
                sparse_sdpa::trid_ring_gather(
                    noc, kv, k_cb, idx_ptr, base, half, valid, k_row_bytes, kv_batch_page_offset);
                kack_cb.wait_front(1);  // writer's half landed in cb_k_rm L1
                kack_cb.pop_front(1);
                // Zero-fill the sentinel suffix: masked-key scores become a defined 0 (compute adds -inf).
                if (valid < k_chunk) {
                    noc.async_write_zeros(k_cb, (k_chunk - valid) * k_row_bytes, {.offset_bytes = valid * k_row_bytes});
                    noc.write_zeros_l1_barrier();
                }
            }
            k_cb.push_back(k_chunk);

            // Boundary mask: only the LAST active chunk can have sentinels. Its key tile that straddles
            // `valid` (tile valid/32) needs cols >= valid%32 set to -inf; the tiles after it are fully
            // masked (compute bcast-adds the persistent cb_neginf there). Build the single partial-boundary
            // tile here only when the boundary lands mid-tile; row 0 only (compute adds it row-broadcast).
            if (chunk == n_active - 1 && (valid % tt::constants::TILE_WIDTH) != 0) {
                mask_part_cb.reserve_back(1);
                {
                    DeviceZoneScopedN("rdr_mask");  // build the single partial-boundary mask tile
                    const uint32_t b = valid % tt::constants::TILE_WIDTH;  // first masked col within the tile
                    volatile tt_l1_ptr uint16_t* p =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_part_cb.get_write_ptr());
                    for (uint32_t col = 0; col < tt::constants::TILE_WIDTH; ++col) {
                        const uint32_t off = (col < 16) ? col : (256 + col - 16);  // face0/face1 row 0
                        p[off] = (col >= b) ? neg_inf_bf16 : (uint16_t)0;
                    }
                }
                mask_part_cb.push_back(1);
            }
        }
    }
}

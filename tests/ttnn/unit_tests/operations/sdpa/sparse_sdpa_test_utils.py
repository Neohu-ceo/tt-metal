# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the sparse_sdpa tests (basic post-commit suite + nightly suite).

Correctness uses SMALL parametric shapes (the golden gathers sel[S,k,D]; the full production
640/2048/56320 shape is ~2.8 GiB and is only exercised by the perf-only test in the nightly suite).
"""

import torch

import ttnn
from models.demos.deepseek_v32.reference_cpu.sparse_sdpa_prefill import sparse_mla, MASKED_INDEX

K_DIM = 576
V_DIM = 512


def make_inputs(H, S, T, TOPK, n_valid_fn, seed=0):
    """Build (q, kv, indices) torch tensors matching the producer contract (tail-shaped sentinels)."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, S, K_DIM, generator=gen, dtype=torch.float32)
    kv = torch.randn(1, 1, T, K_DIM, generator=gen, dtype=torch.float32)
    indices = torch.full((1, 1, S, TOPK), MASKED_INDEX, dtype=torch.int64)
    for s in range(S):
        nv = max(1, min(TOPK, n_valid_fn(s)))
        perm = torch.randperm(T, generator=gen)[:nv]
        indices[0, 0, s, :nv] = perm
    return q, kv, indices


def golden(q, kv, indices, scale):
    # sparse_mla expects kvpe [T,576] and indices reshaped (it accepts [..,S,k]).
    return sparse_mla(q, kv[0, 0], indices.to(torch.int64), scale)  # [1,H,S,512]


def to_dev(t, device, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def run_op(
    q, kv, indices, device, k_chunk_size, compute_kernel_config=None, kv_dtype=ttnn.bfloat16, q_dtype=ttnn.bfloat16
):
    q_host = q.to(torch.bfloat16) if q_dtype == ttnn.bfloat16 else q.to(torch.float32)
    tt_q = to_dev(q_host, device, q_dtype)  # ttnn quantizes float -> fp8 when q_dtype is fp8_e4m3
    kv_host = kv.to(torch.bfloat16) if kv_dtype == ttnn.bfloat16 else kv.to(torch.float32)
    tt_kv = to_dev(kv_host, device, kv_dtype)  # ttnn quantizes float -> fp8 when kv_dtype is fp8_e4m3
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)
    scale = K_DIM**-0.5
    tt_out = ttnn.transformer.sparse_sdpa(
        tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=k_chunk_size, compute_kernel_config=compute_kernel_config
    )
    # Output dtype matches q. fp8 tensors can't be converted directly with to_torch, so typecast to bf16.
    if tt_out.dtype == ttnn.fp8_e4m3:
        tt_out = ttnn.typecast(tt_out, ttnn.bfloat16)
    return ttnn.to_torch(tt_out), scale


def pcc(out, golden_t):
    return torch.corrcoef(torch.stack([out.flatten().float(), golden_t.flatten().float()]))[0, 1].item()

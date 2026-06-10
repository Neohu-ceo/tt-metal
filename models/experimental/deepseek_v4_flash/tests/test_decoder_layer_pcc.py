# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-layer PCC test for the ttnn ``DeepSeekV4DecoderLayer`` (prefill).

Like ``test_attention_pcc.py`` this is a dual-interpreter test, but it runs on
the **real** V4-Flash checkpoint weights for one full decoder layer:

* **reference side** (run as ``__main__`` under the *system* interpreter):
  instantiates the HuggingFace ``transformers==5.8.1``
  ``DeepseekV4DecoderLayer`` for the requested layer, loads the real
  (dequantized) checkpoint weights for that layer into it via
  :class:`DeepseekV4WeightLoader` + the ``quant`` dequantizers, runs the
  reference forward over a random residual-stream stack, and dumps the RoPE
  tables, the exact additive mask, the input streams, and the reference output
  to a ``.pt`` bundle.
* **pytest side** (ttnn venv): builds the ttnn ``DeepSeekV4DecoderLayer`` from
  the *same* loader (so both sides see identical dequantized weights), runs it
  on the dumped inputs, and PCC-compares the updated residual-stream stack.

The two interpreters are kept apart for the same reason as the attention test:
the cached ``transformers==5.8.1`` (the only install shipping ``deepseek_v4``)
imports cleanly only under the system interpreter, while ttnn lives in the venv.
The ``__main__`` guard runs before the ttnn imports so the subprocess never
touches ttnn.

The routed experts live on device in bf16 (one layer fits the Blackhole DRAM),
so the only precision gap vs the fp32 reference is bf16 device arithmetic.

Set ``DEEPSEEK_V4_CACHE_DIR=<dir>`` to skip the slow weight loading on reruns:
the converted ttnn weight tiles are dumped/reused (the 256-expert dequant is
skipped entirely on a hit) and the HF reference bundle is cached too, so the
expensive system-python subprocess only runs once per (layer, batch, seq_len).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


# Cached transformers 5.8.1 (the only install with ``deepseek_v4``).
_CACHED_TRANSFORMERS = "/home/ttuser/.cache/uv/archive-v0/U5SPsIWJupLz-bDcPI13a"
_DEFAULT_MODEL_DIR = "/home/ttuser/models/hub/models--deepseek-ai--DeepSeek-V4-Flash"


# --------------------------------------------------------------------------- #
# Reference side (executed only as ``__main__`` under the system interpreter).
# --------------------------------------------------------------------------- #
def _reference_sliding_causal_mask(seq_len: int, sliding_window: int, dtype: torch.dtype) -> torch.Tensor:
    """Additive [1, 1, S, S] sliding-window causal mask (0 keep / min mask)."""
    i = torch.arange(seq_len).view(seq_len, 1)
    j = torch.arange(seq_len).view(1, seq_len)
    keep = (j <= i) & (i - j < sliding_window)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype).masked_fill(~keep, torch.finfo(dtype).min)
    return mask.view(1, 1, seq_len, seq_len)


def _reference_block_bias(
    position_ids: torch.Tensor, n_windows: int, compress_rate: int, dtype: torch.dtype
) -> torch.Tensor:
    """Additive [B, 1, S, n_windows] causal block bias over compressed windows.

    Query ``t`` may attend compressed entry ``w`` iff ``w < (t + 1) // compress_rate``
    — the degenerate CSA/HCA top-k for ``seq_len <= index_topk * compress_rate``.
    """
    batch, seq_len = position_ids.shape
    entry = torch.arange(n_windows).view(1, 1, 1, n_windows)
    threshold = ((position_ids + 1) // compress_rate).view(batch, 1, seq_len, 1)
    bias = torch.zeros(batch, 1, seq_len, n_windows, dtype=dtype)
    return bias.masked_fill(entry >= threshold, torch.finfo(dtype).min)


def _ref_load_layer_weights(layer, loader, quant, layer_idx: int) -> None:
    """Populate an HF ``DeepseekV4DecoderLayer`` state-dict from the checkpoint.

    Every parameter is fetched (and dequantized) from the loader by its HF name
    ``layers.<idx>.<key>``. The two *packed* expert params are assembled from the
    per-expert ``gate_proj`` / ``up_proj`` / ``down_proj`` tensors.
    """

    def w(name: str) -> torch.Tensor:
        full = f"layers.{layer_idx}.{name}"
        return quant.dequantize_weight(loader.get_tensor(full), loader.get_scale(full)).to(torch.float32)

    state = layer.state_dict()
    n_experts = state["mlp.experts.gate_up_proj"].shape[0]
    sd: dict[str, torch.Tensor] = {}
    for key in state:
        if key == "mlp.experts.gate_up_proj":
            rows = [
                torch.cat([w(f"mlp.experts.{e}.gate_proj.weight"), w(f"mlp.experts.{e}.up_proj.weight")], dim=0)
                for e in range(n_experts)
            ]
            sd[key] = torch.stack(rows, dim=0)  # [E, 2I, H]
        elif key == "mlp.experts.down_proj":
            sd[key] = torch.stack(
                [w(f"mlp.experts.{e}.down_proj.weight") for e in range(n_experts)], dim=0
            )  # [E, H, I]
        else:
            sd[key] = w(key)
    layer.load_state_dict(sd, strict=True, assign=True)


def _reference_main() -> None:
    """Generate the gold-reference bundle. Args: <out_path> <layer_idx> [batch] [seq_len] [model_dir]."""
    import importlib.metadata as _md

    _orig_version = _md.version
    _md.version = lambda name: "0.22.0" if name.lower() == "tokenizers" else _orig_version(name)
    sys.path.insert(0, _CACHED_TRANSFORMERS)
    # weight_loader / quant are standalone (torch + safetensors only); import by path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tt"))

    import weight_loader as WL  # noqa: E402
    import quant as Q  # noqa: E402
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M  # noqa: E402
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402

    out_path = sys.argv[1]
    layer_idx = int(sys.argv[2])
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    model_dir = sys.argv[5] if len(sys.argv) > 5 else _DEFAULT_MODEL_DIR

    loader = WL.DeepseekV4WeightLoader(model_dir)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    dtype = torch.float32

    layer = M.DeepseekV4DecoderLayer(config, layer_idx).to(dtype).eval()
    _ref_load_layer_weights(layer, loader, Q, layer_idx)

    layer_type = config.layer_types[layer_idx]
    rope_layer_type = "main" if layer_type == "sliding_attention" else "compress"

    torch.manual_seed(1234)
    hc_mult = config.hc_mult
    streams = torch.randn(batch, seq_len, hc_mult, config.hidden_size, dtype=dtype)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    rotary = M.DeepseekV4RotaryEmbedding(config).to(dtype)
    cos_q, sin_q = rotary(streams, position_ids=position_ids, layer_type=rope_layer_type)
    position_embeddings = {
        "main": rotary(streams, position_ids=position_ids, layer_type="main"),
        "compress": rotary(streams, position_ids=position_ids, layer_type="compress"),
    }

    mask = _reference_sliding_causal_mask(seq_len, config.sliding_window, dtype).expand(batch, 1, seq_len, seq_len)

    bundle: dict = {}
    if layer_type != "sliding_attention":
        cr = config.compress_rates[layer_type]
        n_windows = seq_len // cr
        assert n_windows >= 1, f"seq_len {seq_len} too short for compress_rate {cr}"
        win_positions = (torch.arange(n_windows) * cr).unsqueeze(0).expand(batch, -1)
        cos_win, sin_win = rotary(streams, position_ids=win_positions, layer_type="compress")
        mask = torch.cat([mask, _reference_block_bias(position_ids, n_windows, cr, dtype)], dim=-1)
        bundle["cos_win"] = cos_win[0].contiguous()
        bundle["sin_win"] = sin_win[0].contiguous()
        bundle["n_windows"] = n_windows
        bundle["compress_rate"] = cr

    with torch.no_grad():
        output = layer(
            streams,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=mask,
            past_key_values=None,
        )

    bundle.update(
        {
            "streams": streams,
            "cos_q": cos_q[0].contiguous(),
            "sin_q": sin_q[0].contiguous(),
            "mask": mask,
            "output": output,
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "config": {
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "head_dim": config.head_dim,
                "qk_rope_head_dim": config.qk_rope_head_dim,
                "o_groups": config.o_groups,
                "o_lora_rank": config.o_lora_rank,
                "rms_norm_eps": config.rms_norm_eps,
                "layer_types": list(config.layer_types),
                "compress_rates": dict(config.compress_rates),
                "num_local_experts": config.n_routed_experts,
                "num_experts_per_tok": config.num_experts_per_tok,
                "moe_intermediate_size": config.moe_intermediate_size,
                "routed_scaling_factor": config.routed_scaling_factor,
                "swiglu_limit": config.swiglu_limit,
                "hc_mult": config.hc_mult,
                "hc_sinkhorn_iters": config.hc_sinkhorn_iters,
                "hc_eps": config.hc_eps,
            },
        }
    )
    torch.save(bundle, out_path)
    print(f"REFERENCE_OK layer={layer_idx} ({layer_type}) -> {out_path}")


if __name__ == "__main__":
    _reference_main()
    raise SystemExit(0)


# --------------------------------------------------------------------------- #
# pytest side (ttnn venv).
# --------------------------------------------------------------------------- #
import shutil  # noqa: E402
import subprocess  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
from loguru import logger  # noqa: E402

import ttnn  # noqa: E402
from models.common.utility_functions import comp_allclose, comp_pcc  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.deepseek_v4_flash import (  # noqa: E402
    DeepSeekV4DecoderLayer,
    DeepSeekV4PreloadedExperts,
    WeightCache,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.weight_loader import (  # noqa: E402
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)


_SYSTEM_PYTHON = (
    "/usr/bin/python3" if Path("/usr/bin/python3").exists() else (shutil.which("python3") or sys.executable)
)
_THIS_FILE = str(Path(__file__).resolve())
_MASK_NEG = -1.0e9
PCC_THRESHOLD = 0.98
# Opt-in on-disk cache (ttnn weight tiles + HF reference bundles). ``None`` keeps
# caching off so every run reloads from the checkpoint (the default behaviour).
_CACHE_DIR = os.environ.get("DEEPSEEK_V4_CACHE_DIR")


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(Path(_DEFAULT_MODEL_DIR))
    except FileNotFoundError:
        return False
    return True


def _reference_path(tmp_path: Path, name: str) -> tuple[Path, bool]:
    """``(path, needs_generation)`` for the reference bundle.

    With the cache enabled the bundle is kept under ``$DEEPSEEK_V4_CACHE_DIR/ref``
    and reused across runs (the slow HF subprocess is skipped if it exists).
    """
    if _CACHE_DIR:
        d = Path(_CACHE_DIR) / "ref"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.pt"
        return p, not p.is_file()
    return tmp_path / f"{name}.pt", True


def _weight_cache(layer_idx: int):
    """Per-layer :class:`WeightCache` for the ttnn weights, or ``None`` (off)."""
    if not _CACHE_DIR:
        return None
    return WeightCache(os.path.join(_CACHE_DIR, "ttnn")).sub(f"layers.{layer_idx}")


def _generate_reference(out_path: Path, layer_idx: int, batch: int, seq_len: int) -> bool:
    proc = subprocess.run(
        [_SYSTEM_PYTHON, _THIS_FILE, str(out_path), str(layer_idx), str(batch), str(seq_len), _DEFAULT_MODEL_DIR],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning(f"reference generation failed for layer {layer_idx}:\n{proc.stderr[-3000:]}")
        return False
    return out_path.is_file()


def _w(loader: DeepseekV4WeightLoader, name: str):
    """Lazy fetch + dequantize: returns a thunk producing the fp32 tensor.

    Deferring the read lets a populated tile cache skip it entirely -- the
    consuming module only calls the thunk when the converted weight is missing.
    """
    return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))


def _attn_keys(layer_type: str) -> list[str]:
    keys = [
        "q_a_proj.weight",
        "q_a_norm.weight",
        "q_b_proj.weight",
        "kv_proj.weight",
        "kv_norm.weight",
        "o_a_proj.weight",
        "o_b_proj.weight",
        "sinks",
    ]
    if layer_type != "sliding_attention":
        keys += [
            "compressor.kv_proj.weight",
            "compressor.gate_proj.weight",
            "compressor.kv_norm.weight",
            "compressor.position_bias",
        ]
    return keys


def _build_layer_weights(loader: DeepseekV4WeightLoader, layer_idx: int, layer_type: str) -> dict:
    """HF-relative-keyed (dequantized) weight dict the ttnn decoder layer expects."""
    weights: dict[str, torch.Tensor] = {}
    for k in _attn_keys(layer_type):
        weights[f"self_attn.{k}"] = _w(loader, f"layers.{layer_idx}.self_attn.{k}")
    for k in ("gate.weight", "gate.e_score_correction_bias"):
        weights[f"mlp.{k}"] = _w(loader, f"layers.{layer_idx}.mlp.{k}")
    for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        weights[f"mlp.shared_experts.{k}"] = _w(loader, f"layers.{layer_idx}.mlp.shared_experts.{k}")
    for hc in ("attn_hc", "ffn_hc"):
        for p in ("fn", "base", "scale"):
            weights[f"{hc}.{p}"] = _w(loader, f"layers.{layer_idx}.{hc}.{p}")
    for k in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        weights[k] = _w(loader, f"layers.{layer_idx}.{k}")
    return weights


def _expert_provider(loader: DeepseekV4WeightLoader, layer_idx: int):
    """``provider(e) -> (gate_up [2I, H], down [H, I])`` in bf16 (HF packed layout)."""

    def provider(e: int):
        base = f"layers.{layer_idx}.mlp.experts.{e}"
        gate = _w(loader, f"{base}.gate_proj.weight")()
        up = _w(loader, f"{base}.up_proj.weight")()
        down = _w(loader, f"{base}.down_proj.weight")()
        gate_up = torch.cat([gate, up], dim=0).to(torch.bfloat16)
        return gate_up, down.to(torch.bfloat16)

    return provider


def _to_tt(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@torch.no_grad()
@pytest.mark.parametrize("layer_idx", (4, 5))  # 4 = CSA + moe, 5 = HCA + moe
@pytest.mark.parametrize("seq_len", (256,))
@pytest.mark.parametrize("batch_size", (1,))
def test_decoder_layer_pcc(device, reset_seeds, tmp_path, layer_idx: int, batch_size: int, seq_len: int) -> None:
    ref_path, need_gen = _reference_path(tmp_path, f"decoder_layer_{layer_idx}_{batch_size}_{seq_len}")
    if need_gen and not _generate_reference(ref_path, layer_idx, batch_size, seq_len):
        pytest.skip(f"could not generate HF reference for layer {layer_idx}")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])
    layer_type = bundle["layer_type"]

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    cache = _weight_cache(layer_idx)
    weights = _build_layer_weights(loader, layer_idx, layer_type)
    # One layer of experts fits Blackhole DRAM in bf16, so the only precision gap
    # vs the fp32 reference is bf16 device arithmetic (no 4-bit quant gap). With a
    # cache the 256-expert dequant is skipped on a hit (the provider isn't called).
    experts = DeepSeekV4PreloadedExperts(
        cfg,
        _expert_provider(loader, layer_idx),
        device,
        dtype=ttnn.bfloat16,
        cache=cache.sub("mlp") if cache else None,
    )
    layer = DeepSeekV4DecoderLayer(cfg, layer_idx, weights, device, experts=experts, cache=cache)

    streams_tt = _to_tt(bundle["streams"], device)
    cos_full, sin_full = make_rope_table(bundle["cos_q"], bundle["sin_q"])
    cos_tt = _to_tt(cos_full, device)
    sin_tt = _to_tt(sin_full, device)
    neg_sin_tt = _to_tt(-sin_full, device)
    mask_tt = _to_tt(bundle["mask"].clamp_min(_MASK_NEG), device)

    cos_win_tt = sin_win_tt = None
    if layer_type != "sliding_attention":
        cw, sw = make_rope_table(bundle["cos_win"], bundle["sin_win"])
        cos_win_tt = _to_tt(cw, device)
        sin_win_tt = _to_tt(sw, device)

    out_tt = layer.forward(streams_tt, cos_tt, sin_tt, neg_sin_tt, mask_tt, cos_win=cos_win_tt, sin_win=sin_win_tt)
    out_torch = ttnn.to_torch(out_tt).reshape(bundle["output"].shape).to(torch.float32)

    reference = bundle["output"].to(torch.float32)
    passing, pcc_message = comp_pcc(reference, out_torch, pcc=PCC_THRESHOLD)
    logger.info(comp_allclose(reference, out_torch))
    logger.info(f"[decoder layer {layer_idx} ({layer_type})] PCC: {pcc_message}")

    assert passing, f"layer {layer_idx} decoder PCC < {PCC_THRESHOLD}: {pcc_message}"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive decode demo on the full ttnn ``DeepSeekV4Model``.

Unlike ``test_bf4_decode_demo.py`` (which wires the layers up by hand and runs a
single forward), this demo builds the *whole* model once via
:class:`DeepSeekV4Model` and then **generates many tokens** from a prompt, one at
a time.

The ttnn V4-Flash port is *prefill-only* (no KV cache; the compressors run in
their stateless single-shot mode). So "decode" here is repeated prefill: each
step re-runs the full model over the growing ``[prompt + generated]`` context,
reads the argmax of the last real position, appends it, and repeats. The RoPE
tables are produced once for the maximum length and sliced to the current length
each step (positions are a prefix, so a slice is exact); the additive masks are
rebuilt per length inside the model.

All the model weights live on device in ``bfloat4_b`` (block-float4) so the big
matmuls fit. The full 43-layer stack does **not** fit a single Blackhole's 32 GB;
cap the stack with ``DEEPSEEK_V4_DECODE_LAYERS=N`` (this is then a partial-stack
sanity check, not a correctness guarantee), and set ``DEEPSEEK_V4_CACHE_DIR`` to
reuse the converted ttnn weight tiles across runs.

Dual-interpreter, like the other V4 tests: the *system* subprocess only
tokenizes and emits the YaRN RoPE tables (no weights, no HF forward). Two
subcommands:

* ``ref  <out> [text] [model_dir]`` -> tokenize + dump RoPE tables / config.
* ``decode <json> [model_dir]``     -> load the tokenizer and print the decode.

Run it (ttnn venv)::

    DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    DEEPSEEK_V4_MAX_NEW_TOKENS=16 \\
    pytest -s models/experimental/deepseek_v4_flash/tests/test_full_model_decode_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


_DEFAULT_MODEL_DIR = "/home/smanoj/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash"
_DEFAULT_TEXT = "The capital of France is"


def _pad_to_tile(n: int) -> int:
    return ((n + 31) // 32) * 32


# --------------------------------------------------------------------------- #
# Reference / helper subprocesses (system interpreter, no ttnn).
# --------------------------------------------------------------------------- #
def _reference_main() -> None:
    """``ref <out_path> [text] [model_dir]``: tokenize + dump RoPE tables.

    The RoPE tables are emitted for the *maximum* generation length
    (``prompt + DEEPSEEK_V4_MAX_NEW_TOKENS``, padded to a tile multiple) so the
    pytest side can slice them to the current length at every decode step without
    re-invoking this subprocess.
    """
    import importlib.metadata as _md

    _orig_version = _md.version
    _md.version = lambda name: "0.22.0" if name.lower() == "tokenizers" else _orig_version(name)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tt"))

    import weight_loader as WL  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M  # noqa: E402
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402

    out_path = sys.argv[2]
    text = sys.argv[3] if len(sys.argv) > 3 else _DEFAULT_TEXT
    model_dir = sys.argv[4] if len(sys.argv) > 4 else _DEFAULT_MODEL_DIR
    max_new_tokens = int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "16"))

    loader = WL.DeepseekV4WeightLoader(model_dir)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    ids = tokenizer(text)["input_ids"]
    real_len = len(ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else config.eos_token_id

    # RoPE tables span the longest sequence we might decode (prompt + new tokens),
    # padded to a tile multiple so a slice to any shorter (also tile-padded)
    # length stays a valid prefix.
    max_seq = _pad_to_tile(real_len + max_new_tokens)

    position_ids = torch.arange(max_seq).unsqueeze(0)
    dummy = torch.zeros(1, max_seq, 1, dtype=dtype)
    rotary = M.DeepseekV4RotaryEmbedding(config).to(dtype)

    def half(cos_sin):
        cos, sin = cos_sin
        return cos[0].contiguous(), sin[0].contiguous()

    rope = {
        "main": half(rotary(dummy, position_ids=position_ids, layer_type="main")),
        "compress": half(rotary(dummy, position_ids=position_ids, layer_type="compress")),
        "win": {},
    }
    # One windowed-RoPE table per distinct compress-rate (CSA / HCA layers), at
    # the max number of windows; sliced to S // cr windows each step.
    for cr in sorted({int(v) for v in config.compress_rates.values()}):
        n_win = max_seq // cr
        win_pos = (torch.arange(n_win) * cr).unsqueeze(0)
        rope["win"][cr] = half(rotary(dummy, position_ids=win_pos, layer_type="compress"))

    cfg_dict = config.to_dict()
    num_hash_layers = int(cfg_dict.get("num_hash_layers", 3))
    mlp_layer_types = list(getattr(config, "mlp_layer_types", None) or cfg_dict.get("mlp_layer_types") or [])
    if not mlp_layer_types:
        mlp_layer_types = ["hash_moe"] * num_hash_layers + ["moe"] * (config.num_hidden_layers - num_hash_layers)

    torch.save(
        {
            "input_ids": ids,  # real (unpadded) prompt ids
            "real_len": real_len,
            "max_seq": max_seq,
            "max_new_tokens": max_new_tokens,
            "pad_id": pad_id,
            "text": text,
            "rope": rope,
            "config": {
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "head_dim": config.head_dim,
                "qk_rope_head_dim": config.qk_rope_head_dim,
                "o_groups": config.o_groups,
                "o_lora_rank": config.o_lora_rank,
                "rms_norm_eps": config.rms_norm_eps,
                "layer_types": list(config.layer_types),
                "mlp_layer_types": mlp_layer_types,
                "compress_rates": {k: int(v) for k, v in config.compress_rates.items()},
                "sliding_window": config.sliding_window,
                "num_hidden_layers": config.num_hidden_layers,
                "num_hash_layers": num_hash_layers,
                "num_local_experts": config.n_routed_experts,
                "num_experts_per_tok": config.num_experts_per_tok,
                "moe_intermediate_size": config.moe_intermediate_size,
                "routed_scaling_factor": config.routed_scaling_factor,
                "swiglu_limit": config.swiglu_limit,
                "hc_mult": config.hc_mult,
                "hc_sinkhorn_iters": config.hc_sinkhorn_iters,
                "hc_eps": config.hc_eps,
                "eos_token_id": config.eos_token_id,
            },
        },
        out_path,
    )
    print(f"REFERENCE_OK real_len={real_len} max_seq={max_seq} -> {out_path}")


def _decode_main() -> None:
    """``decode <json_path> [model_dir]``: load the tokenizer and print the decode."""
    import importlib.metadata as _md

    _orig_version = _md.version
    _md.version = lambda name: "0.22.0" if name.lower() == "tokenizers" else _orig_version(name)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tt"))

    import weight_loader as WL  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    json_path = sys.argv[2]
    model_dir = sys.argv[3] if len(sys.argv) > 3 else _DEFAULT_MODEL_DIR
    with open(json_path) as fh:
        data = json.load(fh)

    loader = WL.DeepseekV4WeightLoader(model_dir)
    tok = AutoTokenizer.from_pretrained(loader.snapshot_dir)

    prompt_ids = data["prompt_ids"]
    generated = data["generated"]
    n_layers = data["n_layers"]

    print("=" * 78)
    print(f"Full-model decode demo  |  {n_layers} of {data['num_hidden_layers']} layers loaded")
    print("=" * 78)
    print(f"PROMPT       : {tok.decode(prompt_ids)!r}")
    print(f"GENERATED    : {tok.decode(generated)!r}   ({len(generated)} tokens)")
    print(f"FULL TEXT    : {tok.decode(prompt_ids + generated)!r}")
    print()
    print("Per-step generated tokens:")
    for i, tid in enumerate(generated):
        print(f"    step {i:3d}  id={tid:<7d} {tok.decode([tid])!r}")
    print("=" * 78)


if __name__ == "__main__":
    _mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if _mode == "ref":
        _reference_main()
    elif _mode == "decode":
        _decode_main()
    raise SystemExit(0)


# --------------------------------------------------------------------------- #
# pytest side (ttnn venv).
# --------------------------------------------------------------------------- #
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
from loguru import logger  # noqa: E402

import ttnn  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.deepseek_v4_flash import (  # noqa: E402
    DeepSeekV4Model,
    Linear,
    WeightCache,
)
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.weight_loader import (  # noqa: E402
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)


_SYSTEM_PYTHON = "/localdev/smanoj/metal_1/python_env/bin/python"
_THIS_FILE = str(Path(__file__).resolve())
_WEIGHT_DTYPE = ttnn.bfloat4_b
_CACHE_DIR = os.environ.get("DEEPSEEK_V4_CACHE_DIR", "../cache")


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(Path(_DEFAULT_MODEL_DIR))
    except FileNotFoundError:
        return False
    return True


def _w(loader: DeepseekV4WeightLoader, name: str):
    """Lazy (dequantized) fetch -> thunk (a populated tile cache skips the read)."""
    return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))


def _generate_reference(out_path: Path, text: str) -> bool:
    proc = subprocess.run(
        [_SYSTEM_PYTHON, _THIS_FILE, "ref", str(out_path), text, _DEFAULT_MODEL_DIR],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning(f"reference generation failed:\n{proc.stderr[-3000:]}")
        return False
    return out_path.is_file()


def _slice_rope(rope: dict, seq_len: int) -> dict:
    """Slice the max-length RoPE bundle down to ``seq_len`` (positions are a prefix)."""
    out = {
        "main": tuple(t[:seq_len].contiguous() for t in rope["main"]),
        "compress": tuple(t[:seq_len].contiguous() for t in rope["compress"]),
        "win": {},
    }
    for cr, (cw, sw) in rope["win"].items():
        n_win = seq_len // cr
        out["win"][cr] = (cw[:n_win].contiguous(), sw[:n_win].contiguous())
    return out


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)  # heavy: bf4 conversion of every expert + many decode steps
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("text", (_DEFAULT_TEXT,))
def test_full_model_decode_demo(mesh_device, reset_seeds, text: str) -> None:
    # --- reference bundle (tokenizer + max-length RoPE tables) --------------- #
    if _CACHE_DIR:
        ref_dir = Path(_CACHE_DIR) / "full_decode" / "ref"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_path = ref_dir / "decode_demo.pt"
    else:
        ref_path = Path(tempfile.mkdtemp()) / "decode_demo.pt"
    # Always regenerate: the tables depend on the prompt + max_new_tokens.
    if not _generate_reference(ref_path, text):
        pytest.skip("could not generate tokenizer / RoPE reference")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])
    prompt_ids: list[int] = list(bundle["input_ids"])
    pad_id = bundle["pad_id"]
    max_seq = bundle["max_seq"]
    max_new_tokens = bundle["max_new_tokens"]
    eos_id = cfg.eos_token_id

    max_layers = int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", cfg.num_hidden_layers))
    max_layers = min(max_layers, cfg.num_hidden_layers)

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    top_cache = WeightCache(os.path.join(_CACHE_DIR, "full_decode", "ttnn")) if _CACHE_DIR else None

    require_cache = False

    # --- build the full model + lm_head once -------------------------------- #
    model = DeepSeekV4Model(
        cfg,
        loader,
        mesh_device,
        cache=top_cache,
        weight_dtype=_WEIGHT_DTYPE,
        max_layers=max_layers,
        use_submeshes=True,
        require_cache=require_cache,
    )
    lm_head = Linear(
        _w(loader, "lm_head.weight"),
        model.last_device,
        top_cache.file("lm_head") if top_cache else None,
        dtype=_WEIGHT_DTYPE,
    )
    logger.info(f"built DeepSeekV4Model with {model.num_layers}/{cfg.num_hidden_layers} layers")

    # --- autoregressive decode loop (repeated prefill, no KV cache) --------- #
    generated: list[int] = []
    for step in range(max_new_tokens):
        ids = prompt_ids + generated
        real_len = len(ids)
        seq_len = _pad_to_tile(real_len)
        if seq_len > max_seq:  # ran past the precomputed RoPE span
            logger.warning(f"hit max RoPE length {max_seq}; stopping at {len(generated)} tokens")
            break
        padded = ids + [pad_id] * (seq_len - real_len)
        input_ids = torch.tensor(padded, dtype=torch.long).unsqueeze(0)

        hidden = model(input_ids, _slice_rope(bundle["rope"], seq_len))  # [1, S, D]
        logits = lm_head(hidden)  # [1, S, vocab]
        logits_t = ttnn.to_torch(logits).reshape(seq_len, -1).float()
        next_id = int(logits_t[real_len - 1].argmax().item())

        generated.append(next_id)
        logger.info(f"step {step:3d}: token id {next_id}")
        if next_id == eos_id:
            logger.info("hit EOS; stopping")
            break

    assert generated, "no tokens were generated"

    # --- decode (system tokenizer subprocess) ------------------------------- #
    out_json = ref_path.with_suffix(".decode.json")
    with open(out_json, "w") as fh:
        json.dump(
            {
                "prompt_ids": prompt_ids,
                "generated": generated,
                "n_layers": model.num_layers,
                "num_hidden_layers": cfg.num_hidden_layers,
            },
            fh,
        )
    proc = subprocess.run(
        [_SYSTEM_PYTHON, _THIS_FILE, "decode", str(out_json), _DEFAULT_MODEL_DIR],
        capture_output=True,
        text=True,
    )
    logger.info(f"\n{proc.stdout}")
    if proc.returncode != 0:
        logger.warning(f"decode subprocess failed:\n{proc.stderr[-2000:]}")

    logger.info(
        f"generated {len(generated)} tokens with {model.num_layers}/{cfg.num_hidden_layers} layers (bf4, repeated prefill)"
    )

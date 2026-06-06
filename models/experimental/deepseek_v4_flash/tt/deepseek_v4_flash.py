import os
from typing import Any, Optional

import ttnn
import torch

from .weight_loader import DeepseekV4WeightLoader


class WeightCache:
    """On-disk cache namespace for converted ttnn weight tensors.

    Wraps a base directory plus a dotted name prefix. :meth:`file` builds a
    ``cache_file_name`` for :func:`ttnn.as_tensor` (the first load tilizes and
    dumps the tensor; later runs read it straight back, skipping re-conversion).
    A ``None`` path disables caching, so :meth:`file` returns ``None`` and
    ``as_tensor`` simply converts every time -- this is the default, keeping the
    modules' behaviour unchanged unless a caller opts in.

    Sub-modules get a namespaced child via :meth:`sub` so every weight maps to a
    unique, stable path mirroring the checkpoint hierarchy (e.g.
    ``layers.5.attn.q_a_proj``).
    """

    __slots__ = ("path", "prefix")

    def __init__(self, path: Optional[str] = None, prefix: str = ""):
        self.path = path
        self.prefix = prefix

    def sub(self, name: str) -> "WeightCache":
        prefix = f"{self.prefix}.{name}" if self.prefix else name
        return WeightCache(self.path, prefix)

    def _name(self, name: str) -> str:
        return f"{self.prefix}.{name}" if self.prefix else name

    def file(self, name: str) -> Optional[str]:
        if not self.path:
            return None
        return os.path.join(self.path, self._name(name).replace("/", "_"))

    def hit(self, name: str, dtype: ttnn.DataType, layout: ttnn.Layout = ttnn.TILE_LAYOUT) -> bool:
        """Whether a cached file already exists for ``name`` (matching the exact
        ``ttnn.as_tensor`` suffix), so callers can skip producing the torch
        tensor entirely on a cache hit."""
        base = self.file(name)
        if not base:
            return False
        return os.path.isfile(f"{base}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")


def _as_cache(cache: Optional[WeightCache]) -> WeightCache:
    """Normalise ``None`` to a disabled cache so call sites stay branch-free."""
    return cache if cache is not None else WeightCache()


def _load_weight(
    tensor: Optional[torch.Tensor],
    device: ttnn.MeshDevice,
    *,
    cache_file_name: Optional[str] = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> ttnn.Tensor:
    """``ttnn.as_tensor`` for a (static) weight, with optional disk caching.

    Equivalent to ``ttnn.from_torch(...)`` when ``cache_file_name`` is ``None``;
    otherwise the tilized tensor is dumped on first use and loaded back on later
    runs. ``tensor`` may be ``None`` only on a verified cache hit (see
    :meth:`WeightCache.hit`).
    """
    return ttnn.as_tensor(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_file_name,
    )


def to_ttnn_device(
    tensor: torch.Tensor,
    device: ttnn.MeshDevice,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    cache_file_name: Optional[str] = None,
) -> ttnn.Tensor:
    return _load_weight(tensor, device, cache_file_name=cache_file_name, layout=layout)


# ---------------------------------------------------------------------------- #
# DeepSeek-V4-Flash attention (prefill, ``past_key_values is None``)
#
# ttnn port of ``DeepseekV4Attention`` (and its CSA / HCA compressors) from
# ``modular_deepseek_v4.py``. Scope is *prefill only*: with no KV cache the
# compressors run in their stateless single-shot mode (compress every complete
# window of the input and drop the remainder), so the whole rolling-window /
# overlap / entry-count cache machinery collapses away.
#
# Layout conventions, matching the reference:
#   B = batch, S = query/seq length, H = num_attention_heads, Dh = head_dim,
#   Rd = qk_rope_head_dim (the trailing RoPE slice of each head).
# V4 is shared-KV MQA (one KV head broadcast to all query heads) and lays each
# head out as ``[nope | rope]`` with interleaved RoPE on the trailing ``Rd``.
# ---------------------------------------------------------------------------- #

# fp32 accumulation everywhere keeps the long (Dh=512) reductions and the
# softmax/RoPE chains from drifting under bf16; the per-layer PCC test needs it.
_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# Additive-mask "-inf": a finite bf16-representable floor. Masked logits feed
# ``exp(x - max)`` which underflows to 0 for both this and a true ``-inf``, but
# the finite value avoids ``inf - inf -> NaN`` if a whole row were masked.
_MASK_NEG = -1.0e9


class DeepSeekV4Module:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


def _interleaved_rotate_matrix(rope_dim: int) -> torch.Tensor:
    """Fixed ``[Rd, Rd]`` matrix ``R`` s.t. ``x @ R == rotate_half(x)``.

    GLM/V4 interleaved ``rotate_half`` maps each consecutive pair
    ``(x_{2p}, x_{2p+1}) -> (-x_{2p+1}, x_{2p})``. As a right-multiply that is a
    block-diagonal matrix of ``[[0, 1], [-1, 0]]`` blocks, which lets us express
    the rotation as a single on-device matmul instead of strided gathers.
    """
    r = torch.zeros(rope_dim, rope_dim, dtype=torch.float32)
    for p in range(rope_dim // 2):
        r[2 * p, 2 * p + 1] = 1.0
        r[2 * p + 1, 2 * p] = -1.0
    return r


def make_rope_table(cos_half: torch.Tensor, sin_half: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand half-sized rotary ``(cos, sin)`` to full ``Rd`` and shape ``[1,1,L,Rd]``.

    ``DeepseekV4RotaryEmbedding`` emits one entry per interleaved pair; the
    reference ``apply_rotary_pos_emb`` does ``repeat_interleave(2)`` before the
    rotation. We bake that into the host-side table (broadcast over batch/heads).
    """
    cos = cos_half.repeat_interleave(2, dim=-1)
    sin = sin_half.repeat_interleave(2, dim=-1)
    cos = cos.reshape(1, 1, cos.shape[-2], cos.shape[-1]).float()
    sin = sin.reshape(1, 1, sin.shape[-2], sin.shape[-1]).float()
    return cos, sin


class Linear(DeepSeekV4Module):
    """``nn.Linear`` (bias-free) as ``x @ Wᵀ`` for ttnn.

    ttnn ``linear`` computes ``a @ b`` with ``b`` shaped ``[in, out]``, so we
    store the torch ``[out, in]`` weight transposed.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        device: ttnn.MeshDevice,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.weight = _load_weight(
            weight.t().contiguous() if weight is not None else None,
            device,
            cache_file_name=cache_file_name,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(x, self.weight, compute_kernel_config=_HIFI4)


class DeepSeekV4RMSNorm(DeepSeekV4Module):
    """Weighted RMSNorm over the last dim (matches ``DeepseekV4RMSNorm``)."""

    def __init__(
        self, weight: torch.Tensor, eps: float, device: ttnn.MeshDevice, cache_file_name: Optional[str] = None
    ):
        self.weight = _load_weight(weight.reshape(1, 1, 1, -1), device, cache_file_name=cache_file_name)
        self.eps = eps

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)


def _rms_norm_unweighted(x: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    """Unweighted RMSNorm over the last dim (matches ``DeepseekV4UnweightedRMSNorm``)."""
    return ttnn.rms_norm(x, epsilon=eps)


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, rot: ttnn.Tensor, rope_dim: int) -> ttnn.Tensor:
    """Interleaved RoPE on the trailing ``rope_dim`` channels of ``x`` ([.., D]).

    ``cos`` / ``sin`` are ``[1,1,L,rope_dim]`` tables (broadcast over batch/heads);
    ``rot`` is the ``[rope_dim, rope_dim]`` ``rotate_half`` matrix. Leading "nope"
    channels pass through untouched.
    """
    shape = list(x.shape)
    d = shape[-1]
    if d == rope_dim:
        nope = None
        rope = x
    else:
        nope = ttnn.slice(x, [0, 0, 0, 0], [shape[0], shape[1], shape[2], d - rope_dim])
        rope = ttnn.slice(x, [0, 0, 0, d - rope_dim], shape)
    rotated = ttnn.add(
        ttnn.multiply(rope, cos), ttnn.multiply(ttnn.matmul(rope, rot, compute_kernel_config=_HIFI4), sin)
    )
    if nope is None:
        return rotated
    return ttnn.concat([nope, rotated], dim=-1)


class DeepSeekV4Embedding(DeepSeekV4Module):
    def __init__(
        self,
        weight_loader: DeepseekV4WeightLoader,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
    ):
        self.weight_loader = weight_loader
        self.device = device
        cache = _as_cache(cache)
        # ``ttnn.embedding`` expects a row-major weight table.
        cfn = cache.file("embed_tokens")
        embed = (
            None
            if cache.hit("embed_tokens", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            else weight_loader.get_tensor("embed_tokens.weight")
        )
        self.embedding_weight = to_ttnn_device(embed, device, layout=ttnn.ROW_MAJOR_LAYOUT, cache_file_name=cfn)

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.embedding(input_ids, self.embedding_weight, layout=ttnn.TILE_LAYOUT)


class DeepSeekV4Flash(DeepSeekV4Module):
    """Stub V4-Flash model wired up to the safetensors weight loader.

    Only the embedding table is materialised today; the rest of the model is
    a placeholder. The loader, however, can already serve every parameter in
    the checkpoint (see ``load_state_dict_torch``), so as more submodules are
    fleshed out they can be populated with one ``loader.get_tensor(...)`` per
    parameter.
    """

    def __init__(
        self,
        config: dict,
        weights_dir: str,
        device: ttnn.MeshDevice,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.weights_dir = weights_dir
        self.device = device
        self.weight_loader = DeepseekV4WeightLoader(weights_dir)
        # Converted-weight cache (opt-in): pass ``cache_dir`` to dump/reuse the
        # tilized ttnn tensors across runs (skipping re-conversion / dequant).
        # ``None`` keeps caching off, so weights are converted every time.
        self.cache = WeightCache(cache_dir)
        self.embed_tokens = DeepSeekV4Embedding(self.weight_loader, device, cache=self.cache)

    def load_weights(self) -> None:
        """Populate the model's submodules from the safetensors checkpoint.

        Currently fills only ``embed_tokens`` (the rest of the model is a
        stub); extend this as new submodules land.
        """
        embed = self.weight_loader.get_tensor("embed_tokens.weight")
        self.embed_tokens.weight = ttnn.from_torch(embed)

    def load_state_dict_torch(self, hf_names: list[str]) -> dict[str, torch.Tensor]:
        """Return a ``{hf_name: torch.Tensor}`` dict for the given names.

        Convenience for tests / reference comparisons that want the raw
        torch tensors keyed by HF-style names.
        """
        return {name: self.weight_loader.get_tensor(name) for name in hf_names}

    def forward(self, input_ids: ttnn.Tensor, attention_mask: ttnn.Tensor) -> ttnn.Tensor:
        input_embeddings = self.embed_tokens(input_ids)
        return input_embeddings


def _softmax_weighted_sum(kv: ttnn.Tensor, gate: ttnn.Tensor, window_axis: int) -> ttnn.Tensor:
    """``sum_w softmax(gate, axis=w) * kv`` over the window axis.

    Shared compressor pooling (``DeepseekV4*Compressor``): the gate logits are
    softmaxed over the per-window token axis and used to convex-combine the kv
    rows into one compressed entry per window.
    """
    weights = ttnn.softmax(gate, dim=window_axis)
    return ttnn.sum(ttnn.multiply(kv, weights), dim=window_axis)


class DeepSeekV4HCACompressor:
    """Heavily-Compressed-Attention compressor, stateless prefill mode.

    Compresses every complete window of ``compress_rate`` (m'=128) source tokens
    into a single softmax-gated KV entry, then RoPEs each entry at its window's
    absolute position. Returns ``compressed_kv`` shaped ``[B, 1, n_windows, Dh]``
    ready to concat onto the sliding KV axis.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device,
        rot,
        rope_dim: int,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.rope_dim = rope_dim
        self.rot = rot
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        cache = _as_cache(cache)
        self.kv_proj = Linear(
            weights["compressor.kv_proj.weight"], device, cache.file("compressor.kv_proj"), dtype=weight_dtype
        )
        self.gate_proj = Linear(
            weights["compressor.gate_proj.weight"], device, cache.file("compressor.gate_proj"), dtype=weight_dtype
        )
        self.kv_norm = DeepSeekV4RMSNorm(
            weights["compressor.kv_norm.weight"], self.eps, device, cache.file("compressor.kv_norm")
        )
        # position_bias: [compress_rate, head_dim] -> broadcast over [B, n_win].
        self.position_bias = _load_weight(
            weights["compressor.position_bias"].reshape(1, 1, self.compress_rate, self.head_dim),
            device,
            cache_file_name=cache.file("compressor.position_bias"),
        )

    def __call__(self, hidden: ttnn.Tensor, cos_win: ttnn.Tensor, sin_win: ttnn.Tensor) -> ttnn.Tensor | None:
        b, s, _ = hidden.shape
        cr = self.compress_rate
        n_win = s // cr
        if n_win == 0:
            return None
        usable = n_win * cr
        kv = self.kv_proj(hidden)
        gate = self.gate_proj(hidden)
        if usable != s:
            kv = ttnn.slice(kv, [0, 0, 0], [b, usable, self.head_dim])
            gate = ttnn.slice(gate, [0, 0, 0], [b, usable, self.head_dim])
        kv = ttnn.reshape(kv, [b, n_win, cr, self.head_dim])
        gate = ttnn.reshape(gate, [b, n_win, cr, self.head_dim])
        gate = ttnn.add(gate, self.position_bias)
        compressed = _softmax_weighted_sum(kv, gate, window_axis=2)  # [B, n_win, Dh]
        compressed = self.kv_norm(compressed)
        compressed = ttnn.reshape(compressed, [b, 1, n_win, self.head_dim])
        compressed = _apply_rope(compressed, cos_win, sin_win, self.rot, self.rope_dim)
        return compressed


class DeepSeekV4CSACompressor:
    """Compressed-Sparse-Attention compressor, stateless prefill mode.

    Like HCA but with the two-series Ca/Cb overlap scheme: each token projects to
    ``2*Dh`` (Ca = its contribution to the *next* window, Cb = to the *current*
    window). Compressed entry ``w`` pools window ``w-1``'s Ca slice with window
    ``w``'s Cb slice over a width-``2*compress_rate`` window. Window 0's Ca half
    is zero-kv / ``-inf``-gate (softmax weight 0), since there is no prior window
    in single-shot prefill.

    The CSA Lightning Indexer only affects *which* compressed entries each query
    may see (the ``block_bias``); for ``seq_len <= index_topk * compress_rate``
    its top-k selects every entry, so the block_bias reduces to plain causal
    masking over windows, which the caller builds on host. The compressed KV
    values themselves (this module's output) do not depend on the indexer.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device,
        rot,
        rope_dim: int,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.rope_dim = rope_dim
        self.rot = rot
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        cache = _as_cache(cache)
        self.kv_proj = Linear(
            weights["compressor.kv_proj.weight"], device, cache.file("compressor.kv_proj"), dtype=weight_dtype
        )
        self.gate_proj = Linear(
            weights["compressor.gate_proj.weight"], device, cache.file("compressor.gate_proj"), dtype=weight_dtype
        )
        self.kv_norm = DeepSeekV4RMSNorm(
            weights["compressor.kv_norm.weight"], self.eps, device, cache.file("compressor.kv_norm")
        )
        self.position_bias = _load_weight(
            weights["compressor.position_bias"].reshape(1, 1, self.compress_rate, 2 * self.head_dim),
            device,
            cache_file_name=cache.file("compressor.position_bias"),
        )

    def __call__(self, hidden: ttnn.Tensor, cos_win: ttnn.Tensor, sin_win: ttnn.Tensor) -> ttnn.Tensor | None:
        b, s, _ = hidden.shape
        cr = self.compress_rate
        dh = self.head_dim
        n_win = s // cr
        if n_win == 0:
            return None
        usable = n_win * cr
        kv = self.kv_proj(hidden)  # [B, S, 2*Dh]
        gate = self.gate_proj(hidden)
        if usable != s:
            kv = ttnn.slice(kv, [0, 0, 0], [b, usable, 2 * dh])
            gate = ttnn.slice(gate, [0, 0, 0], [b, usable, 2 * dh])
        kv = ttnn.reshape(kv, [b, n_win, cr, 2 * dh])
        gate = ttnn.reshape(gate, [b, n_win, cr, 2 * dh])
        gate = ttnn.add(gate, self.position_bias)

        ca, cb = ttnn.split(kv, dh, dim=3)
        ca_g, cb_g = ttnn.split(gate, dh, dim=3)

        # Shift Ca down one window: entry w sees window w-1's Ca; entry 0 sees a
        # zero-kv / -inf-gate filler (softmax weight 0).
        zeros = ttnn.from_torch(
            torch.zeros(b, 1, cr, dh), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        neg = ttnn.from_torch(
            torch.full((b, 1, cr, dh), _MASK_NEG), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        ca_prev_src = ttnn.slice(ca, [0, 0, 0, 0], [b, n_win - 1, cr, dh])
        cag_prev_src = ttnn.slice(ca_g, [0, 0, 0, 0], [b, n_win - 1, cr, dh])
        ca_prev = ttnn.concat([zeros, ca_prev_src], dim=1)
        cag_prev = ttnn.concat([neg, cag_prev_src], dim=1)

        new_kv = ttnn.concat([ca_prev, cb], dim=2)  # [B, n_win, 2*cr, Dh]
        new_gate = ttnn.concat([cag_prev, cb_g], dim=2)
        compressed = _softmax_weighted_sum(new_kv, new_gate, window_axis=2)  # [B, n_win, Dh]
        compressed = self.kv_norm(compressed)
        compressed = ttnn.reshape(compressed, [b, 1, n_win, dh])
        compressed = _apply_rope(compressed, cos_win, sin_win, self.rot, self.rope_dim)
        return compressed


_COMPRESSORS = {
    "compressed_sparse_attention": DeepSeekV4CSACompressor,
    "heavily_compressed_attention": DeepSeekV4HCACompressor,
}


class DeepSeekV4Attention(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4Attention`` (prefill, no KV cache).

    Construct from a ``config`` (the HF ``DeepseekV4Config`` or any object
    exposing the same attributes), the layer's torch ``weights`` (HF-named
    ``state_dict`` entries), and a device. ``forward`` consumes pre-built RoPE
    tables + additive attention mask (see :func:`make_rope_table`); these are
    inputs because the rotary embedding and causal/sliding mask are owned by the
    surrounding model in the reference, not by the attention block.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        self.scaling = self.head_dim**-0.5
        cache = _as_cache(cache)

        self.q_a_proj = Linear(weights["q_a_proj.weight"], device, cache.file("q_a_proj"), dtype=weight_dtype)
        self.q_a_norm = DeepSeekV4RMSNorm(weights["q_a_norm.weight"], self.eps, device, cache.file("q_a_norm"))
        self.q_b_proj = Linear(weights["q_b_proj.weight"], device, cache.file("q_b_proj"), dtype=weight_dtype)
        self.kv_proj = Linear(weights["kv_proj.weight"], device, cache.file("kv_proj"), dtype=weight_dtype)
        self.kv_norm = DeepSeekV4RMSNorm(weights["kv_norm.weight"], self.eps, device, cache.file("kv_norm"))
        self.o_b_proj = Linear(weights["o_b_proj.weight"], device, cache.file("o_b_proj"), dtype=weight_dtype)

        # Grouped output projection (``DeepseekV4GroupedLinear``): block-diagonal
        # over o_groups. Store the per-group weight as [g, in_per_group, out_per_group]
        # so a single batched matmul (batch axis = group) does all groups at once.
        oa = weights["o_a_proj.weight"]  # [g*o_lora_rank, (H*Dh)//g]
        in_per_group = (self.num_heads * self.head_dim) // self.o_groups
        oa = oa.reshape(self.o_groups, self.o_lora_rank, in_per_group).transpose(1, 2).contiguous()
        self.o_a_weight = _load_weight(oa, device, cache_file_name=cache.file("o_a_proj"), dtype=weight_dtype)

        self.sinks_torch = weights["sinks"].reshape(1, self.num_heads, 1, 1).float()

        # The rotate-half matrix must stay precise (a bf4 rotation would corrupt RoPE).
        self.rot = _load_weight(_interleaved_rotate_matrix(self.rope_dim), device, cache_file_name=cache.file("rot"))

        compressor_cls = _COMPRESSORS.get(self.layer_type)
        self.compressor = (
            compressor_cls(config, weights, device, self.rot, self.rope_dim, cache=cache, weight_dtype=weight_dtype)
            if compressor_cls is not None
            else None
        )

    def _attention(self, q: ttnn.Tensor, kv: ttnn.Tensor, mask: ttnn.Tensor) -> ttnn.Tensor:
        """Eager attention with per-head learnable sinks (gpt-oss style).

        ``q`` is ``[B, H, S, Dh]``; ``kv`` (shared K=V) is ``[B, 1, Skv, Dh]``.
        The sink is an extra per-head logit column folded into the softmax
        denominator and then dropped — equivalently a rescale of the standard
        softmax by ``1 / (1 + exp(sink - m) / Σ)``.
        """
        b, h, s, _ = q.shape
        skv = kv.shape[2]
        k = ttnn.repeat(kv, ttnn.Shape([1, h, 1, 1]))  # broadcast single KV head to all heads
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=_HIFI4)  # [B, H, S, Skv]
        scores = ttnn.multiply(scores, self.scaling)
        scores = ttnn.add(scores, mask)

        sinks = ttnn.from_torch(
            self.sinks_torch.expand(b, h, s, 1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        row_max = ttnn.max(scores, dim=-1, keepdim=True)  # [B, H, S, 1]
        m = ttnn.maximum(row_max, sinks)
        exp_scores = ttnn.exp(ttnn.subtract(scores, m))
        denom = ttnn.add(ttnn.sum(exp_scores, dim=-1, keepdim=True), ttnn.exp(ttnn.subtract(sinks, m)))
        probs = ttnn.div(exp_scores, denom)
        attn = ttnn.matmul(probs, k, compute_kernel_config=_HIFI4)  # [B, H, S, Dh]
        return attn

    def _grouped_output(self, attn: ttnn.Tensor) -> ttnn.Tensor:
        """``DeepseekV4GroupedLinear`` (o_a) + ``o_b_proj``.

        ``attn`` is ``[B, S, H, Dh]``. Reshape to per-group feature blocks, run a
        batched matmul over the group axis, then mix groups back to hidden.
        """
        b, s, h, dh = attn.shape
        in_per_group = (h * dh) // self.o_groups
        x = ttnn.reshape(attn, [b * s, self.o_groups, in_per_group])
        x = ttnn.permute(x, [1, 0, 2])  # [g, B*S, in_per_group]
        y = ttnn.matmul(x, self.o_a_weight, compute_kernel_config=_HIFI4)  # [g, B*S, o_lora_rank]
        y = ttnn.permute(y, [1, 0, 2])  # [B*S, g, o_lora_rank]
        y = ttnn.reshape(y, [b, s, self.o_groups * self.o_lora_rank])
        return self.o_b_proj(y)

    def forward(
        self,
        hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        mask: ttnn.Tensor,
        cos_win: ttnn.Tensor | None = None,
        sin_win: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Prefill attention.

        Args:
            hidden: ``[B, S, hidden_size]`` input.
            cos/sin: ``[1,1,S,Rd]`` RoPE tables for this layer's rope type, at the
                query positions (used for Q, KV, and the output conjugate rotation).
            neg_sin: ``-sin`` table for the output conjugate (``-i``) rotation.
            mask: ``[B,1,S,Skv]`` additive attention mask. ``Skv == S`` for sliding
                layers; ``S + n_windows`` for CSA/HCA (sliding-causal cols followed
                by the compressed-window block_bias cols).
            cos_win/sin_win: ``[1,1,n_windows,Rd]`` RoPE tables at the compressor's
                window positions (required for CSA/HCA layers).
        """
        b, s, _ = hidden.shape
        h, dh = self.num_heads, self.head_dim

        q_residual = self.q_a_norm(self.q_a_proj(hidden))  # [B, S, q_lora_rank]
        q = self.q_b_proj(q_residual)  # [B, S, H*Dh]
        q = ttnn.reshape(q, [b, s, h, dh])
        q = ttnn.transpose(q, 1, 2)  # [B, H, S, Dh]
        q = _rms_norm_unweighted(q, self.eps)
        q = _apply_rope(q, cos, sin, self.rot, self.rope_dim)

        kv = self.kv_norm(self.kv_proj(hidden))  # [B, S, Dh]
        kv = ttnn.reshape(kv, [b, s, 1, dh])
        kv = ttnn.transpose(kv, 1, 2)  # [B, 1, S, Dh]
        kv = _apply_rope(kv, cos, sin, self.rot, self.rope_dim)

        if self.compressor is not None:
            compressed = self.compressor(hidden, cos_win, sin_win)
            if compressed is not None:
                kv = ttnn.concat([kv, compressed], dim=2)  # [B, 1, S + n_win, Dh]

        attn = self._attention(q, kv, mask)  # [B, H, S, Dh]

        # Conjugate (-i) RoPE on the output's rope slice (K=V picked up RoPE).
        attn = _apply_rope(attn, cos, neg_sin, self.rot, self.rope_dim)
        attn = ttnn.transpose(attn, 1, 2)  # [B, S, H, Dh]

        return self._grouped_output(attn)


# ---------------------------------------------------------------------------- #
# DeepSeek-V4-Flash Mixture-of-Experts (prefill)
#
# ttnn port of ``DeepseekV4SparseMoeBlock`` (and its ``DeepseekV4TopKRouter`` /
# ``DeepseekV4Experts`` / ``DeepseekV4MLP`` shared expert) from
# ``modular_deepseek_v4.py``. Scope is the standard top-k routed MoE block (the
# ``mlp_layer_types == "moe"`` path); the static ``hash_moe`` router is out of
# scope here (it only swaps the *which-experts* selection for a frozen
# ``tid2eid[input_ids]`` lookup, leaving the expert / shared-expert compute
# identical).
#
# Layout conventions, matching the reference:
#   B = batch, S = seq length, T = B*S flattened tokens, H = hidden_size,
#   E = num routed experts, I = moe_intermediate_size, k = num_experts_per_tok.
#
# The reference dispatches each token to its top-k experts and loops over the
# *hit* experts. We instead run a *dense* batched compute: every expert is
# evaluated for every token, then masked by the per-token routing weight (0 for
# unselected experts) and summed across the expert axis. This is the standard
# small-mesh ttnn MoE shape (cf. ``models/demos/gpt_oss``); it is mathematically
# identical to the gather/scatter reference because unselected experts get a
# routing weight of exactly 0.
# ---------------------------------------------------------------------------- #


class DeepSeekV4MLP(DeepSeekV4Module):
    """Dense SwiGLU MLP (matches ``DeepseekV4MLP`` / ``LlamaMLP``).

    Used as the always-on *shared expert*: ``down(silu(gate(x)) * up(x))`` with
    no clamp (the routed experts clamp; the shared expert does not).
    """

    def __init__(
        self,
        weights: dict,
        prefix: str,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        cache = _as_cache(cache)
        self.gate_proj = Linear(
            weights[f"{prefix}.gate_proj.weight"], device, cache.file(f"{prefix}.gate_proj"), dtype=weight_dtype
        )
        self.up_proj = Linear(
            weights[f"{prefix}.up_proj.weight"], device, cache.file(f"{prefix}.up_proj"), dtype=weight_dtype
        )
        self.down_proj = Linear(
            weights[f"{prefix}.down_proj.weight"], device, cache.file(f"{prefix}.down_proj"), dtype=weight_dtype
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.silu(self.gate_proj(x))
        return self.down_proj(ttnn.multiply(gate, self.up_proj(x)))


class DeepSeekV4TopKRouter(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4TopKRouter``.

    Produces a *dense* ``[1, 1, T, E]`` routing-weight tensor: ``sqrtsoftplus``
    of the gate logits gives per-expert scores; the top-k experts (by
    ``scores + e_score_correction_bias``) are selected via ``ttnn.topk`` +
    ``ttnn.scatter`` into a one-hot mask; the masked scores are renormalised to
    sum to 1 per token and scaled by ``routed_scaling_factor``. Unselected
    experts carry weight 0, which lets the dense expert compute drop them.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)
        self.gate = Linear(weights["gate.weight"], device, cache.file("gate"))
        self.e_score_correction_bias = _load_weight(
            weights["gate.e_score_correction_bias"].reshape(1, 1, 1, self.num_experts),
            device,
            cache_file_name=cache.file("gate.e_score_correction_bias"),
        )

    def forward(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """``x_flat`` is ``[1, 1, T, H]``; returns routing weights ``[1, 1, T, E]``."""
        logits = self.gate(x_flat)  # [1, 1, T, E]
        scores = ttnn.sqrt(ttnn.softplus(logits))
        biased = ttnn.add(scores, self.e_score_correction_bias)

        # Top-k selection -> one-hot mask. Scatter (rather than a >= threshold
        # compare) selects exactly k experts even if two scores collide under
        # bf16 rounding. Scatter wants ROW_MAJOR + a matching-rank index tensor.
        _, top_idx = ttnn.topk(biased, self.top_k, dim=-1)  # [1, 1, T, k]
        t = x_flat.shape[2]
        top_idx = ttnn.to_layout(top_idx, ttnn.ROW_MAJOR_LAYOUT)
        mask = ttnn.zeros([1, 1, t, self.num_experts], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.device)
        src = ttnn.ones([1, 1, t, self.top_k], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.device)
        mask = ttnn.scatter(mask, -1, top_idx, src)
        mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)

        # Weights are the *unbiased* scores gathered at the selected experts,
        # normalised per token, then scaled. Masking before the sum makes the
        # dense [1,1,T,E] tensor equal the reference's gathered/normalised one.
        selected = ttnn.multiply(scores, mask)
        denom = ttnn.add(ttnn.sum(selected, dim=-1, keepdim=True), 1.0e-20)
        return ttnn.multiply(ttnn.div(selected, denom), self.routed_scaling_factor)


class DeepSeekV4HashRouter(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4HashRouter`` (the first ``num_hash_layers`` MoE
    layers, paper §2.1).

    Expert *selection* is a frozen ``tid2eid[input_ids]`` lookup — a fixed
    token-id -> expert-id table — rather than a learned top-k argmax. The learned
    gate still produces the per-expert ``sqrtsoftplus`` scores that weight the
    selected experts; only the *which-experts* decision is static. As with
    :class:`DeepSeekV4TopKRouter` we emit a dense ``[1,1,T,E]`` weight tensor
    (selected experts carry their renormalised score, the rest are 0) so the same
    dense / preloaded expert compute consumes it.

    The selection mask is built host-side from ``input_ids`` (known at call time)
    — no on-device gather/scatter is needed since the table is fixed.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)
        self.gate = Linear(weights["gate.weight"], device, cache.file("gate"))
        # tid2eid [vocab, top_k]: frozen token-id -> expert-id table (host-side).
        self.tid2eid = weights["gate.tid2eid"].long()

    def forward(self, x_flat: ttnn.Tensor, input_ids: torch.Tensor) -> ttnn.Tensor:
        """``x_flat`` ``[1,1,T,H]`` and ``input_ids`` torch ``[..]`` (T tokens);
        returns dense routing weights ``[1,1,T,E]``."""
        logits = self.gate(x_flat)  # [1, 1, T, E]
        scores = ttnn.sqrt(ttnn.softplus(logits))
        t = x_flat.shape[2]

        # Static per-token expert selection -> host one-hot mask [1,1,T,E].
        eids = self.tid2eid[input_ids.reshape(-1).long()]  # [T, top_k]
        mask = torch.zeros(t, self.num_experts, dtype=torch.float32)
        mask.scatter_(1, eids, 1.0)
        mask_tt = ttnn.from_torch(
            mask.reshape(1, 1, t, self.num_experts), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        selected = ttnn.multiply(scores, mask_tt)
        denom = ttnn.add(ttnn.sum(selected, dim=-1, keepdim=True), 1.0e-20)
        return ttnn.multiply(ttnn.div(selected, denom), self.routed_scaling_factor)


def _swiglu_gate(gate_up: ttnn.Tensor, intermediate: int, limit: float) -> ttnn.Tensor:
    """Clamp + SiLU-GLU on a packed ``[..., 2I]`` gate_up output (matches
    ``DeepseekV4Experts._apply_gate``): ``silu(clamp(gate)) * clamp(up)``.

    Slices (rather than ``ttnn.split``) into the gate / up halves — the
    ``split_two_chunks`` kernel miscompiles for this layout on blackhole.
    """
    shape = list(gate_up.shape)
    end = list(shape)
    end[-1] = intermediate
    start = [0] * len(shape)
    gate = ttnn.slice(gate_up, start, end)
    start_up = [0] * len(shape)
    start_up[-1] = intermediate
    up = ttnn.slice(gate_up, start_up, shape)
    gate = ttnn.clamp(gate, min=None, max=limit)
    up = ttnn.clamp(up, min=-limit, max=limit)
    return ttnn.multiply(ttnn.silu(gate), up)


class DeepSeekV4PreloadedExperts(DeepSeekV4Module):
    """Routed-experts compute with all expert weights resident on device.

    The dense :class:`DeepSeekV4Experts` stacks every expert's weights in a
    single batched parameter — fine for the reduced test config, but the real
    V4-Flash layer has 256 experts of ``[2*2048, 4096]`` / ``[4096, 2048]``
    (~13 GB in bf16). This variant keeps every expert on device but stored as
    ``BFloat4_b`` (4-bit block-float), which fits comfortably (~3.5 GB) and is a
    natural match for these weights since the checkpoint already ships the routed
    experts in 4-bit (MXFP4). At init it pulls each expert's dequantized weights
    from the host ``provider`` once and uploads the transposed matmul-ready
    tensors; ``forward`` then runs purely on device with no per-step host
    transfers.

    ``provider(expert_idx) -> (gate_up [2I, H], down [H, I])`` returns host
    torch tensors (the HF packed layout: ``gate_up`` is ``cat([w_gate, w_up])``).
    Experts with zero total routing weight are skipped (matching the reference's
    ``hit`` set), so only the experts some token actually selected are computed.
    """

    def __init__(
        self,
        config,
        provider,
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat4_b,
        cache: Optional[WeightCache] = None,
    ):
        self.device = device
        self.num_experts = config.num_local_experts
        self.intermediate = config.moe_intermediate_size
        self.hidden = config.hidden_size
        self.limit = config.swiglu_limit
        cache = _as_cache(cache)

        # Upload every expert once, transposed to matmul-ready [H, 2I] / [I, H]
        # and stored in low precision on device. With caching enabled and a hit,
        # the provider (and its expensive dequant) is skipped entirely.
        self._gate_up: list[ttnn.Tensor] = []
        self._down: list[ttnn.Tensor] = []
        for e in range(self.num_experts):
            gu_name, dn_name = f"experts.{e}.gate_up", f"experts.{e}.down"
            need_torch = not (cache.hit(gu_name, dtype) and cache.hit(dn_name, dtype))
            gate_up_w, down_w = provider(e) if need_torch else (None, None)
            self._gate_up.append(
                _load_weight(
                    gate_up_w.t().contiguous() if gate_up_w is not None else None,
                    device,
                    cache_file_name=cache.file(gu_name),
                    dtype=dtype,
                )
            )
            self._down.append(
                _load_weight(
                    down_w.t().contiguous() if down_w is not None else None,
                    device,
                    cache_file_name=cache.file(dn_name),
                    dtype=dtype,
                )
            )

    def expert_matmul_weights(self, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Device-resident (BFloat4) weights for expert ``e`` as host fp32 torch
        ``(gate_up [H, 2I], down [I, H])`` — exactly what the on-device matmuls
        consume. Handy for parity references that must use the same quantized
        weights rather than the full-precision originals.
        """
        gate_up = ttnn.to_torch(self._gate_up[e]).reshape(self.hidden, 2 * self.intermediate).float()
        down = ttnn.to_torch(self._down[e]).reshape(self.intermediate, self.hidden).float()
        return gate_up, down

    def forward(self, x_flat: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """``x_flat`` ``[1,1,T,H]`` and ``routing_weights`` ``[1,1,T,E]``; returns ``[1,1,T,H]``."""
        t = x_flat.shape[2]
        # Read the (small) routing weights to host once to find which experts
        # were actually selected — skip the rest instead of looping all 256.
        rw_host = ttnn.to_torch(routing_weights).reshape(t, self.num_experts).float()
        hit = (rw_host.abs().sum(dim=0) > 0).nonzero().flatten().tolist()

        acc = None
        for e in hit:
            gate_up = ttnn.matmul(x_flat, self._gate_up[e], compute_kernel_config=_HIFI4)  # [1,1,T,2I]
            act = _swiglu_gate(gate_up, self.intermediate, self.limit)  # [1,1,T,I]
            down = ttnn.matmul(act, self._down[e], compute_kernel_config=_HIFI4)  # [1,1,T,H]

            w_e = ttnn.slice(routing_weights, [0, 0, 0, e], [1, 1, t, e + 1])  # [1,1,T,1]
            weighted = ttnn.multiply(down, w_e)
            acc = weighted if acc is None else ttnn.add(acc, weighted)

            ttnn.deallocate(gate_up)
            ttnn.deallocate(act)
            ttnn.deallocate(down)

        if acc is None:  # no expert selected (degenerate) -> zeros
            acc = ttnn.multiply(x_flat, 0.0)
        return acc


class DeepSeekV4Experts(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4Experts`` (GPT-OSS-style, bias-free, dense).

    Weights are stored per-expert as 3D parameters in the reference:
      ``gate_up_proj`` ``[E, 2I, H]`` and ``down_proj`` ``[E, H, I]``.
    We pre-transpose them to ``[1, E, H, 2I]`` / ``[1, E, I, H]`` so a single
    batched matmul (batch axis = expert) projects all experts at once.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.num_experts = config.num_local_experts
        self.intermediate = config.moe_intermediate_size
        self.hidden = config.hidden_size
        self.limit = config.swiglu_limit
        cache = _as_cache(cache)

        gate_up = weights["experts.gate_up_proj"]  # [E, 2I, H]
        gate_up = gate_up.transpose(1, 2).contiguous().unsqueeze(0)  # [1, E, H, 2I]
        self.gate_up_proj = _load_weight(gate_up, device, cache_file_name=cache.file("experts.gate_up_proj"))

        down = weights["experts.down_proj"]  # [E, H, I]
        down = down.transpose(1, 2).contiguous().unsqueeze(0)  # [1, E, I, H]
        self.down_proj = _load_weight(down, device, cache_file_name=cache.file("experts.down_proj"))

    def _apply_gate(self, gate_up: ttnn.Tensor) -> ttnn.Tensor:
        """Clamp + SiLU-GLU on the packed ``[..., 2I]`` gate_up output."""
        return _swiglu_gate(gate_up, self.intermediate, self.limit)

    def forward(self, x_flat: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """``x_flat`` ``[1,1,T,H]`` and ``routing_weights`` ``[1,1,T,E]``; returns ``[1,1,T,H]``."""
        e = self.num_experts
        t = x_flat.shape[2]
        # Broadcast tokens across the expert (batch) axis: [1, E, T, H].
        x_e = ttnn.reshape(x_flat, [1, 1, t, self.hidden])
        x_e = ttnn.repeat(x_e, ttnn.Shape([1, e, 1, 1]))

        gate_up = ttnn.matmul(x_e, self.gate_up_proj, compute_kernel_config=_HIFI4)  # [1, E, T, 2I]
        act = self._apply_gate(gate_up)  # [1, E, T, I]
        down = ttnn.matmul(act, self.down_proj, compute_kernel_config=_HIFI4)  # [1, E, T, H]

        # Per-(token, expert) routing weight -> [1, E, T, 1] to broadcast over H.
        rw = ttnn.permute(routing_weights, [0, 3, 2, 1])  # [1, E, T, 1]
        weighted = ttnn.multiply(down, rw)  # [1, E, T, H]
        return ttnn.experimental.fast_reduce_nc(weighted, dims=[1])  # [1, 1, T, H]


class DeepSeekV4SparseMoeBlock(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4SparseMoeBlock`` (standard ``moe`` layer).

    ``routed = experts(router(x)) ; return routed + shared_experts(x)``.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        experts=None,
        gate=None,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.hidden = config.hidden_size
        cache = _as_cache(cache)
        # ``gate`` may be injected (e.g. a :class:`DeepSeekV4HashRouter` for the
        # first ``num_hash_layers`` layers); otherwise the learned top-k router.
        self.gate = gate if gate is not None else DeepSeekV4TopKRouter(config, weights, device, cache=cache)
        self.is_hash = isinstance(self.gate, DeepSeekV4HashRouter)
        # ``experts`` may be injected (e.g. a device-resident BFloat4
        # :class:`DeepSeekV4PreloadedExperts` for the real 256-expert
        # checkpoint); otherwise build the dense stacked-weights variant.
        self.experts = experts if experts is not None else DeepSeekV4Experts(config, weights, device, cache=cache)
        self.shared_experts = DeepSeekV4MLP(weights, "shared_experts", device, cache=cache, weight_dtype=weight_dtype)

    def forward(self, hidden: ttnn.Tensor, input_ids: Optional[torch.Tensor] = None) -> ttnn.Tensor:
        """``hidden`` ``[B, S, H]`` -> ``[B, S, H]``. ``input_ids`` is required
        only for hash-routed layers (frozen ``tid2eid`` selection)."""
        b, s, h = hidden.shape
        x_flat = ttnn.reshape(hidden, [1, 1, b * s, h])

        if self.is_hash:
            routing_weights = self.gate(x_flat, input_ids)  # [1, 1, T, E]
        else:
            routing_weights = self.gate(x_flat)  # [1, 1, T, E]
        routed = self.experts(x_flat, routing_weights)  # [1, 1, T, H]
        routed = ttnn.reshape(routed, [b, s, h])

        shared = self.shared_experts(hidden)  # [B, S, H]
        return ttnn.add(routed, shared)


class DeepSeekV4HyperConnection(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4HyperConnection`` (Manifold-Constrained Hyper-
    Connections / mHC).

    Given the residual stream stack ``hidden_streams [B, S, H, D]`` (``H`` =
    ``hc_mult`` parallel streams, ``D`` = ``hidden_size``) it returns the triple
    ``(post, comb, collapsed)``:
      * ``collapsed [B, S, D]`` -- the ``pre``-weighted collapse of the streams
        into a single sequence (the sublayer input),
      * ``post [B, S, H]`` -- the sublayer-output placement weights
        (``2 * sigmoid(.)``),
      * ``comb [B, S, H, H]`` -- the stream-mixing matrix projected onto the
        doubly-stochastic manifold by ``hc_sinkhorn_iters`` Sinkhorn-Knopp steps.

    The learned ``fn`` / ``base`` / ``scale`` parameters are split host-side into
    their ``pre`` / ``post`` / ``comb`` parts (and ``fn`` run as three separate
    linears) so we never sub-tile-slice the packed ``(2+H)*H``-wide projection.
    See ``modular_deepseek_v4.py`` for the reference math.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        cache = _as_cache(cache)

        hc = self.hc
        fn = weights["fn"]  # [(2+H)*H, H*D]
        base = weights["base"]  # [(2+H)*H]
        scale = weights["scale"].flatten().tolist()  # 3 learned scalars

        # Split the packed projection into pre [H], post [H], comb [H*H] rows.
        self.fn_pre = Linear(fn[:hc], device, cache.file("fn_pre"))
        self.fn_post = Linear(fn[hc : 2 * hc], device, cache.file("fn_post"))
        self.fn_comb = Linear(fn[2 * hc : 2 * hc + hc * hc], device, cache.file("fn_comb"))
        self.pre_b = _load_weight(base[:hc].reshape(1, 1, 1, hc), device, cache_file_name=cache.file("pre_b"))
        self.post_b = _load_weight(base[hc : 2 * hc].reshape(1, 1, 1, hc), device, cache_file_name=cache.file("post_b"))
        self.comb_b = _load_weight(
            base[2 * hc : 2 * hc + hc * hc].reshape(1, 1, 1, hc * hc), device, cache_file_name=cache.file("comb_b")
        )
        self.pre_scale, self.post_scale, self.comb_scale = (float(scale[0]), float(scale[1]), float(scale[2]))

    def forward(self, hidden_streams: ttnn.Tensor):
        """``hidden_streams`` ``[B, S, H, D]`` -> ``(post [B,S,H], comb [B,S,H,H], collapsed [B,S,D])``."""
        b, s, hc, d = hidden_streams.shape
        t = b * s

        # Flatten streams to [1,1,T,H*D] and unweighted-RMSNorm over H*D.
        flat = ttnn.reshape(hidden_streams, [1, 1, t, hc * d])
        flat = _rms_norm_unweighted(flat, self.norm_eps)

        pre_w = self.fn_pre(flat)  # [1,1,T,H]
        post_w = self.fn_post(flat)  # [1,1,T,H]
        comb_w = self.fn_comb(flat)  # [1,1,T,H*H]

        # pre = sigmoid(w*scale + b) + eps ; post = 2*sigmoid(w*scale + b).
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(pre_w, self.pre_scale), self.pre_b)), self.eps)
        post = ttnn.multiply(ttnn.sigmoid(ttnn.add(ttnn.multiply(post_w, self.post_scale), self.post_b)), 2.0)

        # comb logits -> [1,T,H,H]; softmax over last dim, then Sinkhorn (alternate
        # row/col normalisation) onto the doubly-stochastic manifold.
        comb_logits = ttnn.add(ttnn.multiply(comb_w, self.comb_scale), self.comb_b)  # [1,1,T,H*H]
        comb_logits = ttnn.reshape(comb_logits, [1, t, hc, hc])
        comb = ttnn.add(ttnn.softmax(comb_logits, dim=-1), self.eps)
        comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), self.eps))  # column
        for _ in range(self.iters - 1):
            comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-1, keepdim=True), self.eps))  # row
            comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), self.eps))  # column

        # collapsed = sum_h pre[..,h] * hidden_streams[..,h,:]  (weighted stream sum).
        hs = ttnn.reshape(hidden_streams, [1, t, hc, d])
        pre_col = ttnn.reshape(pre, [1, t, hc, 1])
        collapsed = ttnn.sum(ttnn.multiply(hs, pre_col), dim=-2, keepdim=True)  # [1,T,1,D]

        post = ttnn.reshape(post, [b, s, hc])
        comb = ttnn.reshape(comb, [b, s, hc, hc])
        collapsed = ttnn.reshape(collapsed, [b, s, d])
        return post, comb, collapsed


class DeepSeekV4HyperHead(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4HyperHead`` (final HC-stream collapse).

    Collapses the ``hc_mult`` residual streams ``[B, S, H, D]`` into a single
    ``[B, S, D]`` sequence before the model's shared RMSNorm + ``lm_head``::

        flat  = unweighted_rmsnorm(streams.flatten(2))
        pre   = sigmoid(hc_fn @ flat * hc_scale + hc_base) + eps
        out   = (pre[..,None] * streams).sum(dim=2)

    ``weights`` keys: ``hc_fn`` ``[H, H*D]``, ``hc_base`` ``[H]``, ``hc_scale``
    (scalar). Unlike :class:`DeepSeekV4HyperConnection` there is no ``post`` /
    ``comb`` placement: the head only produces the collapsed sequence.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        cache = _as_cache(cache)

        self.fn = Linear(weights["hc_fn"], device, cache.file("hc_fn"))  # [H, H*D]
        self.base = _load_weight(
            weights["hc_base"].reshape(1, 1, 1, self.hc), device, cache_file_name=cache.file("hc_base")
        )
        self.scale = float(weights["hc_scale"].flatten().tolist()[0])

    def forward(self, hidden_streams: ttnn.Tensor) -> ttnn.Tensor:
        """``hidden_streams`` ``[B, S, H, D]`` -> ``[B, S, D]``."""
        b, s, hc, d = hidden_streams.shape
        t = b * s

        flat = ttnn.reshape(hidden_streams, [1, 1, t, hc * d])
        flat = _rms_norm_unweighted(flat, self.norm_eps)

        mixes = self.fn(flat)  # [1,1,T,H]
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(mixes, self.scale), self.base)), self.eps)

        hs = ttnn.reshape(hidden_streams, [1, t, hc, d])
        pre_col = ttnn.reshape(pre, [1, t, hc, 1])
        out = ttnn.sum(ttnn.multiply(hs, pre_col), dim=-2, keepdim=True)  # [1,T,1,D]
        return ttnn.reshape(out, [b, s, d])


def _strip_prefix(weights: dict, prefix: str) -> dict:
    """Sub-dict of ``weights`` whose keys start with ``prefix.`` (prefix stripped)."""
    p = f"{prefix}."
    return {k[len(p) :]: v for k, v in weights.items() if k.startswith(p)}


class DeepSeekV4DecoderLayer(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4DecoderLayer`` (prefill).

    The residual is a stack of ``hc_mult`` parallel streams kept in
    ``[B, S, H, D]`` (``H`` = ``hc_mult``, ``D`` = ``hidden_size``) throughout the
    block, mixed in/out by two :class:`DeepSeekV4HyperConnection` modules. For
    each sublayer (attention, then MoE) the matching HC collapses the streams
    into the sublayer input, and the sublayer output is folded back into the
    streams via the learned ``post`` placement weights plus the Sinkhorn
    ``comb`` stream-mixing matrix::

        post, comb, collapsed = hc(streams)
        out = sublayer(norm(collapsed))
        streams = post[..,None] * out[..,None,:] + (comb.T @ streams)

    ``comb`` is consumed *transposed* (mix over the first hc axis), matching the
    reference ``torch.matmul(comb.transpose(-1, -2), streams)``.

    ``weights`` keys mirror the HF decoder-layer param names: ``self_attn.*``,
    ``mlp.*``, ``attn_hc.{fn,base,scale}``, ``ffn_hc.{fn,base,scale}``,
    ``input_layernorm.weight``, ``post_attention_layernorm.weight``. RoPE tables
    and the additive mask are inputs (built by the surrounding model / test, per
    :func:`make_rope_table`), exactly as for :class:`DeepSeekV4Attention`.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        experts=None,
        gate=None,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        eps = config.rms_norm_eps
        cache = _as_cache(cache)

        self.self_attn = DeepSeekV4Attention(
            config,
            layer_idx,
            _strip_prefix(weights, "self_attn"),
            device,
            cache=cache.sub("self_attn"),
            weight_dtype=weight_dtype,
        )
        self.mlp = DeepSeekV4SparseMoeBlock(
            config,
            _strip_prefix(weights, "mlp"),
            device,
            experts=experts,
            gate=gate,
            cache=cache.sub("mlp"),
            weight_dtype=weight_dtype,
        )
        self.input_layernorm = DeepSeekV4RMSNorm(
            weights["input_layernorm.weight"], eps, device, cache.file("input_layernorm")
        )
        self.post_attention_layernorm = DeepSeekV4RMSNorm(
            weights["post_attention_layernorm.weight"], eps, device, cache.file("post_attention_layernorm")
        )
        self.attn_hc = DeepSeekV4HyperConnection(
            config, _strip_prefix(weights, "attn_hc"), device, cache=cache.sub("attn_hc")
        )
        self.ffn_hc = DeepSeekV4HyperConnection(
            config, _strip_prefix(weights, "ffn_hc"), device, cache=cache.sub("ffn_hc")
        )

    def _mix(
        self, post: ttnn.Tensor, comb: ttnn.Tensor, sublayer_out: ttnn.Tensor, streams: ttnn.Tensor
    ) -> ttnn.Tensor:
        """``post[..,None] * out[..,None,:] + comb.T @ streams`` -> new streams.

        ``post`` ``[B,S,H]``, ``comb`` ``[B,S,H,H]``, ``sublayer_out`` ``[B,S,D]``,
        ``streams`` ``[B,S,H,D]``; returns ``[B,S,H,D]``.
        """
        b, s, hc, d = streams.shape
        t = b * s

        # placement = post.unsqueeze(-1) * sublayer_out.unsqueeze(-2) -> [1,T,H,D].
        out = ttnn.reshape(sublayer_out, [1, t, 1, d])
        out = ttnn.repeat(out, ttnn.Shape([1, 1, hc, 1]))  # broadcast over the stream axis
        placement = ttnn.multiply(out, ttnn.reshape(post, [1, t, hc, 1]))

        # mix = matmul(comb.transpose(-1, -2), streams): sum over the FIRST hc axis.
        comb_t = ttnn.transpose(ttnn.reshape(comb, [1, t, hc, hc]), -2, -1)
        mixed = ttnn.matmul(comb_t, ttnn.reshape(streams, [1, t, hc, d]), compute_kernel_config=_HIFI4)

        return ttnn.reshape(ttnn.add(placement, mixed), [b, s, hc, d])

    def forward(
        self,
        hidden_streams: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        mask: ttnn.Tensor,
        cos_win: ttnn.Tensor | None = None,
        sin_win: ttnn.Tensor | None = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """``hidden_streams`` ``[B, S, hc_mult, D]`` -> updated streams ``[B, S, hc_mult, D]``."""
        post, comb, collapsed = self.attn_hc(hidden_streams)
        attn_out = self.self_attn(
            self.input_layernorm(collapsed), cos, sin, neg_sin, mask, cos_win=cos_win, sin_win=sin_win
        )
        hidden_streams = self._mix(post, comb, attn_out, hidden_streams)

        post, comb, collapsed = self.ffn_hc(hidden_streams)
        mlp_out = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        return self._mix(post, comb, mlp_out, hidden_streams)

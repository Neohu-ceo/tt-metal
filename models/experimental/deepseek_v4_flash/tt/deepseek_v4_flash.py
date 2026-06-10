import os
from typing import Any, Optional

import ttnn
import torch

from .quant import dequantize_weight
from .weight_loader import DeepseekV4WeightLoader
from loguru import logger


class _CachePath(str):
    """A ``str`` cache path that also carries the cache's ``require_cache`` flag.

    Lets the low-level loaders (:func:`_materialize`) see -- from the
    ``cache_file_name`` alone -- whether a cache miss should hard-fail instead of
    falling back to the (expensive) HF checkpoint read, without threading the
    flag through every ``Linear`` / ``RMSNorm`` constructor. It is a plain
    ``str`` everywhere else (``ttnn.as_tensor``, ``os.path``, f-strings).
    """

    __slots__ = ("require_cache",)

    def __new__(cls, value: str, require_cache: bool = False):
        s = super().__new__(cls, value)
        s.require_cache = require_cache
        return s


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

    ``require_cache`` turns a cache *miss* (for a tile-cached weight) into a hard
    error instead of a fallback HF-checkpoint read -- used to assert a populated
    cache is actually being consumed (see :class:`DeepSeekV4Model`). It rides
    along on the :class:`_CachePath` returned by :meth:`file` and is propagated to
    every child via :meth:`sub`.
    """

    __slots__ = ("path", "prefix", "require_cache")

    def __init__(self, path: Optional[str] = None, prefix: str = "", require_cache: bool = False):
        self.path = path
        self.prefix = prefix
        self.require_cache = require_cache

    def sub(self, name: str) -> "WeightCache":
        prefix = f"{self.prefix}.{name}" if self.prefix else name
        return WeightCache(self.path, prefix, self.require_cache)

    def require(self, flag: bool = True) -> "WeightCache":
        """A sibling cache (same path / prefix) with ``require_cache`` set."""
        return WeightCache(self.path, self.prefix, flag)

    def _name(self, name: str) -> str:
        return f"{self.prefix}.{name}" if self.prefix else name

    def file(self, name: str) -> Optional[str]:
        if not self.path:
            return None
        return _CachePath(os.path.join(self.path, self._name(name).replace("/", "_")), self.require_cache)

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
# Lazy weight resolution
#
# A "weight source" handed to a module may be either an eager ``torch.Tensor``
# or a zero-arg callable (a *thunk*) that produces one on demand. The thunk lets
# a populated on-disk cache short-circuit the (expensive) checkpoint read /
# dequant entirely: when the converted tile already exists for a weight, the
# source is never touched, so no tensor is pulled from the safetensors shards.
# ---------------------------------------------------------------------------- #
def _cache_hit_file(
    cache_file_name: Optional[str],
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> bool:
    """Whether the converted-tile cache file for ``cache_file_name`` exists.

    Mirrors the suffix :func:`ttnn.as_tensor` appends, so a leaf weight loader
    can tell -- from the cache path alone -- that a populated cache will serve
    the tensor and the (lazy) checkpoint read can be skipped entirely.
    """
    if not cache_file_name:
        return False
    return os.path.isfile(f"{cache_file_name}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")


def _materialize(
    weight,
    cache_file_name: Optional[str],
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> Optional[torch.Tensor]:
    """Resolve a (possibly lazy) weight source to a torch tensor, or ``None``.

    ``weight`` is either a ``torch.Tensor`` or a zero-arg callable returning one.
    On a cache hit the source is never touched (so a populated cache reads
    nothing from the checkpoint); otherwise the thunk is called -- or the tensor
    returned as-is for the eager path.

    When the cache path carries ``require_cache`` (a :class:`_CachePath`), a miss
    is a hard error instead of an HF-checkpoint fallback read.
    """
    if _cache_hit_file(cache_file_name, dtype, layout):
        return None
    if getattr(cache_file_name, "require_cache", False):
        raise RuntimeError(
            f"weight cache miss for {str(cache_file_name)!r} "
            f"(dtype={dtype.name}, layout={layout.name}) with require_cache=True; "
            "refusing to load from the HF checkpoint"
        )
    return weight() if callable(weight) else weight


def _memo(weight):
    """Wrap a tensor-or-thunk as a memoized zero-arg thunk (loads at most once).

    Used where one checkpoint tensor is sliced into several device weights (e.g.
    the packed hyper-connection projection) so the underlying source is read a
    single time even across multiple cache-miss slices.
    """
    if not callable(weight):
        return lambda: weight
    box: dict = {}

    def get():
        if "v" not in box:
            box["v"] = weight()
        return box["v"]

    return get


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
        weight,
        device: ttnn.MeshDevice,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        w = _materialize(weight, cache_file_name, dtype)
        self.weight = _load_weight(
            w.t().contiguous() if w is not None else None,
            device,
            cache_file_name=cache_file_name,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(x, self.weight, compute_kernel_config=_HIFI4)


class DeepSeekV4RMSNorm(DeepSeekV4Module):
    """Weighted RMSNorm over the last dim (matches ``DeepseekV4RMSNorm``)."""

    def __init__(self, weight, eps: float, device: ttnn.MeshDevice, cache_file_name: Optional[str] = None):
        w = _materialize(weight, cache_file_name, ttnn.bfloat16)
        self.weight = _load_weight(
            w.reshape(1, 1, 1, -1) if w is not None else None, device, cache_file_name=cache_file_name
        )
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
        hit = cache.hit("embed_tokens", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        if cache.require_cache and not hit:
            raise RuntimeError("weight cache miss for 'embed_tokens' with require_cache=True")
        embed = None if hit else weight_loader.get_tensor("embed_tokens.weight")
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
        pb = _materialize(weights["compressor.position_bias"], cache.file("compressor.position_bias"), ttnn.bfloat16)
        self.position_bias = _load_weight(
            pb.reshape(1, 1, self.compress_rate, self.head_dim) if pb is not None else None,
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
        pb = _materialize(weights["compressor.position_bias"], cache.file("compressor.position_bias"), ttnn.bfloat16)
        self.position_bias = _load_weight(
            pb.reshape(1, 1, self.compress_rate, 2 * self.head_dim) if pb is not None else None,
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
        # oa: [g*o_lora_rank, (H*Dh)//g] -> per-group [g, in_per_group, o_lora_rank].
        oa = _materialize(weights["o_a_proj.weight"], cache.file("o_a_proj"), weight_dtype)
        in_per_group = (self.num_heads * self.head_dim) // self.o_groups
        if oa is not None:
            oa = oa.reshape(self.o_groups, self.o_lora_rank, in_per_group).transpose(1, 2).contiguous()
        self.o_a_weight = _load_weight(oa, device, cache_file_name=cache.file("o_a_proj"), dtype=weight_dtype)

        # sinks live on host (folded into the softmax denominator), so there is
        # no tile cache for them -- always materialise.
        sinks = weights["sinks"]
        sinks = sinks() if callable(sinks) else sinks
        self.sinks_torch = sinks.reshape(1, self.num_heads, 1, 1).float()

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
        bias = _materialize(
            weights["gate.e_score_correction_bias"], cache.file("gate.e_score_correction_bias"), ttnn.bfloat16
        )
        self.e_score_correction_bias = _load_weight(
            bias.reshape(1, 1, 1, self.num_experts) if bias is not None else None,
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
        # tid2eid [vocab, top_k]: frozen token-id -> expert-id table (host-side,
        # no tile cache) -- always materialise.
        tid = weights["gate.tid2eid"]
        tid = tid() if callable(tid) else tid
        self.tid2eid = tid.long()

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
            if cache.require_cache and need_torch:
                raise RuntimeError(f"weight cache miss for routed expert {e} (gate_up/down) with require_cache=True")
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

        gate_up = _materialize(weights["experts.gate_up_proj"], cache.file("experts.gate_up_proj"), ttnn.bfloat16)
        if gate_up is not None:  # [E, 2I, H] -> [1, E, H, 2I]
            gate_up = gate_up.transpose(1, 2).contiguous().unsqueeze(0)
        self.gate_up_proj = _load_weight(gate_up, device, cache_file_name=cache.file("experts.gate_up_proj"))

        down = _materialize(weights["experts.down_proj"], cache.file("experts.down_proj"), ttnn.bfloat16)
        if down is not None:  # [E, H, I] -> [1, E, I, H]
            down = down.transpose(1, 2).contiguous().unsqueeze(0)
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
        # ``fn`` / ``base`` are one packed checkpoint tensor each, sliced into
        # pre [H] / post [H] / comb [H*H] parts; memoize so a cache miss reads
        # each source once across its slices. ``scale`` is 3 host scalars (no
        # tile cache), so it is always materialised.
        fn = _memo(weights["fn"])  # [(2+H)*H, H*D]
        base = _memo(weights["base"])  # [(2+H)*H]
        scale_src = weights["scale"]
        scale = (scale_src() if callable(scale_src) else scale_src).flatten().tolist()  # 3 learned scalars

        self.fn_pre = Linear(lambda: fn()[:hc], device, cache.file("fn_pre"))
        self.fn_post = Linear(lambda: fn()[hc : 2 * hc], device, cache.file("fn_post"))
        self.fn_comb = Linear(lambda: fn()[2 * hc : 2 * hc + hc * hc], device, cache.file("fn_comb"))
        self.pre_b = _load_weight(
            _materialize(lambda: base()[:hc].reshape(1, 1, 1, hc), cache.file("pre_b"), ttnn.bfloat16),
            device,
            cache_file_name=cache.file("pre_b"),
        )
        self.post_b = _load_weight(
            _materialize(lambda: base()[hc : 2 * hc].reshape(1, 1, 1, hc), cache.file("post_b"), ttnn.bfloat16),
            device,
            cache_file_name=cache.file("post_b"),
        )
        self.comb_b = _load_weight(
            _materialize(
                lambda: base()[2 * hc : 2 * hc + hc * hc].reshape(1, 1, 1, hc * hc),
                cache.file("comb_b"),
                ttnn.bfloat16,
            ),
            device,
            cache_file_name=cache.file("comb_b"),
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
        base_src = weights["hc_base"]
        self.base = _load_weight(
            _materialize(
                lambda: (base_src() if callable(base_src) else base_src).reshape(1, 1, 1, self.hc),
                cache.file("hc_base"),
                ttnn.bfloat16,
            ),
            device,
            cache_file_name=cache.file("hc_base"),
        )
        # hc_scale is a host scalar (no tile cache) -- always materialise.
        scale_src = weights["hc_scale"]
        self.scale = float((scale_src() if callable(scale_src) else scale_src).flatten().tolist()[0])

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


# ---------------------------------------------------------------------------- #
# DeepSeek-V4-Flash full model (prefill, ``past_key_values is None``)
#
# ttnn port of ``DeepseekV4Model`` from ``modular_deepseek_v4.py``. Wires the
# embedding, the stack of :class:`DeepSeekV4DecoderLayer`s, the final
# :class:`DeepSeekV4HyperHead` stream-collapse and the model's shared RMSNorm
# into one module driven straight off the safetensors checkpoint (via
# :class:`DeepseekV4WeightLoader` + the ``quant`` dequantizers).
#
# Differences from the reference, all forced by the prefill-only / on-device
# scope already established by the sub-modules in this file:
#   * The rotary tables are *inputs* (built host-side, e.g. by the YaRN rotary
#     in the system interpreter — see ``test_bf4_decode_demo.py``) rather than
#     produced by an owned ``DeepseekV4RotaryEmbedding``; ttnn has no rope-init.
#   * The additive sliding-window / compressed-window masks are built here on
#     host (mirroring ``create_sliding_window_causal_mask`` + the compressors'
#     ``block_bias``), since the device attention consumes a plain additive mask.
#   * Every layer's weights are resident at once (the reference also holds the
#     whole stack); on the real 43-layer checkpoint cap with ``max_layers`` /
#     a populated ``cache`` or run the per-layer load/free loop in the demo.
# ---------------------------------------------------------------------------- #


def _sliding_causal_mask(seq_len: int, sliding_window: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Additive ``[1, 1, S, S]`` sliding-window causal mask (0 keep / ``_MASK_NEG``)."""
    i = torch.arange(seq_len).view(seq_len, 1)
    j = torch.arange(seq_len).view(1, seq_len)
    keep = (j <= i) & (i - j < sliding_window)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype).masked_fill(~keep, _MASK_NEG)
    return mask.view(1, 1, seq_len, seq_len)


def _block_bias(seq_len: int, n_windows: int, compress_rate: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Additive ``[1, 1, S, n_windows]`` causal block bias over compressed windows.

    Query ``t`` may attend compressed entry ``w`` iff ``w < (t + 1) // compress_rate``
    — the degenerate CSA/HCA top-k for ``seq_len <= index_topk * compress_rate``.
    """
    position_ids = torch.arange(seq_len).unsqueeze(0)
    entry = torch.arange(n_windows).view(1, 1, 1, n_windows)
    threshold = ((position_ids + 1) // compress_rate).view(1, 1, seq_len, 1)
    bias = torch.zeros(1, 1, seq_len, n_windows, dtype=dtype)
    return bias.masked_fill(entry >= threshold, _MASK_NEG)


class DeepSeekV4Model(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4Model`` (prefill).

    Builds the embedding, the ``num_hidden_layers`` decoder stack, the final
    :class:`DeepSeekV4HyperHead` and the shared RMSNorm from the checkpoint, then
    runs the V4 forward: embed the ids, expand to the ``hc_mult`` residual-stream
    stack, run every decoder layer (building each layer's RoPE tables + additive
    mask from the supplied ``rope`` bundle), collapse the streams and normalise.

    ``rope`` matches the bundle emitted by the reference rotary (see
    ``test_bf4_decode_demo.py``)::

        rope["main"]    = (cos_half, sin_half)          # sliding layers
        rope["compress"]= (cos_half, sin_half)          # CSA / HCA layers
        rope["win"][cr] = (cos_half, sin_half)          # per compress-rate windows

    ``forward`` returns the model's ``last_hidden_state`` ``[B, S, hidden_size]``
    (the reference's pre-``lm_head`` output); apply an external ``lm_head``
    :class:`Linear` for logits.
    """

    def __init__(
        self,
        config,
        loader: DeepseekV4WeightLoader,
        full_device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        cache_dir: Optional[str] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat4_b,
        max_layers: Optional[int] = None,
        use_submeshes: bool = False,
        require_cache: bool = False,
    ):
        """Build the V4-Flash model off the checkpoint.

        Caching: pass either a pre-built ``cache`` :class:`WeightCache` or a
        ``cache_dir`` (the model builds ``WeightCache(cache_dir)`` and owns the
        per-layer ``layers.N`` / head namespacing internally, so callers no longer
        repeat the ``WeightCache(...).sub("layers.N")`` dance). ``None`` for both
        disables caching (every weight is converted from the checkpoint).

        ``require_cache=True`` asserts the converted-tile cache is fully populated:
        any tile-cached weight that would otherwise be (re)loaded from the HF
        checkpoint raises instead. The small host-side scalars (attention sinks,
        the HC ``scale`` triplets, the hash router's ``tid2eid`` table) and the
        locally-computed RoPE rotate matrix have no tile cache by design and are
        always materialised, so they are exempt.
        """
        self.config = config
        self.loader = loader
        self.weight_dtype = weight_dtype
        if cache is None and cache_dir is not None:
            cache = WeightCache(cache_dir)
        cache = _as_cache(cache)
        if require_cache:
            if not cache.path:
                raise ValueError(
                    "require_cache=True needs a populated cache; pass cache=WeightCache(dir) or cache_dir=..."
                )
            cache = cache.require(True)
        self.cache = cache
        self.require_cache = require_cache

        self.use_submeshes = use_submeshes
        self.num_submeshes = full_device.get_num_devices()
        if use_submeshes:
            logger.info(f"Using submeshes: {self.num_submeshes}")
            full_device.reshape(ttnn.MeshShape(1, full_device.get_num_devices()))
            self.submeshes = []
            for i in range(self.num_submeshes):
                self.submeshes.append(full_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, i)))
            self.first_device = self.submeshes[0]
            self.last_device = self.submeshes[-1]

            # Create socket pairs between submeshes for copying hidden_states .
            # One pair per (from_id, to_id) with from_id != to_id; reused for all forward passes.
            num_submeshes = full_device.get_num_devices()
            self.submesh_socket_pairs = {}
            socket_memconfig = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16 * 1024)
            for from_id in range(num_submeshes - 1):
                to_id = from_id + 1
                from_submesh = self.submeshes[from_id]
                to_submesh = self.submeshes[to_id]
                socket_connections = []
                for coord in ttnn.MeshCoordinateRange(from_submesh.shape):
                    socket_connections.append(
                        ttnn.SocketConnection(
                            ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
                            ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
                        )
                    )
                socket_config = ttnn.SocketConfig(socket_connections, socket_memconfig)
                sender_socket, receiver_socket = ttnn.create_socket_pair(from_submesh, to_submesh, socket_config)
                self.submesh_socket_pairs[(from_id, to_id)] = (sender_socket, receiver_socket)
        else:
            self.first_device = full_device
            self.last_device = full_device

        n = config.num_hidden_layers if max_layers is None else min(max_layers, config.num_hidden_layers)
        self.num_layers = n

        self.embed_tokens = DeepSeekV4Embedding(loader, self.first_device, cache=cache)

        self.layers: list[DeepSeekV4DecoderLayer] = []
        self.layer_devices: list[ttnn.MeshDevice] = []
        self.layers_per_device = 6  # 43 layers across 8 devices.
        for li in range(n):
            if self.use_submeshes:
                layer_device_id = li // self.layers_per_device
                current_device = self.submeshes[layer_device_id]
                logger.info(f"Layer {li} is on device {layer_device_id}")
            else:
                current_device = self.device
            self.layer_devices.append(current_device)
            layer_type = config.layer_types[li]
            is_hash = config.mlp_layer_types[li] == "hash_moe"
            layer_cache = cache.sub(f"layers.{li}")
            weights = self._build_layer_weights(li, layer_type, is_hash)
            gate = self._hash_gate(li) if is_hash else None
            experts = DeepSeekV4PreloadedExperts(
                config,
                self._expert_provider(li),
                current_device,
                dtype=weight_dtype,
                cache=layer_cache.sub("mlp"),
            )
            self.layers.append(
                DeepSeekV4DecoderLayer(
                    config,
                    li,
                    weights,
                    current_device,
                    experts=experts,
                    gate=gate,
                    cache=layer_cache,
                    weight_dtype=weight_dtype,
                )
            )

        self.hc_head = DeepSeekV4HyperHead(
            config,
            {
                "hc_fn": self._thunk("hc_head.hc_fn"),
                "hc_base": self._thunk("hc_head.hc_base"),
                "hc_scale": self._thunk("hc_head.hc_scale"),
            },
            self.last_device,
            cache=cache.sub("hc_head"),
        )
        self.norm = DeepSeekV4RMSNorm(
            self._thunk("norm.weight"), config.rms_norm_eps, self.last_device, cache.file("norm")
        )

    # -- weight plumbing (lazy dequant; a populated tile cache skips the read) -- #
    def _thunk(self, name: str):
        loader = self.loader
        return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))

    @staticmethod
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

    def _build_layer_weights(self, layer_idx: int, layer_type: str, is_hash: bool) -> dict:
        weights: dict = {}
        for k in self._attn_keys(layer_type):
            weights[f"self_attn.{k}"] = self._thunk(f"layers.{layer_idx}.self_attn.{k}")
        weights["mlp.gate.weight"] = self._thunk(f"layers.{layer_idx}.mlp.gate.weight")
        if not is_hash:
            weights["mlp.gate.e_score_correction_bias"] = self._thunk(
                f"layers.{layer_idx}.mlp.gate.e_score_correction_bias"
            )
        for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            weights[f"mlp.shared_experts.{k}"] = self._thunk(f"layers.{layer_idx}.mlp.shared_experts.{k}")
        for hc in ("attn_hc", "ffn_hc"):
            for p in ("fn", "base", "scale"):
                weights[f"{hc}.{p}"] = self._thunk(f"layers.{layer_idx}.{hc}.{p}")
        for k in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            weights[k] = self._thunk(f"layers.{layer_idx}.{k}")
        return weights

    def _expert_provider(self, layer_idx: int):
        def provider(e: int):
            base = f"layers.{layer_idx}.mlp.experts.{e}"
            gate = self._thunk(f"{base}.gate_proj.weight")()
            up = self._thunk(f"{base}.up_proj.weight")()
            down = self._thunk(f"{base}.down_proj.weight")()
            return torch.cat([gate, up], dim=0).float(), down.float()

        return provider

    def _hash_gate(self, layer_idx: int) -> DeepSeekV4HashRouter:
        weights = {
            "gate.weight": self._thunk(f"layers.{layer_idx}.mlp.gate.weight"),
            "gate.tid2eid": self.loader.get_tensor(f"layers.{layer_idx}.mlp.gate.tid2eid").long(),
        }
        if self.use_submeshes:
            this_device = self.submeshes[layer_idx // self.layers_per_device]
        else:
            this_device = self.first_device
        return DeepSeekV4HashRouter(self.config, weights, this_device)

    # -- per-layer RoPE tables / masks ------------------------------------------ #
    def _to_tt(self, t: torch.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _rope_tables(
        self, rope: dict, layer_type: str, compress_rate: Optional[int], cache: dict, device: ttnn.MeshDevice
    ):
        key = f'{"sliding" if layer_type == "sliding_attention" else compress_rate}_{device.id()}'
        if key in cache:
            return cache[key]
        cos, sin = rope["main"] if layer_type == "sliding_attention" else rope["compress"]
        cos_full, sin_full = make_rope_table(cos, sin)
        cos_tt = self._to_tt(cos_full, device)
        sin_tt = self._to_tt(sin_full, device)
        neg_sin_tt = self._to_tt(-sin_full, device)
        cos_win_tt = sin_win_tt = None
        if layer_type != "sliding_attention":
            cw, sw = rope["win"][compress_rate]
            cw, sw = make_rope_table(cw, sw)
            cos_win_tt = self._to_tt(cw, device)
            sin_win_tt = self._to_tt(sw, device)
        out = (cos_tt, sin_tt, neg_sin_tt, cos_win_tt, sin_win_tt)
        cache[key] = out
        return out

    def _mask(
        self, seq_len: int, layer_type: str, compress_rate: Optional[int], cache: dict, device: ttnn.MeshDevice
    ) -> ttnn.Tensor:
        key = "sliding" if layer_type == "sliding_attention" else compress_rate
        if key in cache:
            return cache[key]
        sliding = _sliding_causal_mask(seq_len, self.config.sliding_window)
        if layer_type == "sliding_attention":
            mask = sliding
        else:
            n_win = seq_len // compress_rate
            mask = torch.cat([sliding, _block_bias(seq_len, n_win, compress_rate)], dim=-1)
        mask_tt = self._to_tt(mask, device)
        cache[key] = mask_tt
        return mask_tt

    def forward(self, input_ids: torch.Tensor, rope: dict) -> ttnn.Tensor:
        """``input_ids`` torch ``[B, S]`` + host ``rope`` bundle -> ``[B, S, hidden]``."""
        seq_len = input_ids.shape[1]
        ids_tt = ttnn.from_torch(
            input_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.first_device
        )
        inputs_embeds = self.embed_tokens(ids_tt)  # [B, S, D]
        b, s, d = inputs_embeds.shape
        streams = ttnn.reshape(inputs_embeds, [b, s, 1, d])
        streams = ttnn.repeat(streams, ttnn.Shape([1, 1, self.config.hc_mult, 1]))  # [B, S, hc_mult, D]

        rope_cache: dict = {}
        mask_cache: dict = {}
        last_submesh_id = 0
        for li, layer in enumerate(self.layers):
            if self.use_submeshes:
                current_submesh_id = li // self.layers_per_device
                if current_submesh_id != last_submesh_id:
                    logger.info(
                        f"Copying hidden states from submesh {last_submesh_id} to submesh {current_submesh_id} for layer {li}"
                    )
                    streams = self._copy_hidden_states_between_submeshes(streams, last_submesh_id, current_submesh_id)
                this_device = self.submeshes[current_submesh_id]
            else:
                this_device = self.first_device
            layer_type = self.config.layer_types[li]
            compress_rate = None if layer_type == "sliding_attention" else self.config.compress_rates[layer_type]
            cos_tt, sin_tt, neg_sin_tt, cos_win_tt, sin_win_tt = self._rope_tables(
                rope, layer_type, compress_rate, rope_cache, this_device
            )
            mask_tt = self._mask(seq_len, layer_type, compress_rate, mask_cache, this_device)
            streams = layer.forward(
                streams,
                cos_tt,
                sin_tt,
                neg_sin_tt,
                mask_tt,
                cos_win=cos_win_tt,
                sin_win=sin_win_tt,
                input_ids=input_ids,
            )
            last_submesh_id = current_submesh_id

        return self.norm(self.hc_head(streams))

    def _copy_hidden_states_between_submeshes(self, hidden_states, from_submesh_id, to_submesh_id):
        """
        Copy hidden_states from one submesh to another using the pre-created socket pair.
        Sockets are created in the constructor and reused for every forward pass.
        """
        to_submesh = self.submeshes[to_submesh_id]
        from_submesh = self.submeshes[from_submesh_id]
        output_tensor = ttnn.allocate_tensor_on_device(hidden_states.spec, to_submesh)
        sender_socket, receiver_socket = self.submesh_socket_pairs[(from_submesh_id, to_submesh_id)]
        ttnn.experimental.send_async(hidden_states, sender_socket)
        ttnn.experimental.recv_async(output_tensor, receiver_socket)
        ttnn.synchronize_device(from_submesh)
        ttnn.synchronize_device(to_submesh)
        hidden_states.deallocate(True)
        return output_tensor

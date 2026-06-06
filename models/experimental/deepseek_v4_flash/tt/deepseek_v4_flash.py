from typing import Any

import ttnn
import torch

from .weight_loader import DeepseekV4WeightLoader


def to_ttnn_device(
    tensor: torch.Tensor,
    device: ttnn.MeshDevice,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, device=device, layout=layout, dtype=ttnn.bfloat16)


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

    def __init__(self, weight: torch.Tensor, device: ttnn.MeshDevice):
        self.weight = ttnn.from_torch(
            weight.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(x, self.weight, compute_kernel_config=_HIFI4)


class DeepSeekV4RMSNorm(DeepSeekV4Module):
    """Weighted RMSNorm over the last dim (matches ``DeepseekV4RMSNorm``)."""

    def __init__(self, weight: torch.Tensor, eps: float, device: ttnn.MeshDevice):
        self.weight = ttnn.from_torch(
            weight.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
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
    def __init__(self, weight_loader: DeepseekV4WeightLoader, device: ttnn.MeshDevice):
        self.weight_loader = weight_loader
        self.device = device
        # ``ttnn.embedding`` expects a row-major weight table.
        self.embedding_weight = to_ttnn_device(
            weight_loader.get_tensor("embed_tokens.weight"), device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

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

    def __init__(self, config: dict, weights_dir: str, device: ttnn.MeshDevice):
        super().__init__()
        self.config = config
        self.weights_dir = weights_dir
        self.device = device
        self.weight_loader = DeepseekV4WeightLoader(weights_dir)
        self.embed_tokens = DeepSeekV4Embedding(self.weight_loader, device)

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

    def __init__(self, config, weights: dict, device, rot, rope_dim: int):
        self.device = device
        self.rope_dim = rope_dim
        self.rot = rot
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.kv_proj = Linear(weights["compressor.kv_proj.weight"], device)
        self.gate_proj = Linear(weights["compressor.gate_proj.weight"], device)
        self.kv_norm = DeepSeekV4RMSNorm(weights["compressor.kv_norm.weight"], self.eps, device)
        # position_bias: [compress_rate, head_dim] -> broadcast over [B, n_win].
        self.position_bias = ttnn.from_torch(
            weights["compressor.position_bias"].reshape(1, 1, self.compress_rate, self.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
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

    def __init__(self, config, weights: dict, device, rot, rope_dim: int):
        self.device = device
        self.rope_dim = rope_dim
        self.rot = rot
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.kv_proj = Linear(weights["compressor.kv_proj.weight"], device)
        self.gate_proj = Linear(weights["compressor.gate_proj.weight"], device)
        self.kv_norm = DeepSeekV4RMSNorm(weights["compressor.kv_norm.weight"], self.eps, device)
        self.position_bias = ttnn.from_torch(
            weights["compressor.position_bias"].reshape(1, 1, self.compress_rate, 2 * self.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
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

    def __init__(self, config, layer_idx: int, weights: dict, device: ttnn.MeshDevice):
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

        self.q_a_proj = Linear(weights["q_a_proj.weight"], device)
        self.q_a_norm = DeepSeekV4RMSNorm(weights["q_a_norm.weight"], self.eps, device)
        self.q_b_proj = Linear(weights["q_b_proj.weight"], device)
        self.kv_proj = Linear(weights["kv_proj.weight"], device)
        self.kv_norm = DeepSeekV4RMSNorm(weights["kv_norm.weight"], self.eps, device)
        self.o_b_proj = Linear(weights["o_b_proj.weight"], device)

        # Grouped output projection (``DeepseekV4GroupedLinear``): block-diagonal
        # over o_groups. Store the per-group weight as [g, in_per_group, out_per_group]
        # so a single batched matmul (batch axis = group) does all groups at once.
        oa = weights["o_a_proj.weight"]  # [g*o_lora_rank, (H*Dh)//g]
        in_per_group = (self.num_heads * self.head_dim) // self.o_groups
        oa = oa.reshape(self.o_groups, self.o_lora_rank, in_per_group).transpose(1, 2).contiguous()
        self.o_a_weight = ttnn.from_torch(oa, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        self.sinks_torch = weights["sinks"].reshape(1, self.num_heads, 1, 1).float()

        self.rot = ttnn.from_torch(
            _interleaved_rotate_matrix(self.rope_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        compressor_cls = _COMPRESSORS.get(self.layer_type)
        self.compressor = (
            compressor_cls(config, weights, device, self.rot, self.rope_dim) if compressor_cls is not None else None
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

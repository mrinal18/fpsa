"""Transformer building blocks shared by the TRM baseline and Implicit TRM.

Faithful ports of the TRM reference layers (Jolicoeur-Martineau, 2025;
github.com/SamsungSAILMontreal/TinyRecursiveModels, MIT license), without the
flash-attn / einops dependencies, plus an optional spectral-norm wrapper used
by the Implicit TRM for contractivity (FPSA paper, Appendix B).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

CosSin = Tuple[torch.Tensor, torch.Tensor]


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Truncated LeCun-style normal init (+-2 std)."""
    if std == 0:
        return tensor.zero_()
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    # Promote half precision to fp32 for the reduction; keep fp64 as fp64
    # (gradient-correctness tests run the model in double precision).
    hidden_states = hidden_states.to(torch.promote_types(input_dtype, torch.float32))
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [bs, seq_len, num_heads, head_dim]; cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    """Linear with fp32 master weights cast to the activation dtype."""

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty(out_features, in_features), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input,
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )


def maybe_spectral_norm(module: nn.Module, enabled: bool) -> nn.Module:
    """Constrain ||W||_2 <= 1 via the modern parametrization API."""
    if enabled:
        from torch.nn.utils.parametrizations import spectral_norm
        return spectral_norm(module, name="weight")
    return module


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty(num_embeddings, embedding_dim), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    """Multi-head attention (SDPA), bidirectional by default, optional RoPE."""

    def __init__(self, hidden_size: int, head_dim: int, num_heads: int,
                 causal: bool = False, spectral_norm: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.output_size = head_dim * num_heads
        self.causal = causal

        self.qkv_proj = maybe_spectral_norm(
            CastedLinear(hidden_size, 3 * self.num_heads * head_dim, bias=False), spectral_norm
        )
        self.o_proj = maybe_spectral_norm(
            CastedLinear(self.output_size, hidden_size, bias=False), spectral_norm
        )

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(B, S, 3 * self.num_heads, self.head_dim)
        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads: 2 * self.num_heads]
        value = qkv[:, :, 2 * self.num_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        query, key, value = (t.transpose(1, 2) for t in (query, key, value))  # B H S D
        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=self.causal)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, self.output_size)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, spectral_norm: bool = False):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = maybe_spectral_norm(
            CastedLinear(hidden_size, inter * 2, bias=False), spectral_norm
        )
        self.down_proj = maybe_spectral_norm(
            CastedLinear(inter, hidden_size, bias=False), spectral_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

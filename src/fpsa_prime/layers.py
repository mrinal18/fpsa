"""One-time encoder/decoder layers and positional utilities."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CosSin = Tuple[torch.Tensor, torch.Tensor]


def inverse_sigmoid(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return math.log(p / (1.0 - p))


def inverse_softplus(x: float) -> float:
    if x <= 0.0:
        raise ValueError("x must be positive")
    return math.log(math.expm1(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.weight._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight.to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float = 2.0, dropout: float = 0.0):
        super().__init__()
        inter = int(expansion * hidden_size * 2.0 / 3.0)
        inter = max(16, ((inter + 15) // 16) * 16)
        self.gate_up = nn.Linear(hidden_size, 2 * inter, bias=False)
        self.down = nn.Linear(inter, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * value))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int, base: float = 10000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_positions).float()
        frequencies = torch.outer(positions, inv_freq)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, seq_len: int) -> CosSin:
        if seq_len > self.cos_cached.shape[0]:
            raise ValueError(
                f"sequence length {seq_len} exceeds RoPE cache {self.cos_cached.shape[0]}"
            )
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin: Optional[CosSin],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to tensors of shape ``(B, H, N, d_h)``."""
    if cos_sin is None:
        return q, k
    cos, sin = cos_sin
    cos = cos.to(device=q.device, dtype=q.dtype).view(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.to(device=q.device, dtype=q.dtype).view(1, 1, sin.shape[0], sin.shape[1])
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

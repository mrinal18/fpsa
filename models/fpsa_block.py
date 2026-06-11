"""FPSA block: the iterated map f(z; x).

Ported from mrinal18/fpsa experiments/sudoku_ablation/fpsa_spatial.py with:
  - 1D RoPE option (toy tasks); 2D RoPE to be added for Sudoku/ARC grids.
  - The map exposed as a pure-ish callable f(z, x, mask) so the solver and
    the implicit backward operate on exactly the same function.
  - NOTE: spectral_norm parametrization is frozen within a solve by the
    solver via torch.nn.utils.parametrize.cached() (power iteration would
    otherwise mutate f between FPI iterations).

Fixed point convention (matches prototype): z* = x + alpha * Update(z*, x)
implemented as damped Picard: z_{k+1} = (1-damping) z_k + damping f_raw(z_k)
where f_raw(z) = x + Update(z, x). Both have the same fixed point.

value_mode:
  fixed     : V = W_V LN(x)                     (original FPSA)
  evolving  : V = W_V LN(z_k)
  blended   : V from (1-beta) LN(x) + beta LN(z_k), beta = per-token gate
  fixed_ffn : fixed V + shared FFN inside the loop (full weight-tied block)
"""
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RoPE1D(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)                       # (N, dh/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    @staticmethod
    def _rot(x, cos, sin):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def forward(self, q, k):  # (B, H, N, dh)
        n = q.shape[-2]
        cos = self.cos[:n].to(q.dtype)
        sin = self.sin[:n].to(q.dtype)
        return self._rot(q, cos, sin), self._rot(k, cos, sin)


class RoPE2D(nn.Module):
    """Axial 2D RoPE (ported from mrinal18/fpsa fpsa_spatial.py): first half of
    head_dim rotates with the ROW index, second half with the COL index, so
    q.k encodes relative (drow, dcol). head_dim % 4 == 0 required."""

    def __init__(self, head_dim: int, grid_hw, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.H, self.W = grid_hw
        half = head_dim // 4
        inv_freq = 1.0 / (base ** (torch.arange(half).float() / half))
        rows = torch.arange(self.H).float()[:, None] * inv_freq   # (H, half)
        cols = torch.arange(self.W).float()[:, None] * inv_freq   # (W, half)
        # per flattened position i: row = i // W, col = i % W
        r_idx = torch.arange(self.H * self.W) // self.W
        c_idx = torch.arange(self.H * self.W) % self.W
        ang = torch.cat([rows[r_idx], cols[c_idx]], dim=-1)        # (N, head_dim//2)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, q, k):  # (B, Hh, N, dh)
        n = q.shape[-2]
        assert n == self.H * self.W, f"expected {self.H*self.W} tokens, got {n}"
        cos = self.cos.to(q.dtype)
        sin = self.sin.to(q.dtype)
        return RoPE1D._rot(q, cos, sin), RoPE1D._rot(k, cos, sin)


class FPSABlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        value_mode: Literal["fixed", "evolving", "blended", "fixed_ffn"] = "fixed_ffn",
        damping: float = 0.5,
        use_spectral_norm: bool = True,
        use_rope: bool = True,
        pos_mode: str = "1d",
        grid_hw=None,
        max_seq_len: int = 2048,
        ffn_mult: float = 2.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.value_mode = value_mode
        self.damping = damping

        sn = (lambda m: nn.utils.parametrizations.spectral_norm(m)) if use_spectral_norm else (lambda m: m)
        self.q_proj = sn(nn.Linear(d_model, d_model, bias=False))
        self.k_proj = sn(nn.Linear(d_model, d_model, bias=False))
        self.o_proj = sn(nn.Linear(d_model, d_model, bias=False))
        self.v_proj = nn.Linear(d_model, d_model, bias=False)   # V not sn'd (prototype)

        self.norm = nn.LayerNorm(d_model)
        self.log_tau = nn.Parameter(torch.zeros(num_heads))      # learned temperature

        self.use_rope = use_rope
        self.pos_mode = pos_mode
        if use_rope:
            if pos_mode == "2d":
                assert grid_hw is not None, "grid_hw required for 2d RoPE"
                self.rope = RoPE2D(self.head_dim, grid_hw)
            else:
                self.rope = RoPE1D(self.head_dim, max_seq_len)

        if value_mode == "blended":
            self.blend_gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        if value_mode == "fixed_ffn":
            h = int(d_model * ffn_mult)
            self.loop_ffn_norm = nn.LayerNorm(d_model)
            self.loop_ffn = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, d_model))

        self.attn_dropout = attn_dropout
        self._attn_drop_mask = None  # variational: one mask per solve

    # ---- plumbing ----
    def _heads(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _unheads(self, x):
        B, H, N, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * dh)

    def precompute(self, x: torch.Tensor):
        """Per-solve precomputation. Returns dict of static tensors."""
        static = {}
        if self.value_mode in ("fixed", "fixed_ffn"):
            static["v"] = self._heads(self.v_proj(self.norm(x)))
        if self.training and self.attn_dropout > 0:
            # variational mask, same across all FPI iterations
            B, N = x.shape[:2]
            keep = 1.0 - self.attn_dropout
            self._attn_drop_mask = (
                torch.rand(B, self.num_heads, N, N, device=x.device) < keep
            ).to(x.dtype) / keep
        else:
            self._attn_drop_mask = None
        return static

    def f(self, z: torch.Tensor, x: torch.Tensor, static: dict,
          attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Damped iteration map. Fixed point: z* = x + AttnUpdate(z*, x) [+FFN]."""
        zn = self.norm(z)
        q = self._heads(self.q_proj(zn))
        k = self._heads(self.k_proj(zn))

        if self.value_mode in ("fixed", "fixed_ffn"):
            v = static["v"]
        elif self.value_mode == "evolving":
            v = self._heads(self.v_proj(self.norm(z)))
        else:  # blended
            beta = self.blend_gate(torch.cat([x, z], dim=-1))
            v = self._heads(self.v_proj((1 - beta) * self.norm(x) + beta * self.norm(z)))

        if self.use_rope:
            q, k = self.rope(q, k)

        tau = self.log_tau.exp().view(1, self.num_heads, 1, 1)
        scores = (q @ k.transpose(-2, -1)) / (math.sqrt(self.head_dim) * tau)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        if self._attn_drop_mask is not None:
            attn = attn * self._attn_drop_mask

        update = self.o_proj(self._unheads(attn @ v))
        z_raw = x + update                                  # in-loop residual to x
        if self.value_mode == "fixed_ffn":
            z_raw = z_raw + self.loop_ffn(self.loop_ffn_norm(z_raw))

        if self.damping < 1.0:
            return (1.0 - self.damping) * z + self.damping * z_raw
        return z_raw

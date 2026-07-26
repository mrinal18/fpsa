"""Shared primitives: RMSNorm, RoPE, SwiGLU, spectral-normalised linears.

Kept deliberately close to the FPRM reference implementation so that FPSA-R and
every baseline in this study share identical sub-modules; the only thing that
varies across architectures is *where the loop sits* and *how the gradient is
computed*.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CosSin = Tuple[torch.Tensor, torch.Tensor]


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return out.to(dtype)


def trunc_normal_(t: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    if std == 0.0:
        return nn.init.zeros_(t)
    return nn.init.trunc_normal_(t, std=std, a=-2 * std, b=2 * std)


class SpectralLinear(nn.Linear):
    """Linear layer whose spectral norm is capped at ``sigma``.

    Uses one power-iteration step per forward (buffered ``u``), in the style of
    Miyato et al., but *rescales only when the estimate exceeds sigma* so the
    layer is free to be a strict contraction. This is the single most important
    stabiliser for the in-layer attention fixed point: removing it collapses the
    loop onto the iteration cap (Appendix H of the FPSA paper).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 sigma: float = 1.0, n_power_iterations: int = 1):
        super().__init__(in_features, out_features, bias=bias)
        self.sigma = sigma
        self.n_power_iterations = n_power_iterations
        self.register_buffer("_u", F.normalize(torch.randn(out_features), dim=0))
        self.register_buffer("_v", F.normalize(torch.randn(in_features), dim=0))

    @torch.no_grad()
    def _power_iterate(self):
        w = self.weight
        u, v = self._u, self._v
        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.mv(w.t(), u), dim=0, eps=1e-12)
            u = F.normalize(torch.mv(w, v), dim=0, eps=1e-12)
        self._u.copy_(u)
        self._v.copy_(v)

    @torch.no_grad()
    def spectral_estimate(self, n_iter: int = 20) -> torch.Tensor:
        """Fresh power iteration, so the estimate is valid in eval mode too
        (the buffered vectors are only refreshed on training forwards)."""
        w = self.weight
        u = self._u.clone()
        for _ in range(n_iter):
            v = F.normalize(torch.mv(w.t(), u), dim=0, eps=1e-12)
            u = F.normalize(torch.mv(w, v), dim=0, eps=1e-12)
        return torch.dot(u, torch.mv(w, v)).abs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._power_iterate()
        # Clone the singular vectors: they are buffers updated in place on every
        # forward, and the autograd graph from an earlier call in the same loop
        # must not see the update.
        u, v = self._u.detach().clone(), self._v.detach().clone()
        s = torch.dot(u, torch.mv(self.weight, v)).abs()
        scale = torch.clamp(self.sigma / (s + 1e-12), max=1.0)
        return F.linear(x, self.weight * scale, self.bias)


def make_linear(in_f: int, out_f: int, bias: bool = True, spectral: bool = False,
                sigma: float = 1.0) -> nn.Linear:
    if spectral:
        return SpectralLinear(in_f, out_f, bias=bias, sigma=sigma)
    return nn.Linear(in_f, out_f, bias=bias)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: Optional[int] = None) -> CosSin:
        if seq_len is None:
            return self.cos_cached, self.sin_cached
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos_sin: Optional[CosSin]):
    """q, k: (B, H, N, dh)."""
    if cos_sin is None:
        return q, k
    cos, sin = cos_sin
    cos = cos.to(q.dtype).view(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.to(q.dtype).view(1, 1, sin.shape[0], sin.shape[1])
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class SwiGLU(nn.Module):
    """SwiGLU MLP, optionally spectrally normalised.

    The gated product is the one genuinely unbounded piece of a transformer
    block: ``silu(Wx) * Vx`` is quadratic in ``x``, so even with unit-norm
    weights its Lipschitz constant grows with the input scale. Since the input
    here is RMS-normalised (fixed norm sqrt(d)), capping the three projections
    at ``sigma`` bounds the block's contribution to the loop's contraction
    factor -- which is what keeps the equilibrium well posed as training
    proceeds.
    """

    def __init__(self, hidden_size: int, expansion: float = 2.0,
                 spectral: bool = False, sigma: float = 1.0):
        super().__init__()
        inter = int(expansion * hidden_size * 2 / 3)
        inter = ((inter + 15) // 16) * 16
        self.gate_up = make_linear(hidden_size, 2 * inter, bias=False,
                                   spectral=spectral, sigma=sigma)
        self.down = make_linear(inter, hidden_size, bias=False,
                                spectral=spectral, sigma=sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class SpatialConv(nn.Module):
    """Depthwise conv over the puzzle grid (FPRM). Identity when disabled."""

    def __init__(self, hidden_size: int, conv_type: str, kernel_size: int, bias: bool):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == "conv1d":
            self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size,
                                  padding=kernel_size - 1, groups=hidden_size, bias=bias)
        elif conv_type == "conv2d":
            if kernel_size % 2 == 0:
                raise ValueError("conv2d needs an odd kernel_size")
            self.conv = nn.Conv2d(hidden_size, hidden_size, kernel_size,
                                  padding=kernel_size // 2, groups=hidden_size, bias=bias)
        elif conv_type != "none":
            raise ValueError(f"unknown conv_type {conv_type!r}")

    def forward(self, x: torch.Tensor, prefix_len: int = 0) -> torch.Tensor:
        if self.conv_type == "none":
            return x
        B, L, D = x.shape
        head, grid = x[:, :prefix_len], x[:, prefix_len:]
        n = grid.shape[1]
        if self.conv_type == "conv1d":
            grid = self.conv(grid.transpose(1, 2)).transpose(1, 2)[:, :n].contiguous()
        else:
            hw = int(math.isqrt(n))
            if hw * hw != n:
                raise ValueError(f"grid length {n} is not a perfect square")
            grid = grid.reshape(B, hw, hw, D).permute(0, 3, 1, 2).contiguous()
            grid = self.conv(grid).permute(0, 2, 3, 1).reshape(B, n, D).contiguous()
        return torch.cat([head, grid], dim=1) if prefix_len else grid

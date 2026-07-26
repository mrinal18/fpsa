"""In-layer Fixed-Point Self-Attention (FPSA) and the single-pass baseline.

FPSA treats the attention sub-computation as a dynamical system in its own
state ``u``:

    V   = x W_V                                   (frozen w.r.t. the inner loop)
    Q_k = u_k W_Q,      K_k = u_k W_K
    A_k = softmax(Q_k K_k^T / (sqrt(d_h) tau_h))
    u_{k+1} = (1 - a) u_k + a * Concat_h(A_k V_h) W_O

Freezing ``V`` on the layer input is what keeps the loop from collapsing every
token onto a single vector; the queries and keys are the only things that
evolve, so the loop refines *token-to-token alignment* rather than re-mixing the
value content over and over.

Two things in this file differ from the reference FPSA implementation and matter
for the reasoning setting:

1. ``step`` is exposed separately from ``solve``. FPSA-R's joint solver takes a
   single inner step per outer step (see ``joint.py``), which has the same
   equilibrium as the nested solve but costs one attention call per outer
   iteration instead of ``inner_max_iter`` of them.
2. ``lipschitz_bound`` returns the analytic per-head contraction estimate
   ``sigma_Q sigma_K sigma_O / (sqrt(d_h) tau)`` used in Appendix B, so training
   runs can *log* the certified contraction factor rather than assume it.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CosSin, SpectralLinear, apply_rotary, make_linear


class FPSAAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, *, damping: float = 0.8,
                 temperature: float = 1.0, spectral_norm: bool = True,
                 causal: bool = False, dropout: float = 0.0,
                 inner_max_iter: int = 6, inner_tol: float = 1e-3):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.inner_max_iter = inner_max_iter
        self.inner_tol = inner_tol
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        self.damping = damping

        self.W_Q = make_linear(hidden_size, hidden_size, bias=False, spectral=spectral_norm)
        self.W_K = make_linear(hidden_size, hidden_size, bias=False, spectral=spectral_norm)
        self.W_O = make_linear(hidden_size, hidden_size, bias=False, spectral=spectral_norm)
        # W_V sits outside the inner loop, so it is deliberately unconstrained.
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)

        self.log_tau = nn.Parameter(torch.full((num_heads,), math.log(temperature)))
        self.dropout = nn.Dropout(dropout)

        # Variational attention dropout mask: sampled once per forward and held
        # fixed across every inner iteration, so f stays deterministic inside the
        # loop and the fixed point is well defined.
        self._attn_drop_mask: Optional[torch.Tensor] = None
        self.attn_dropout_p = dropout

    # -- shape plumbing -----------------------------------------------------
    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, dh = x.shape
        return x.transpose(1, 2).reshape(B, N, H * dh)

    # -- the two halves of the layer ---------------------------------------
    def value_stream(self, x: torch.Tensor) -> torch.Tensor:
        """V = x W_V, split into heads. Constant for the whole inner loop."""
        return self._split(self.W_V(x))

    def step(self, u: torch.Tensor, v: torch.Tensor, x_res: torch.Tensor,
             cos_sin: Optional[CosSin] = None,
             attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One damped FPSA iteration.

        ``u``: (B,N,d) inner state; ``v``: (B,H,N,dh) frozen values;
        ``x_res``: (B,N,d) the layer input, re-injected every iteration.

        The ``x_res +`` term is not optional. Without it the inner map is
        ``u <- W_O A(u) V`` with ``A`` row-stochastic, and repeatedly applying a
        stochastic averaging operator drives every token to the same vector:
        the fixed point is rank-1, and the gradient to ``W_Q``/``W_K`` vanishes
        because the alignment no longer distinguishes tokens. Re-injecting the
        layer input keeps token identity across the loop and makes the fixed
        point ``u* = x + W_O A(u*) V`` -- the form used in the FPSA paper
        (Eq. 5) and in this repository's original ``src/model.py``.
        """
        q = self._split(self.W_Q(u))
        k = self._split(self.W_K(u))
        q, k = apply_rotary(q, k, cos_sin)

        tau = self.log_tau.exp().clamp(min=1e-2).view(1, -1, 1, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.head_dim) * tau)
        if attn_mask is not None:
            scores = scores + attn_mask
        if self.causal:
            n = scores.shape[-1]
            causal = torch.ones(n, n, dtype=torch.bool, device=scores.device).triu(1)
            scores = scores.masked_fill(causal, float("-inf"))

        a = F.softmax(scores, dim=-1)
        if self._attn_drop_mask is not None:
            a = a * self._attn_drop_mask
        out = x_res + self.W_O(self._merge(torch.matmul(a, v)))
        if self.damping < 1.0:
            return (1.0 - self.damping) * u + self.damping * out
        return out

    def solve(self, x: torch.Tensor, cos_sin: Optional[CosSin] = None,
              attn_mask: Optional[torch.Tensor] = None,
              u0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Nested inner solve: iterate ``step`` to tolerance. Used by
        ``solver_mode="nested"`` and by the diagnostics."""
        v = self.value_stream(x)
        u = x if u0 is None else u0
        info = {"iters": 0, "residual": float("inf")}
        for i in range(self.inner_max_iter):
            u_next = self.step(u, v, x, cos_sin, attn_mask)
            with torch.no_grad():
                r = ((u_next - u).norm(dim=-1) / u.norm(dim=-1).clamp_min(1e-8)).max()
            u = u_next
            info["iters"] = i + 1
            info["residual"] = float(r)
            if r < self.inner_tol:
                break
        return u, info

    # -- diagnostics --------------------------------------------------------
    def sample_dropout_mask(self, B: int, N: int, device, dtype):
        if self.training and self.attn_dropout_p > 0:
            keep = 1.0 - self.attn_dropout_p
            self._attn_drop_mask = (
                torch.empty(B, self.num_heads, N, N, device=device, dtype=dtype)
                .bernoulli_(keep) / keep
            )
        else:
            self._attn_drop_mask = None

    def clear_dropout_mask(self):
        self._attn_drop_mask = None

    @torch.no_grad()
    def lipschitz_bound(self) -> float:
        """Analytic contraction estimate L <= a * sQ*sK*sO / (sqrt(dh)*tau) + (1-a).

        Matches Eq. (13) of the FPSA paper, with the damping correction
        ``gamma_a = a L_f + (1 - a)`` from Proposition B.1.
        """
        def snorm(layer):
            if isinstance(layer, SpectralLinear):
                return float(torch.clamp(layer.spectral_estimate(), max=layer.sigma))
            return float(torch.linalg.matrix_norm(layer.weight, ord=2))

        tau = float(self.log_tau.exp().clamp(min=1e-2).min())
        lf = snorm(self.W_Q) * snorm(self.W_K) * snorm(self.W_O) / (math.sqrt(self.head_dim) * tau)
        return self.damping * lf + (1.0 - self.damping)


class VanillaAttention(nn.Module):
    """Single-pass MHA with the same parameter budget, for the baselines."""

    def __init__(self, hidden_size: int, num_heads: int, *, causal: bool = False,
                 dropout: float = 0.0, spectral_norm: bool = False, **_unused):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.W_Q = make_linear(hidden_size, hidden_size, bias=False, spectral=spectral_norm)
        self.W_K = make_linear(hidden_size, hidden_size, bias=False, spectral=spectral_norm)
        self.W_O = make_linear(hidden_size, hidden_size, bias=False, spectral=spectral_norm)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(num_heads))
        self.attn_dropout_p = dropout
        self._attn_drop_mask = None

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        B, H, N, dh = x.shape
        return x.transpose(1, 2).reshape(B, N, H * dh)

    def forward(self, x, cos_sin=None, attn_mask=None):
        q, k, v = self._split(self.W_Q(x)), self._split(self.W_K(x)), self._split(self.W_V(x))
        q, k = apply_rotary(q, k, cos_sin)
        tau = self.log_tau.exp().clamp(min=1e-2).view(1, -1, 1, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.head_dim) * tau)
        if attn_mask is not None:
            scores = scores + attn_mask
        if self.causal:
            n = scores.shape[-1]
            scores = scores.masked_fill(
                torch.ones(n, n, dtype=torch.bool, device=scores.device).triu(1), float("-inf"))
        a = F.softmax(scores, dim=-1)
        if self._attn_drop_mask is not None:
            a = a * self._attn_drop_mask
        return self.W_O(self._merge(torch.matmul(a, v)))

    def sample_dropout_mask(self, B, N, device, dtype):
        if self.training and self.attn_dropout_p > 0:
            keep = 1.0 - self.attn_dropout_p
            self._attn_drop_mask = (
                torch.empty(B, self.num_heads, N, N, device=device, dtype=dtype)
                .bernoulli_(keep) / keep)
        else:
            self._attn_drop_mask = None

    def clear_dropout_mask(self):
        self._attn_drop_mask = None

    @torch.no_grad()
    def lipschitz_bound(self) -> float:
        def snorm(layer):
            if isinstance(layer, SpectralLinear):
                return float(torch.clamp(layer.spectral_estimate(), max=layer.sigma))
            return float(torch.linalg.matrix_norm(layer.weight, ord=2))
        tau = float(self.log_tau.exp().clamp(min=1e-2).min())
        return snorm(self.W_Q) * snorm(self.W_K) * snorm(self.W_O) / (math.sqrt(self.head_dim) * tau)

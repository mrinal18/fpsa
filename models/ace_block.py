"""ACE: Anchored Contractive Equilibrium block (docs/ACE_PROPOSAL.md).

Iteration:  z_{k+1} = B(x) x + (I - B(x)) T(z_k),  beta_i(x) in [b_min, b_max].
T is nonexpansive BY CONSTRUCTION in the token-max norm ||z||_{inf,2} :=
max_i ||z_i||_2 (Banach holds in any complete norm):

  1. softnorm(u) = u / sqrt(||u||^2 + 1): 1-Lipschitz, output norm < 1.
     (Jacobian = (I - u u^T/(||u||^2+1)) / sqrt(||u||^2+1), norm <= 1.)
  2. Attention: q^,k^,v^ = softnorm(W_q z), softnorm(W_k z), softnorm(W_v z)
     per head, RoPE on q^,k^ (orthogonal, free), scores tau * q^.k^,
     out = W_o concat_h(A_h v^_h) / L_attn,  L_attn = 1 + 4*tau*sqrt(H).
     Derivation (token-max norm): value path with row-stochastic A is a
     per-token convex combination of v^_j -> <= max_j ||dv^_j|| (1-Lip);
     score path: |ds_ij| <= 2 tau d, softmax l_inf->l1 constant <= 2,
     ||v^|| <= 1  ->  <= 4 tau d per head, sqrt(H) across concat.
     All spectral norms ||W|| <= 1 by EXACT normalization (see SpectralCap).
     The constant is conservative; V2 audits the true Jacobian numerically.
  3. Convex residuals: y = (1-g) z + g h(z), g = sigmoid(param) in (0,1):
     convex combos of nonexpansive maps are nonexpansive (vs z + h(z),
     which is (1+L)-Lipschitz — the FPSA gain leak).
  4. FFN: W2 GroupSort(W1 .) per token, exact-SN <= 1, GroupSort is a
     norm-preserving rearrangement (1-Lipschitz; universality: Anil 2019).
  5. NO LayerNorm, NO ball projection inside the loop (bounds above are
     scale-free thanks to softnorm).

Contraction: ||F(z)-F(z')||_{inf,2} <= (1-b_min) ||z-z'||_{inf,2}.
beta depends on x ONLY (never z) — load-bearing for the proof.
Encoder (before loop) and head (after loop) are unconstrained.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fpsa_block import RoPE1D, RoPE2D


class SpectralCap(nn.Module):
    """Exact parametrization W -> W / max(1, sigma_max(W)).
    Guarantees ||W||_2 <= 1 (up to float error); differentiable."""

    def forward(self, W):
        s = torch.linalg.matrix_norm(W, ord=2)
        return W / torch.clamp(s, min=1.0)


def sn_exact(linear: nn.Linear) -> nn.Linear:
    nn.utils.parametrize.register_parametrization(linear, "weight", SpectralCap())
    return linear


def softnorm(u: torch.Tensor) -> torch.Tensor:
    return u / torch.sqrt(u.pow(2).sum(-1, keepdim=True) + 1.0)


def groupsort(u: torch.Tensor) -> torch.Tensor:
    a, b = u[..., 0::2], u[..., 1::2]
    out = torch.empty_like(u)
    out[..., 0::2] = torch.maximum(a, b)
    out[..., 1::2] = torch.minimum(a, b)
    return out


class ACEBlock(nn.Module):
    """Exposes precompute(x) + f(z, x, static, mask): drop-in for
    fixed_point_solve. Use damping=1.0 (anchor replaces damping)."""

    def __init__(self, d_model: int, num_heads: int, tau: float = 0.5,
                 beta_min: float = 0.2, beta_max: float = 0.9,
                 pos_mode: str = "1d", grid_hw=None, max_seq_len: int = 256,
                 ffn_mult: float = 2.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.tau = tau
        self.beta_min, self.beta_max = beta_min, beta_max
        self.L_attn = 1.0 + 4.0 * tau * math.sqrt(num_heads)

        self.q_proj = sn_exact(nn.Linear(d_model, d_model, bias=False))
        self.k_proj = sn_exact(nn.Linear(d_model, d_model, bias=False))
        self.v_proj = sn_exact(nn.Linear(d_model, d_model, bias=False))
        self.o_proj = sn_exact(nn.Linear(d_model, d_model, bias=False))
        h = int(d_model * ffn_mult)
        assert h % 2 == 0
        self.ffn1 = sn_exact(nn.Linear(d_model, h, bias=False))
        self.ffn2 = sn_exact(nn.Linear(h, d_model, bias=False))

        # convex-residual gates (scalars, learned, in (0,1))
        self.gate_attn = nn.Parameter(torch.tensor(0.0))   # sigmoid -> 0.5
        self.gate_ffn = nn.Parameter(torch.tensor(0.0))
        # per-token anchor head (x-only)
        self.beta_head = nn.Linear(d_model, 1)

        if pos_mode == "2d":
            assert grid_hw is not None
            self.rope = RoPE2D(self.head_dim, grid_hw)
        else:
            self.rope = RoPE1D(self.head_dim, max_seq_len)

    def _heads(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _unheads(self, x):
        B, H, N, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * dh)

    def rho(self) -> float:
        """Certified global contraction factor."""
        return 1.0 - self.beta_min

    def precompute(self, x: torch.Tensor):
        beta = self.beta_min + (self.beta_max - self.beta_min) * \
            torch.sigmoid(self.beta_head(x))            # (B, N, 1), x-only
        return {"beta": beta}

    def T(self, z: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        qh = softnorm(self._heads(self.q_proj(z)))
        kh = softnorm(self._heads(self.k_proj(z)))
        vh = softnorm(self._heads(self.v_proj(z)))
        qh, kh = self.rope(qh, kh)
        scores = self.tau * (qh @ kh.transpose(-2, -1))
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        a_out = self.o_proj(self._unheads(attn @ vh)) / self.L_attn
        g_a = torch.sigmoid(self.gate_attn)
        u = (1.0 - g_a) * z + g_a * a_out
        f_out = self.ffn2(groupsort(self.ffn1(u)))
        g_f = torch.sigmoid(self.gate_ffn)
        return (1.0 - g_f) * u + g_f * f_out

    def f(self, z: torch.Tensor, x: torch.Tensor, static: dict,
          attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        beta = static["beta"]
        return beta * x + (1.0 - beta) * self.T(z, attn_mask)


class ACESeqModel(nn.Module):
    """Unconstrained encoder -> ACE fixed-point loop -> unconstrained head.
    Mirrors FPSASeqModel's interface (drop-in for train/eval harnesses)."""

    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 64,
                 num_heads: int = 4, tau: float = 0.5, beta_min: float = 0.2,
                 beta_max: float = 0.9, max_iter: int = 32, tol: float = 1e-3,
                 backward: str = "neumann", neumann_steps: int = 12,
                 pos_mode: str = "1d", grid_hw=None, max_seq_len: int = 256,
                 ffn_mult: float = 2.0, **_ignored):
        super().__init__()
        from models.solver import fixed_point_solve
        self._solve = fixed_point_solve
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                     nn.Linear(2 * d_model, d_model))
        self.block = ACEBlock(d_model, num_heads, tau=tau, beta_min=beta_min,
                              beta_max=beta_max, pos_mode=pos_mode,
                              grid_hw=grid_hw, max_seq_len=max_seq_len,
                              ffn_mult=ffn_mult)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.max_iter, self.tol = max_iter, tol
        self.backward_mode, self.neumann_steps = backward, neumann_steps
        self.jac_reg = False   # ACE needs no stability machinery (the point)

    def forward(self, tokens, attn_mask=None, max_iter=None):
        x = self.encoder(self.embed(tokens))
        z_star, stats = self._solve(
            self.block, x, attn_mask=attn_mask,
            max_iter=max_iter or self.max_iter, tol=self.tol,
            backward=self.backward_mode, neumann_steps=self.neumann_steps,
            neumann_tol=getattr(self, 'neumann_tol', 1e-4), jac_reg=False)
        logits = self.head(self.out_norm(z_star))
        return logits, stats

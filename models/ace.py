"""ACE: Anchored Contractive Equilibrium block (v0, certified constants).

Iteration:  z+ = beta(x) * x_anchor + (1 - beta(x)) * T(z)
with beta_i(x) in [beta_min, beta_max] subset (0,1), depending on x ONLY,
and T nonexpansive BY CONSTRUCTION. Then F is a (1-beta_min)-contraction
(two-line proof in docs/ACE_PROPOSAL.md), so existence/uniqueness, the
residual certificate, gradient validity and anytime monotonicity are
theorems, not regularizers. NO Jacobian regularization, NO divergence
guard, NO extra damping.

T = Pi_R o FFNres o ATTNres, each factor 1-Lipschitz:

ATTN (fixed V, certified): per head h,
  q = Wq z, k = Wk z (exact spectral normalization: W <- W / max(1, s(W))),
  scores s_ij = (RoPE q_i) . (RoPE k_j) / c_h,  a = softmax,  o_i = sum_j a_ij v_j
  with v = Wv x_anchor FIXED during the solve.
DERIVED BOUND (vs z, stacked l2, on the ball ||z_l|| <= R):
  |ds_ij| <= sQ sK R (||d_i|| + ||d_j||) / c_h              (RoPE orthogonal)
  ||da_i||_2 <= (1/2) ||ds_i.||_2                            (softmax Jacobian <= 1/2)
  ||ds_i.||_2^2 <= (sQ sK R/c)^2 * 2 (N ||d_i||^2 + ||d||^2)
  ||do_i|| <= ||V||_2 ||da_i||_2          (V = matrix of v_j rows, EXACT norm)
  => sum_i ||do_i||^2 <= N (R ||V||_2 sQ sK / c)^2 ||d||^2
  => L_attn_head <= sqrt(N) R ||V||_2 sQ sK / c_h.
Setting c_h = sqrt(N) R ||V||_2 sQ sK / g_h with learnable g_h in (0, g_max),
g_max < 1, certifies L_attn_head <= g_h. Heads concatenated and scaled by
1/sqrt(H), then exact-normalized W_O: L_attn <= max_h g_h < 1.
ATTNres: u = (1-ga) z + ga Attn(z), ga in (0,1)  => 1-Lipschitz.

FFN: exact-normalized linears + GroupSort (pairwise (max,min): a permutation
of each pair => exactly norm-preserving, 1-Lipschitz; universal for
Lipschitz functions per Anil et al. 2019).
FFNres: w = (1-gf) u + gf FFN(u)  => 1-Lipschitz.

Pi_R: per-token projection z_i * min(1, R/||z_i||): firmly nonexpansive.
Domain invariance: x_anchor projected, T ends in Pi_R, convex anchor
=> ||z_i|| <= R for all iterates, so the attention bound applies globally.

Unconstrained encoder (before loop) and head (after loop) carry the
representational burden; only the iterated map is constrained.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fpsa_block import RoPE1D, RoPE2D


def exact_sn(W: torch.Tensor) -> torch.Tensor:
    """W / max(1, sigma_max(W)) with EXACT sigma (differentiable SVD norm)."""
    s = torch.linalg.matrix_norm(W, ord=2)
    return W / torch.clamp(s, min=1.0)


def group_sort(x: torch.Tensor) -> torch.Tensor:
    """Pairwise (max, min) on adjacent channels; exactly 1-Lipschitz."""
    a, b = x[..., 0::2], x[..., 1::2]
    hi, lo = torch.maximum(a, b), torch.minimum(a, b)
    out = torch.empty_like(x)
    out[..., 0::2] = hi
    out[..., 1::2] = lo
    return out


def ball_project(z: torch.Tensor, R: float) -> torch.Tensor:
    n = z.norm(dim=-1, keepdim=True)
    return z * torch.clamp(R / n.clamp(min=1e-12), max=1.0)


class ACEBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, R: float = 3.0,
                 beta_min: float = 0.2, beta_max: float = 0.8,
                 g_max: float = 0.97, ffn_mult: float = 2.0,
                 certified: bool = True,
                 pos_mode: str = "1d", grid_hw=None, max_seq_len: int = 256):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.R, self.beta_min, self.beta_max, self.g_max = R, beta_min, beta_max, g_max
        self.certified = certified
        if not certified:  # ABLATION: learnable temperature, bound NOT enforced
            self.log_tau = nn.Parameter(torch.zeros(num_heads))
        # damping handled by the anchor; expose 1.0 for solver compatibility
        self.damping = 1.0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_logit = nn.Parameter(torch.zeros(num_heads))      # attention gain
        self.ga_logit = nn.Parameter(torch.tensor(0.0))          # attn residual weight
        self.gf_logit = nn.Parameter(torch.tensor(0.0))          # ffn residual weight

        h = int(d_model * ffn_mult)
        assert h % 2 == 0 and d_model % 2 == 0
        self.ffn1 = nn.Linear(d_model, h, bias=True)
        self.ffn2 = nn.Linear(h, d_model, bias=True)

        self.beta_head = nn.Linear(2 * d_model, 1)

        if pos_mode == "2d":
            self.rope = RoPE2D(self.head_dim, grid_hw)
        else:
            self.rope = RoPE1D(self.head_dim, max_seq_len)

    def _heads(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _unheads(self, x):
        B, H, N, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * dh)

    def precompute(self, x: torch.Tensor):
        """x is the ENCODED input (any scale); everything z-independent is
        computed here once per solve, with graph, and certified."""
        B, N, _ = x.shape
        x_anchor = ball_project(x, self.R)

        Wq, Wk, Wo = exact_sn(self.q_proj.weight), exact_sn(self.k_proj.weight), \
            exact_sn(self.o_proj.weight)
        v = self._heads(F.linear(x_anchor, self.v_proj.weight))   # V unconstrained
        # exact ||V||_2 per (batch, head): spectral norm of the N x dh matrix
        v_norm = torch.linalg.matrix_norm(v, ord=2)               # (B, H)

        if self.certified:
            g = self.g_max * torch.sigmoid(self.g_logit)          # (H,)
            # certified: c = sqrt(N) R ||V||2 sQ sK / g ; sQ=sK=1 exact
            c = (math.sqrt(N) * self.R * v_norm) / g.view(1, -1)  # (B, H)
        else:
            # ABLATION (a-posteriori monitored, not a-priori certified)
            c = (self.log_tau.exp() * math.sqrt(self.head_dim)).view(1, -1).expand(B, -1)

        bmean = x_anchor.mean(dim=1, keepdim=True).expand_as(x_anchor)
        blogit = self.beta_head(torch.cat([x_anchor, bmean], dim=-1))
        beta = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(blogit)

        return {"x_anchor": x_anchor, "Wq": Wq, "Wk": Wk, "Wo": Wo,
                "W1": exact_sn(self.ffn1.weight), "W2": exact_sn(self.ffn2.weight),
                "v": v, "c": c, "beta": beta,
                "ga": torch.sigmoid(self.ga_logit),
                "gf": torch.sigmoid(self.gf_logit)}

    def f(self, z: torch.Tensor, x: torch.Tensor, static: dict,
          attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = static
        q = self._heads(F.linear(z, s["Wq"]))
        k = self._heads(F.linear(z, s["Wk"]))
        q, k = self.rope(q, k)
        scores = (q @ k.transpose(-2, -1)) / s["c"][..., None, None]
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        o = self._unheads(attn @ s["v"]) / math.sqrt(self.num_heads)
        o = F.linear(o, s["Wo"])
        u = (1 - s["ga"]) * z + s["ga"] * o                      # attn convex residual

        w = F.linear(group_sort(F.linear(u, s["W1"], self.ffn1.bias)),
                     s["W2"], self.ffn2.bias)
        u = (1 - s["gf"]) * u + s["gf"] * w                      # ffn convex residual

        t = ball_project(u, self.R)                              # T(z), 1-Lipschitz
        return s["beta"] * s["x_anchor"] + (1 - s["beta"]) * t   # anchored update


class ACESeqModel(nn.Module):
    """Unconstrained encoder -> ACE fixed-point solve -> unconstrained head."""

    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 64,
                 num_heads: int = 4, beta_min: float = 0.2, beta_max: float = 0.8,
                 R: float = 3.0, g_max: float = 0.97, ffn_mult: float = 2.0,
                 certified: bool = True,
                 max_iter: int = 40, tol: float = 1e-3, backward: str = "neumann",
                 neumann_steps: int = 10, pos_mode: str = "1d", grid_hw=None,
                 max_seq_len: int = 256):
        super().__init__()
        from models.solver import fixed_point_solve
        self._solve = fixed_point_solve
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model))
        self.block = ACEBlock(d_model, num_heads, R=R, beta_min=beta_min,
                              beta_max=beta_max, g_max=g_max, ffn_mult=ffn_mult,
                              certified=certified,
                              pos_mode=pos_mode, grid_hw=grid_hw,
                              max_seq_len=max_seq_len)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.max_iter, self.tol = max_iter, tol
        self.backward_mode, self.neumann_steps = backward, neumann_steps
        self.rho = 1.0 - beta_min  # certified contraction factor

    def forward(self, tokens: torch.Tensor, max_iter: Optional[int] = None):
        x = self.encoder(self.embed(tokens))
        # init INSIDE the ball: the certified attention bound assumes
        # ||z_l|| <= R; F maps the ball to itself, so all iterates stay in.
        z0 = ball_project(x, self.block.R)
        z_star, stats = self._solve(
            self.block, x, z0=z0, max_iter=max_iter or self.max_iter, tol=self.tol,
            backward=self.backward_mode, neumann_steps=self.neumann_steps)
        return self.head(self.out_norm(z_star)), stats

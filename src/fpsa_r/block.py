"""The recurrent reasoning block and the *joint* update map G.

FPSA-R has two nested equilibria: an outer one over the block state ``z`` (the
looped-transformer recursion) and an inner one over the attention state ``u``
(FPSA). Solving them nested costs ``inner_iters x outer_iters`` attention calls
and, worse, makes the backward pass a nested linear solve.

Instead we lift both into a *single* joint state ``s = (z, u)`` with the map

    u_{t+1} = a(u_t ; z_t, x)                (one damped FPSA step)
    z_{t+1} = B(z_t, u_{t+1}, x)             (rest of the block)

whose fixed points are exactly the pairs where ``u* = a(u*; z*, x)`` and
``z* = B(z*, u*, x)`` -- i.e. the same solutions as the nested formulation, since
the joint map is block-triangular in the two states. One step of G costs one
attention call, and its transposed Jacobian is a single VJP, so the backward
solve is *single-level* even though the model is two-level. That is the whole
trick: two-level expressivity at one-level cost, in both directions.

``s`` is carried as one stacked tensor of shape ``(1 + n_layers, B, N, D)``
(row 0 is ``z``, rows 1.. are the per-sub-block attention states) so the
implicit-differentiation machinery in ``implicit.py`` never has to know how many
pieces the state has.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import FPSAAttention, VanillaAttention
from .config import FPSARConfig
from .layers import SpatialConv, SwiGLU, rms_norm


class _SubBlock(nn.Module):
    """attention -> MLP, both with pre-norm and FPRM residual scaling."""

    def __init__(self, cfg: FPSARConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.fpsa:
            self.attn = FPSAAttention(
                cfg.hidden_size, cfg.num_heads, damping=cfg.inner_damping,
                temperature=cfg.attn_temperature, spectral_norm=cfg.spectral_norm,
                causal=cfg.causal, dropout=cfg.dropout,
                inner_max_iter=cfg.inner_max_iter, inner_tol=cfg.inner_tol)
        else:
            self.attn = VanillaAttention(
                cfg.hidden_size, cfg.num_heads, causal=cfg.causal,
                dropout=cfg.dropout, spectral_norm=cfg.spectral_norm)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.expansion,
                          spectral=cfg.spectral_norm, sigma=cfg.mlp_sigma)
        self.eps = cfg.rms_norm_eps

    def attn_and_mlp(self, h: torch.Tensor, attn_out: torch.Tensor,
                     alpha_1, beta_1) -> torch.Tensor:
        h = alpha_1 * h + beta_1 * attn_out
        h = alpha_1 * h + beta_1 * self.mlp(rms_norm(h, self.eps))
        return h


class ReasoningBlock(nn.Module):
    """One recurrent block: input injection -> conv -> [attn -> MLP] x n_layers."""

    def __init__(self, cfg: FPSARConfig):
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.n_block_layers
        self.layers = nn.ModuleList([_SubBlock(cfg) for _ in range(self.n_layers)])
        self.conv = SpatialConv(cfg.hidden_size, cfg.conv_type,
                                cfg.conv_kernel_size, cfg.conv_bias)
        self.eps = cfg.rms_norm_eps

        d = cfg.hidden_size
        if cfg.residual_scale in ("none", None):
            self.residual_scale = None
        else:
            self.residual_scale = cfg.residual_scale
            import math
            l1 = math.log(cfg.alpha_1_init / (1 - cfg.alpha_1_init))
            l2 = math.log(cfg.alpha_2_init / (1 - cfg.alpha_2_init))
            requires_grad = cfg.residual_scale == "input-independent"
            self.alpha_1_param = nn.Parameter(l1 * torch.ones(d), requires_grad=requires_grad)
            self.alpha_2_param = nn.Parameter(l2 * torch.ones(d), requires_grad=requires_grad)
            self.alpha_1_param._no_weight_decay = True
            self.alpha_2_param._no_weight_decay = True

    # -- state bookkeeping --------------------------------------------------
    @property
    def n_state_rows(self) -> int:
        return 1 + (self.n_layers if self.cfg.fpsa else 0)

    def init_state(self, B: int, N: int, device, dtype) -> torch.Tensor:
        rows = self.n_state_rows
        if self.cfg.init_std > 0:
            s = torch.randn(rows, B, N, self.cfg.hidden_size, device=device, dtype=dtype)
            s = s * self.cfg.init_std
        else:
            s = torch.zeros(rows, B, N, self.cfg.hidden_size, device=device, dtype=dtype)
        return s

    def _scales(self, h: torch.Tensor):
        if self.residual_scale is None:
            one = h.new_ones(1, 1, 1)
            return one, one, one, one
        a1 = torch.sigmoid(self.alpha_1_param).to(h.dtype).view(1, 1, -1)
        a2 = torch.sigmoid(self.alpha_2_param).to(h.dtype).view(1, 1, -1)
        b2 = 1 - a2 * a1.pow(2 * self.n_layers)
        b1 = b2 * (1 - a1) / (1 - a1.pow(2 * self.n_layers) + 1e-5)
        return a1, a2, b1, b2

    # -- the joint map G ----------------------------------------------------
    def joint_step(self, s: torch.Tensor, x_inj: torch.Tensor, seq_info: dict) -> torch.Tensor:
        """s -> G(s, x). ``s`` is (rows, B, N, D)."""
        cos_sin = seq_info.get("cos_sin")
        attn_mask = seq_info.get("attn_mask")
        prefix = seq_info.get("prefix_len", 0)

        z = s[0]
        a1, a2, b1, b2 = self._scales(z)

        h = a2 * z + b2 * x_inj
        h = self.conv(h, prefix)

        new_rows = []
        for i, layer in enumerate(self.layers):
            xa = rms_norm(h, self.eps)
            if self.cfg.fpsa:
                u = s[1 + i]
                v = layer.attn.value_stream(xa)
                # A zero-initialised carry means "no inner state yet": start the
                # inner loop from the layer input, as FPSA does.
                u_in = torch.where(u.abs().sum(dim=-1, keepdim=True) > 0, u, xa)
                u_next = layer.attn.step(u_in, v, cos_sin, attn_mask)
                new_rows.append(u_next)
                attn_out = u_next
            else:
                attn_out = layer.attn(xa, cos_sin, attn_mask)
            h = layer.attn_and_mlp(h, attn_out, a1, b1)

        return torch.stack([h] + new_rows, dim=0)

    def nested_step(self, s: torch.Tensor, x_inj: torch.Tensor, seq_info: dict) -> torch.Tensor:
        """Naive composition: run each inner FPSA loop to tolerance inside one
        outer step. Same equilibrium as ``joint_step``, ``inner_max_iter`` times
        the cost. Kept for the solver ablation."""
        cos_sin = seq_info.get("cos_sin")
        attn_mask = seq_info.get("attn_mask")
        prefix = seq_info.get("prefix_len", 0)

        z = s[0]
        a1, a2, b1, b2 = self._scales(z)
        h = a2 * z + b2 * x_inj
        h = self.conv(h, prefix)

        new_rows = []
        for i, layer in enumerate(self.layers):
            xa = rms_norm(h, self.eps)
            if self.cfg.fpsa:
                u = s[1 + i]
                u0 = torch.where(u.abs().sum(dim=-1, keepdim=True) > 0, u, xa)
                u_star, _ = layer.attn.solve(xa, cos_sin, attn_mask, u0=u0)
                new_rows.append(u_star)
                attn_out = u_star
            else:
                attn_out = layer.attn(xa, cos_sin, attn_mask)
            h = layer.attn_and_mlp(h, attn_out, a1, b1)
        return torch.stack([h] + new_rows, dim=0)

    # -- dropout plumbing ---------------------------------------------------
    def sample_dropout_masks(self, B: int, N: int, device, dtype):
        for layer in self.layers:
            layer.attn.sample_dropout_mask(B, N, device, dtype)

    def clear_dropout_masks(self):
        for layer in self.layers:
            layer.attn.clear_dropout_mask()

    @torch.no_grad()
    def lipschitz_bounds(self):
        return [layer.attn.lipschitz_bound() for layer in self.layers]

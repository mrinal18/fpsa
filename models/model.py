"""Embed -> FPSA fixed-point solve -> per-token classification head."""
from typing import Optional

import torch
import torch.nn as nn

from models.fpsa_block import FPSABlock
from models.solver import fixed_point_solve, SolveStats


class FPSASeqModel(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 64,
                 num_heads: int = 4, value_mode: str = "fixed_ffn",
                 damping: float = 0.5, max_iter: int = 16, tol: float = 1e-3,
                 backward: str = "neumann", neumann_steps: int = 5,
                 use_spectral_norm: bool = True, use_rope: bool = True,
                 pos_mode: str = '1d', grid_hw=None, max_seq_len: int = 256, ffn_mult: float = 2.0,
                 attn_dropout: float = 0.0, jac_reg: bool = False,
                 freeze_sn: bool = True, jac_power_iters: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.block = FPSABlock(d_model, num_heads, value_mode=value_mode,
                               damping=damping, use_spectral_norm=use_spectral_norm,
                               use_rope=use_rope, pos_mode=pos_mode, grid_hw=grid_hw,
                               max_seq_len=max_seq_len,
                               ffn_mult=ffn_mult, attn_dropout=attn_dropout)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.max_iter, self.tol = max_iter, tol
        self.backward_mode, self.neumann_steps = backward, neumann_steps
        self.jac_reg = jac_reg
        self.freeze_sn = freeze_sn
        self.jac_power_iters = jac_power_iters

    def forward(self, tokens: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                max_iter: Optional[int] = None):
        x = self.embed(tokens)
        z_star, stats = fixed_point_solve(
            self.block, x, attn_mask=attn_mask,
            max_iter=max_iter or self.max_iter, tol=self.tol,
            backward=self.backward_mode, neumann_steps=self.neumann_steps,
            jac_reg=self.jac_reg and self.training, freeze_sn=self.freeze_sn,
            jac_power_iters=self.jac_power_iters)
        logits = self.head(self.out_norm(z_star))
        return logits, stats

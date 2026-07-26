"""FPSA-R: the full reasoner, plus every baseline in the study.

All architectures are the *same* module with different switches, so a comparison
is width-, depth- and parameter-matched by construction:

    fpsa_r       in-layer FPSA + joint equilibrium + implicit gradient   (ours)
    deq_block    block equilibrium + implicit gradient, no in-layer FPSA
    fprm         block fixed point + truncated BPTT                      (FPRM)
    looped_bptt  weight-tied looped transformer, full BPTT
    ut_act       Universal-Transformer-style ACT halting, full BPTT
    transformer  non-recursive depth-matched stack
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .block import ReasoningBlock
from .config import FPSARConfig
from .implicit import solve_equilibrium
from .layers import RotaryEmbedding, rms_norm


class FPSAReasoner(nn.Module):
    def __init__(self, cfg: FPSARConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size

        self.embed_scale = math.sqrt(d)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, d)
        nn.init.trunc_normal_(self.embed_tokens.weight, std=1.0 / self.embed_scale)

        self.puzzle_emb_len = cfg.puzzle_emb_len
        if self.puzzle_emb_len > 0:
            self.puzzle_emb = nn.Embedding(cfg.num_puzzle_identifiers,
                                           self.puzzle_emb_len * d)
            nn.init.zeros_(self.puzzle_emb.weight)

        total_len = cfg.seq_len + self.puzzle_emb_len
        if cfg.pos_encodings == "rope":
            self.rotary = RotaryEmbedding(d // cfg.num_heads, total_len, cfg.rope_theta)
        elif cfg.pos_encodings == "learned":
            self.embed_pos = nn.Embedding(total_len, d)
            nn.init.trunc_normal_(self.embed_pos.weight, std=1.0 / self.embed_scale)

        self.block = ReasoningBlock(cfg)
        self.lm_head = nn.Linear(d, cfg.out_vocab_size or cfg.vocab_size, bias=False)
        self.q_head = nn.Linear(d, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

        self.last_info = None
        self.last_rho = None
        self._power_vec = None

    # ------------------------------------------------------------------
    def _inputs(self, tokens: torch.Tensor, puzzle_ids: Optional[torch.Tensor]):
        e = self.embed_tokens(tokens.to(torch.long))
        if self.puzzle_emb_len > 0:
            if puzzle_ids is None:
                puzzle_ids = torch.zeros(tokens.shape[0], dtype=torch.long,
                                         device=tokens.device)
            p = self.puzzle_emb(puzzle_ids).view(-1, self.puzzle_emb_len,
                                                 self.cfg.hidden_size)
            e = torch.cat([p, e], dim=1)
        if self.cfg.pos_encodings == "learned":
            pos = torch.arange(e.shape[1], device=e.device)
            e = 0.7071067811865476 * (e + self.embed_pos(pos).unsqueeze(0))
        return self.embed_scale * e

    def _seq_info(self, n: int, attn_mask=None):
        cos_sin = None
        if self.cfg.pos_encodings == "rope":
            cos_sin = self.rotary(n)
        return {"cos_sin": cos_sin, "attn_mask": attn_mask,
                "prefix_len": self.puzzle_emb_len}

    def _readout(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(rms_norm(z, self.cfg.rms_norm_eps))
        return logits[:, self.puzzle_emb_len:]

    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor, puzzle_ids: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                max_iter: Optional[int] = None,
                record_trace: bool = False) -> Dict[str, torch.Tensor]:
        x_inj = self._inputs(tokens, puzzle_ids)
        B, N, _ = x_inj.shape
        seq_info = self._seq_info(N, attn_mask)

        self.block.sample_dropout_masks(B, N, x_inj.device, x_inj.dtype)
        try:
            if self.cfg.halting == "act":
                return self._forward_act(x_inj, seq_info, max_iter)

            step = (self.block.nested_step if self.cfg.solver_mode == "nested"
                    else self.block.joint_step)
            step_fn = lambda s: step(s, x_inj, seq_info)

            s0 = self.block.init_state(B, N, x_inj.device, x_inj.dtype)
            budget = max_iter if max_iter is not None else (
                self.cfg.max_iter if self.training else self.cfg.max_iter_eval)
            s_star, info = solve_equilibrium(step_fn, s0, self.cfg, self.training,
                                             max_iter=budget, record_trace=record_trace)
            self.last_info = info
            z = s_star[0]
            out = {"logits": self._readout(z), "info": info,
                   "q_halt_logits": self.q_head(z.mean(1))[:, 0]}
            pen = self._contraction_penalty(step_fn, s_star)
            if pen is not None:
                out["contraction_loss"] = pen
            return out
        finally:
            self.block.clear_dropout_masks()

    # -- contraction control ---------------------------------------------
    def _contraction_penalty(self, step_fn, s_star):
        """Spectral-radius-targeted contraction regulariser.

        A looped model has no incentive to stay contractive: nothing in a task
        loss punishes an expansive update map, and in practice the spectral
        radius of an unregularised loop climbs past 1 within a few hundred
        steps. At that point the "fixed point" does not exist, the forward
        solver runs to its cap, and the implicit gradient is being evaluated at
        a point that is not an equilibrium -- the entire method rests on
        rho < 1 actually holding.

        The usual Jacobian regulariser (Hutchinson estimate of ||J||_F^2, as in
        FPRM) is a poor instrument here: averaged over a state of a few thousand
        coordinates it is diluted by that dimension, so it barely moves the one
        eigenvalue that decides whether the loop converges. We instead estimate
        the *spectral radius* directly by finite-difference power iteration and
        apply a one-sided hinge at a target below 1, which leaves the model free
        to use all the capacity it wants right up to the stability boundary and
        pushes back only when it crosses.

        Cost: ``2 (n_power + 1)`` extra single-step forwards, no double backward.
        """
        cfg = self.cfg
        if not self.training or cfg.contraction_lambda <= 0:
            return None
        if cfg.max_iter <= 1:
            # Non-recursive control: there is no loop to keep contractive, and
            # charging it the regulariser's extra forwards would distort the
            # per-step timing it is being compared on.
            return None
        s0 = s_star.detach()
        # Relative finite-difference step. An absolute epsilon is badly scaled
        # here: the probe direction is normalised over the whole state, so each
        # coordinate is perturbed by only eps/sqrt(numel), which in float32
        # leaves the difference dominated by round-off and biases the estimate
        # high. Scaling by ||s|| makes the perturbation a fixed *relative* size.
        eps = cfg.jacobian_eps * s0.norm().clamp_min(1e-6)

        def fd_jvp(v):
            return (step_fn(s0 + eps * v) - step_fn(s0 - eps * v)) / (2 * eps)

        v = self._power_vec
        if v is None or v.shape != s0.shape or not torch.isfinite(v).all():
            v = torch.randn_like(s0)
        v = v / v.norm().clamp_min(1e-12)

        with torch.no_grad():
            for _ in range(cfg.n_power_iterations):
                jv = fd_jvp(v)
                n = jv.norm()
                if n < 1e-12:
                    break
                v = jv / n
        self._power_vec = v.detach()

        sigma = fd_jvp(v).norm()          # differentiable estimate of rho(J)
        self.last_rho = float(sigma.detach())
        return torch.relu(sigma - cfg.contraction_target).pow(2)

    # -- Universal-Transformer-style ACT baseline ------------------------
    def _forward_act(self, x_inj, seq_info, max_iter=None):
        """PonderNet/ACT halting over the outer loop, trained with full BPTT."""
        B, N, _ = x_inj.shape
        T = max_iter if max_iter is not None else (
            self.cfg.max_iter if self.training else self.cfg.max_iter_eval)
        s = self.block.init_state(B, N, x_inj.device, x_inj.dtype)

        remainder = x_inj.new_ones(B)
        weighted_logits = None
        ponder = x_inj.new_zeros(B)
        halt_probs = []
        for t in range(T):
            s = s + self.cfg.stepsize * (self.block.joint_step(s, x_inj, seq_info) - s)
            z = s[0]
            p = torch.sigmoid(self.q_head(z.mean(1))[:, 0])
            p = p if t < T - 1 else torch.ones_like(p)
            w = remainder * p
            logits = self._readout(z)
            weighted_logits = w.view(-1, 1, 1) * logits if weighted_logits is None \
                else weighted_logits + w.view(-1, 1, 1) * logits
            ponder = ponder + remainder * (t + 1)
            remainder = remainder * (1 - p)
            halt_probs.append(w)

        from .solvers import SolverInfo
        info = SolverInfo(n_iters=T, rel_residual=0.0, converged_frac=1.0)
        self.last_info = info
        return {"logits": weighted_logits, "info": info,
                "ponder_cost": ponder.mean(),
                "q_halt_logits": torch.stack(halt_probs, -1).sum(-1)}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def lipschitz_report(self):
        return self.block.lipschitz_bounds()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(arch: str, **cfg_kwargs) -> FPSAReasoner:
    from .config import build_config
    cfg = build_config(arch, **cfg_kwargs)
    return FPSAReasoner(cfg)

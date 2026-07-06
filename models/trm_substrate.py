"""TRM-substrate integration of the fixed-point program (the 4-point plan).

One model class, four orthogonal axes, all committed-config selectable:

  value_mode : fixed | evolving | blended | fixed_ffn | ace_relaxed
               ('fixed' here = frozen-anchored values from [x ; y_t], the
                SAFE coupling topology: values move only between outer
                segments, never inside a solve — the blended-3 postmortem
                rule. 'blended'/'evolving' reintroduce live-state values
                as explicitly-flagged risk arms.)
  backward   : neumann_k | bptt_k | phantom1     (H1 axis, matched compute:
               identical no-grad forward solve; backward budget k =
               k backward passes through f for BOTH estimators)
  supervision: TRM-style deep supervision — T outer segments, loss on the
               answer head every segment, y detached between segments.
  halting    : per-token adaptive freezing with a-posteriori certificate
               res_i / (1 - lam_hat_i) < eps   (a-priori certified only
               under the ACE arm; labeled accordingly).

TRM lineage kept: (z, y) two-stream recursion, deep supervision, shared
recurrent computation. Deviations (explicit): fixed-point inner solve with
residual halting instead of fixed n; separate z/y attention cores instead
of TRM's single shared net (v0); no ACT head (certificate replaces it).
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fpsa_block import RoPE1D, RoPE2D
from models.ace import ACEBlock, ball_project


class _Core(nn.Module):
    """Attention core with pluggable value source; pre-norm; RoPE."""

    def __init__(self, d, h, pos_mode="1d", grid_hw=None, max_seq_len=128,
                 ffn_mult=0.0):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(h))
        self.rope = RoPE2D(self.dh, grid_hw) if pos_mode == "2d" else \
            RoPE1D(self.dh, max_seq_len)
        self.ffn = None
        if ffn_mult > 0:
            m = int(d * ffn_mult)
            self.ffn_norm = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, m), nn.GELU(), nn.Linear(m, d))

    def _sp(self, t):
        B, N, D = t.shape
        return t.view(B, N, self.h, self.dh).transpose(1, 2)

    def attend(self, qk_src, values):
        q, k = self._sp(self.q(qk_src)), self._sp(self.k(qk_src))
        q, k = self.rope(q, k)
        tau = self.log_tau.exp().view(1, -1, 1, 1)
        a = F.softmax((q @ k.transpose(-2, -1)) / (math.sqrt(self.dh) * tau), -1)
        B, N, D = values.shape
        out = (a @ self._sp(self.v(values))).transpose(1, 2).reshape(B, N, D)
        out = self.o(out)
        return out


class _Pack:
    def __init__(self, fzA, z0, k):
        self.fzA, self.z0, self.k = fzA, z0, k


class _ImplicitAttach(torch.autograd.Function):
    """Forward: pass z* through unchanged. Backward: turn the incoming
    gradient g = dL/dz* into lam = sum_{j<k} (J^T)^j g via graph A, and
    RETURN lam as the gradient for graph B's output — the engine then
    deposits lam^T df_z/d(theta, x, y_t) through graph B natively."""

    @staticmethod
    def forward(ctx, z_value, fzB_out, pack):
        ctx.pack = pack
        return z_value

    @staticmethod
    def backward(ctx, g):
        p = ctx.pack
        lam = g
        for _ in range(p.k - 1):
            lam = torch.autograd.grad(p.fzA, p.z0, lam, retain_graph=True)[0] + g
        ctx.pack = None
        p.fzA = p.z0 = None
        return None, lam, None


class TRMSubstrate(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=64, num_heads=4,
                 value_mode="fixed", pos_mode="1d", grid_hw=None,
                 max_seq_len=128, n_inner=8, T_outer=4, alpha=0.5,
                 tol=1e-3, ffn_mult=2.0,
                 backward="neumann_k", bwd_k=6,
                 freeze_eps=0.0,   # 0 = freezing off; >0 = per-token halting
                 ace_beta=(0.2, 0.8), ace_R=3.0):
        super().__init__()
        self.vm, self.backward, self.bwd_k = value_mode, backward, bwd_k
        self.n_inner, self.T_outer, self.alpha, self.tol = n_inner, T_outer, alpha, tol
        self.freeze_eps = freeze_eps
        self.embed = nn.Embedding(vocab_size, d_model)
        self.ln_x, self.ln_y, self.ln_z = (nn.LayerNorm(d_model) for _ in range(3))
        use_ffn = ffn_mult if value_mode == "fixed_ffn" else 0.0
        self.z_core = _Core(d_model, num_heads, pos_mode, grid_hw, max_seq_len, use_ffn)
        self.y_core = _Core(d_model, num_heads, pos_mode, grid_hw, max_seq_len)
        if value_mode == "blended":
            self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        if value_mode == "ace_relaxed":
            # PORT of the validated ACE block (relaxed): ball projection in
            # place of in-loop LN, exact-SN weights, GroupSort FFN, anchor.
            self.ace_block = ACEBlock(d_model, num_heads, R=ace_R,
                                      beta_min=ace_beta[0], beta_max=ace_beta[1],
                                      certified=False, pos_mode=pos_mode,
                                      grid_hw=grid_hw, max_seq_len=max_seq_len)
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    # ---------- value construction (the topology rule lives here) ----------
    def _values(self, x, y_t, z):
        """x, y_t are FROZEN within an inner solve; z is live state."""
        nx, ny = self.ln_x(x), self.ln_y(y_t)
        anchor = 0.5 * (nx + ny)                     # frozen [x ; y_t] memory
        if self.vm in ("fixed", "fixed_ffn", "ace_relaxed"):
            return anchor
        nz = self.ln_z(z)
        if self.vm == "evolving":
            return nz                                # RISK ARM: live values
        if self.vm == "blended":                     # RISK ARM: live gate
            b = self.gate(torch.cat([x, z], -1))
            return (1 - b) * anchor + b * nz
        raise ValueError(self.vm)

    # ---------- the inner map f_z ----------
    def ace_ctx(self, x, y_t):
        return self.ace_block.precompute(0.5 * (x + y_t))

    def f_z(self, z, x, y_t, ace_static=None):
        if self.vm == "ace_relaxed":
            st = ace_static if ace_static is not None else self.ace_ctx(x, y_t)
            return self.ace_block.f(z, None, st)
        u = self.z_core.attend(self.ln_z(z), self._values(x, y_t, z))
        z_raw = x + u
        if self.z_core.ffn is not None:
            z_raw = z_raw + self.z_core.ffn(self.z_core.ffn_norm(z_raw))
        return (1 - self.alpha) * z + self.alpha * z_raw

    def f_y(self, y, z):
        y_raw = z + self.y_core.attend(self.ln_y(y), self.ln_z(z))
        return (1 - self.alpha) * y + self.alpha * y_raw

    # ---------- inner solve with optional per-token adaptive freezing ------
    def _solve_z(self, x, y_t, z0, rec):
        st = self.ace_ctx(x, y_t) if self.vm == "ace_relaxed" else None
        if st is not None:
            z0 = ball_project(z0, self.ace_block.R)
        z = z0
        prev_res = None
        frozen = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)
        used = torch.zeros(z.shape[:2], device=z.device)
        for k in range(self.n_inner):
            zn = self.f_z(z, x, y_t, ace_static=st)
            res = (zn - z).norm(dim=-1) / z.norm(dim=-1).clamp(min=1e-8)   # (B,N)
            if self.freeze_eps > 0:
                lam = (res / prev_res.clamp(min=1e-12)).clamp(max=0.999) \
                    if prev_res is not None else torch.ones_like(res) * 0.999
                cert = res / (1 - lam)                              # a-posteriori
                newly = (~frozen) & (cert < self.freeze_eps)
                frozen |= newly
                zn = torch.where(frozen.unsqueeze(-1), z, zn)
            used += (~frozen).float()
            prev_res = res
            z = zn
            rec["res_z"].append(res.mean().item())
            if res[~frozen].numel() == 0 or res[~frozen].max() < self.tol:
                break
        rec["token_iters"] = used.mean().item()
        rec["frozen_frac"] = frozen.float().mean().item()
        return z

    # ---------- one supervised segment with the H1 backward axis ----------
    def segment(self, x, y_t, rec):
        with torch.no_grad():
            z_star = self._solve_z(x, y_t, y_t, rec)
        if not torch.is_grad_enabled():          # eval fast path: no graphs
            return self.f_y(y_t, z_star)

        if self.backward == "bptt_full":     # VALIDATION reference only
            st = self.ace_ctx(x, y_t) if self.vm == "ace_relaxed" else None
            z = ball_project(y_t, self.ace_block.R) if st is not None else y_t
            for _ in range(self.n_inner):
                z = self.f_z(z, x, y_t, ace_static=st)
            return self.f_y(y_t, z)

        if self.backward == "bptt_k":
            # truncated BPTT: re-materialize last k inner steps WITH grad
            # -> k backward passes through f_z at loss.backward() time.
            st = self.ace_ctx(x, y_t) if self.vm == "ace_relaxed" else None
            z = z_star.detach()
            for _ in range(self.bwd_k):
                z = self.f_z(z, x, y_t, ace_static=st)
            return self.f_y(y_t, z)

        # implicit family (neumann_k / phantom1), single-engine-pass form:
        #   graph A (VJP series): built at x.detach() — J = df_z/dz at fixed x
        #   graph B (deposit)   : live x/y_t; receives lam AS ITS GRADIENT via
        #                         the custom Function below, so the engine
        #                         traverses it inside the SAME backward pass
        #                         (no nested backward, no shared-graph frees).
        # matched compute: (k-1) VJPs through A + 1 backward through B
        #                = k backward passes through f_z, same as bptt_k.
        k = 1 if self.backward == "phantom1" else self.bwd_k
        z0 = z_star.detach().clone().requires_grad_(True)
        stA = self.ace_ctx(x.detach(), y_t.detach()) if self.vm == "ace_relaxed" else None
        stB = self.ace_ctx(x, y_t) if self.vm == "ace_relaxed" else None
        fzA = self.f_z(z0, x.detach(), y_t.detach(), ace_static=stA)
        fzB = self.f_z(z_star.detach(), x, y_t, ace_static=stB)
        z_att = _ImplicitAttach.apply(z_star.detach(), fzB, _Pack(fzA, z0, k))
        return self.f_y(y_t, z_att)

    def forward(self, tokens):
        x = self.embed(tokens)
        y = x
        rec = {"res_z": [], "seg_losses": None}
        logits_per_seg = []
        for t in range(self.T_outer):
            y = self.segment(x, y, rec)
            logits_per_seg.append(self.head(self.head_norm(y)))
            y = y.detach()               # TRM-style segment detachment
        rec["rounds"] = self.T_outer
        return logits_per_seg, rec

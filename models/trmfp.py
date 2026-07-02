"""TRM-FP: TRM-style coupled fixed point, solved and differentiated properly.

Joint system (see chat derivation / docs):
    z* = f_z(z*; x, y*)     latent reasoning stream
    y* = f_y(y*; z*)        answer stream
f_z uses BLENDED-3 values: per-token, per-channel simplex gate
    (bx, by, bz)_j = softmax over 3 logits  ->  m_j = bx*LN(x)+by*LN(y)+bz*LN(z)
f_y: queries from LN(y), keys/values from LN(z), input-injected by z.

Solve: outer Gauss-Seidel rounds (inner z-Picard to tol, then damped y step),
joint residual logged. Backward: STACKED Neumann implicit gradient on
w = (z, y):  lam = (I - J_w^T)^{-1} (0, g_y), implemented with three separate
graphs (VJP graph / z-param-deposit graph / hooked y-output graph) to avoid
the hook-recursion and closure-leak failure modes documented in
docs/ARCHITECTURE.md.

v0 deviations from TRM (explicit): two separate blocks instead of TRM's
single shared net; no deep supervision; answer stream is token-aligned.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fpsa_block import RoPE1D


class _AttnCore(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len):
        super().__init__()
        self.h, self.dh = num_heads, d_model // num_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(num_heads))
        self.rope = RoPE1D(self.dh, max_seq_len)

    def _sp(self, t):
        B, N, D = t.shape
        return t.view(B, N, self.h, self.dh).transpose(1, 2)

    def attend(self, q_src, k_src, values):
        q, k = self._sp(self.q(q_src)), self._sp(self.k(k_src))
        q, k = self.rope(q, k)
        tau = self.log_tau.exp().view(1, -1, 1, 1)
        a = F.softmax((q @ k.transpose(-2, -1)) / (math.sqrt(self.dh) * tau), -1)
        B, N, D = values.shape
        out = (a @ self._sp(self.v(values))).transpose(1, 2).reshape(B, N, D)
        return self.o(out)


class TRMFPModel(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=64, num_heads=4,
                 alpha_z=0.5, alpha_y=0.5, n_inner=8, T_outer=4,
                 tol=1e-3, neumann_steps=12, neumann_tol=1e-4,
                 max_seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.ln_x = nn.LayerNorm(d_model)
        self.ln_z = nn.LayerNorm(d_model)
        self.ln_y = nn.LayerNorm(d_model)
        self.blk_z = _AttnCore(d_model, num_heads, max_seq_len)
        self.blk_y = _AttnCore(d_model, num_heads, max_seq_len)
        self.gate = nn.Linear(3 * d_model, 3 * d_model)   # per-channel 3-way logits
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.d = d_model
        self.alpha_z, self.alpha_y = alpha_z, alpha_y
        self.n_inner, self.T_outer = n_inner, T_outer
        self.tol, self.neumann_steps, self.neumann_tol = tol, neumann_steps, neumann_tol

    # ---- the two maps (damped) ----
    def f_z(self, z, x, y):
        nx, ny, nz = self.ln_x(x), self.ln_y(y), self.ln_z(z)
        logits = self.gate(torch.cat([x, y, z], -1)).view(*z.shape, 3)
        b = F.softmax(logits, dim=-1)                       # (B,N,D,3) simplex
        m = b[..., 0] * nx + b[..., 1] * ny + b[..., 2] * nz
        upd = self.blk_z.attend(q_src=nz, k_src=nz, values=m)
        z_raw = x + upd
        return (1 - self.alpha_z) * z + self.alpha_z * z_raw, b

    def f_y(self, y, z):
        ny, nz = self.ln_y(y), self.ln_z(z)
        y_raw = z + self.blk_y.attend(q_src=ny, k_src=nz, values=nz)
        return (1 - self.alpha_y) * y + self.alpha_y * y_raw

    # ---- joint solve with instrumentation ----
    def solve(self, x, n_inner=None, T_outer=None, record=None):
        n_inner = n_inner or self.n_inner
        T_outer = T_outer or self.T_outer
        z, y = x, x
        rz = ry = float("inf")
        for t in range(T_outer):
            for k in range(n_inner):
                z_new, b = self.f_z(z, x, y)
                rz = ((z_new - z).norm(-1) / z.norm(-1).clamp(min=1e-8)).mean().item()
                z = z_new
                if record is not None:
                    record["res_z"].append(rz)
                if rz < self.tol:
                    break
            y_new = self.f_y(y, z)
            ry = ((y_new - y).norm(-1) / y.norm(-1).clamp(min=1e-8)).mean().item()
            y = y_new
            if record is not None:
                record["res_y"].append(ry)
            if max(rz, ry) < self.tol:
                break
        if record is not None:
            record["gate_mean"] = b.mean(dim=(0, 1, 2)).detach()   # (3,) bx,by,bz
            record["disagree"] = (self.ln_z(z) - self.ln_x(x)).abs().amax(-1).mean().item()
            record["joint_res"] = max(rz, ry)
            record["rounds"] = t + 1
        return z, y

    def forward(self, tokens, backward: str = "neumann",
                n_inner=None, T_outer=None):
        x = self.embed(tokens)
        rec = {"res_z": [], "res_y": []}
        if backward == "bptt" or not torch.is_grad_enabled():
            z, y = self.solve(x, n_inner, T_outer, record=rec)
            return self.head(self.out_norm(y)), rec

        with torch.no_grad():
            z, y = self.solve(x, n_inner, T_outer, record=rec)
        z_s, y_s = z.detach(), y.detach()

        # Graph A (VJP): leaves for the stacked Jacobian
        z0 = z_s.clone().requires_grad_(True)
        y0 = y_s.clone().requires_grad_(True)
        fzA, _ = self.f_z(z0, x, y0)
        fyA = self.f_y(y0, z0)
        # Graph B (z-path param deposit)
        fzB, _ = self.f_z(z_s, x, y_s)
        # Graph C (output path, hooked)
        y_out = self.f_y(y_s, z_s)

        stats_bwd = rec.setdefault("bwd_res", [])
        ns, ntol = self.neumann_steps, self.neumann_tol
        holder = {"A": (fzA, fyA, z0, y0), "B": fzB}

        def hook(g_y):
            fzA_, fyA_, z0_, y0_ = holder["A"]
            lz = torch.zeros_like(g_y)
            ly = g_y
            for _ in range(ns):
                vz, vy = torch.autograd.grad((fzA_, fyA_), (z0_, y0_),
                                             (lz, ly), retain_graph=True)
                lz_n, ly_n = vz, vy + g_y     # stacked g = (0, g_y)
                rel = ((lz_n - lz).norm() + (ly_n - ly).norm()) / \
                      (lz.norm() + ly.norm() + 1e-12)
                stats_bwd.append(rel.item())
                lz, ly = lz_n, ly_n
                if rel < ntol:
                    break
            torch.autograd.backward(holder["B"], lz)   # deposit z-path grads
            holder["A"] = holder["B"] = None            # break closure cycles
            return ly

        y_out.register_hook(hook)
        return self.head(self.out_norm(y_out)), rec


# ---- coupled-contraction audit: block Lipschitz a,b,c,d and rho ----
def coupled_audit(model, tokens, iters=25):
    model.eval()
    x = model.embed(tokens)
    with torch.no_grad():
        z, y = model.solve(x)
    est = {}
    for name, (out_fn, wrt) in {
        "a": (lambda z0, y0: model.f_z(z0, x, y0)[0], "z"),
        "b": (lambda z0, y0: model.f_z(z0, x, y0)[0], "y"),
        "c": (lambda z0, y0: model.f_y(y0, z0), "z"),
        "d": (lambda z0, y0: model.f_y(y0, z0), "y"),
    }.items():
        z0 = z.detach().clone().requires_grad_(True)
        y0 = y.detach().clone().requires_grad_(True)
        out = out_fn(z0, y0)
        leaf = z0 if wrt == "z" else y0
        u = torch.randn_like(leaf); u /= u.norm()
        nu = torch.tensor(0.0)
        for _ in range(iters):
            u = torch.autograd.grad(out, leaf, u, retain_graph=True)[0]
            nu = u.norm(); u = u / nu.clamp(min=1e-30)
        est[name] = nu.item()
    a, b, c, d = est["a"], est["b"], est["c"], est["d"]
    rho = ((a + d) + math.sqrt((a - d) ** 2 + 4 * b * c)) / 2
    model.train()
    return est, rho

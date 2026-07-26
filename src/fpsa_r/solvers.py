"""Forward fixed-point solvers and the Anderson/Neumann linear solvers used by
the backward pass.

The forward solver is a damped Picard iteration with FPRM's adaptive step-size
decay and per-sample halting; it additionally returns the *per-token* relative
residual, which FPSA-R uses to mask the adjoint (Appendix B.2 of the FPSA
paper: masking non-converged coordinates yields the exact gradient of the
restricted equilibrium problem).
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch


@dataclass
class SolverInfo:
    n_iters: int = 0
    rel_residual: float = float("inf")
    residual_trace: List[float] = field(default_factory=list)
    per_sample_iters: Optional[torch.Tensor] = None
    converged_frac: float = 0.0
    token_converged: Optional[torch.Tensor] = None   # (B, N) bool
    stepsize: float = 1.0


def _token_residual(s_next: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """(B, N) relative residual, max over the stacked state rows."""
    num = (s_next - s).norm(dim=-1)                       # (rows, B, N)
    den = s.norm(dim=-1).clamp_min(1e-8)
    return (num / den).amax(dim=0)


@torch.no_grad()
def picard_solve(step_fn: Callable[[torch.Tensor], torch.Tensor],
                 s0: torch.Tensor,
                 max_iter: int,
                 tol: float,
                 stepsize: float = 1.0,
                 stepsize_decay: float = 0.9,
                 decay_patience: int = 5,
                 record_trace: bool = False,
                 token_tol: Optional[float] = None):
    """Damped Picard iteration with adaptive damping and per-sample halting."""
    s = s0
    B = s.shape[1]
    dev = s.device
    alpha = torch.full((B, 1, 1), float(stepsize), device=dev, dtype=s.dtype)
    best = torch.full((B,), float("inf"), device=dev)
    patience = torch.full((B,), float(decay_patience), device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    iters = torch.zeros(B, dtype=torch.int32, device=dev)

    info = SolverInfo(stepsize=float(stepsize))
    tok_res = torch.full(s.shape[1:3], float("inf"), device=dev)

    for k in range(max_iter):
        s_new = step_fn(s)
        tok_res = _token_residual(s_new, s)
        r = tok_res.amax(dim=1)                              # (B,)

        a = alpha.unsqueeze(0) * (~done).view(1, B, 1, 1).to(s.dtype)
        s = s + a * (s_new - s)

        improved = r < best
        best = torch.where(improved, r, best)
        patience = torch.where(improved, float(decay_patience), patience - 1)
        adapt = (patience <= 0) & (r >= tol)
        patience = torch.where(adapt, float(decay_patience), patience)
        alpha = alpha * torch.where(adapt, stepsize_decay, 1.0).view(B, 1, 1).to(alpha.dtype)

        iters = iters + (~done).int()
        done = done | (r < tol) | (alpha.view(-1) < 1e-3)

        info.n_iters = k + 1
        info.rel_residual = float(r.max())
        if record_trace:
            info.residual_trace.append(float(r.mean()))
        if bool(done.all()):
            break

    info.per_sample_iters = iters
    info.converged_frac = float((tok_res < (token_tol or tol)).float().mean())
    info.token_converged = tok_res < (token_tol or tol)
    info.stepsize = float(alpha.mean())
    return s, info


def neumann_solve(vjp: Callable[[torch.Tensor], torch.Tensor], g: torch.Tensor,
                  max_iter: int = 12, tol: float = 1e-4,
                  mask: Optional[torch.Tensor] = None):
    """lambda <- J^T lambda + g, i.e. the truncated Neumann series for
    (I - J^T)^{-1} g."""
    lam = g.clone()
    n = 0
    rel = float("inf")
    for n in range(1, max_iter + 1):
        new = vjp(lam) + g
        if mask is not None:
            new = new * mask
        rel = float((new - lam).norm() / (lam.norm() + 1e-12))
        lam = new
        if rel < tol:
            break
    return lam, n, rel


def anderson_solve(vjp: Callable[[torch.Tensor], torch.Tensor], g: torch.Tensor,
                   max_iter: int = 12, tol: float = 1e-4, m: int = 5,
                   beta: float = 1.0, lam_reg: float = 1e-4,
                   mask: Optional[torch.Tensor] = None):
    """Anderson-accelerated solve of the same linear fixed point.

    Anderson mixing converges in markedly fewer VJPs than the plain Neumann
    series when ||J|| is close to 1 -- exactly the regime a well-trained
    reasoning model sits in, since a contraction factor near 1 is what lets the
    loop keep making progress for many steps.
    """
    shape = g.shape
    flat = lambda t: t.reshape(-1)
    x0 = g.reshape(-1)
    f0 = (vjp(g) + g).reshape(-1)
    if mask is not None:
        f0 = (f0.reshape(shape) * mask).reshape(-1)

    n_dim = x0.numel()
    m = max(1, min(m, max_iter))
    X = torch.zeros(m, n_dim, dtype=g.dtype, device=g.device)
    F = torch.zeros(m, n_dim, dtype=g.dtype, device=g.device)
    X[0], F[0] = x0, f0
    x, fx = f0, None
    rel = float("inf")
    k = 1

    for k in range(1, max_iter + 1):
        fx = (vjp(x.reshape(shape)) + g).reshape(-1)
        if mask is not None:
            fx = (fx.reshape(shape) * mask).reshape(-1)
        rel = float((fx - x).norm() / (x.norm() + 1e-12))
        if rel < tol:
            x = fx
            break

        idx = k % m
        X[idx], F[idx] = x, fx
        n_hist = min(k + 1, m)

        G = F[:n_hist] - X[:n_hist]                       # (n_hist, dim)
        H = G @ G.t()
        H = H + lam_reg * torch.eye(n_hist, dtype=G.dtype, device=G.device) * (
            H.diagonal().mean().abs() + 1e-8)
        ones = torch.ones(n_hist, 1, dtype=G.dtype, device=G.device)
        try:
            alpha = torch.linalg.solve(H, ones)
        except Exception:
            alpha = ones
        alpha = alpha / (alpha.sum() + 1e-12)
        alpha = alpha.squeeze(-1)

        x = beta * (alpha @ F[:n_hist]) + (1 - beta) * (alpha @ X[:n_hist])
        if not torch.isfinite(x).all():          # fall back to a Picard step
            x = fx

    out = x.reshape(shape)
    if mask is not None:
        out = out * mask
    return out, k, rel

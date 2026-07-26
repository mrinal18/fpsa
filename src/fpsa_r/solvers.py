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


@torch.no_grad()
def broyden_solve(step_fn: Callable[[torch.Tensor], torch.Tensor],
                  s0: torch.Tensor,
                  max_iter: int,
                  tol: float,
                  m: int = 8,
                  line_search: bool = True,
                  token_tol: Optional[float] = None):
    """Limited-memory Broyden root-find on ``g(s) = f(s) - s = 0``.

    Picard iteration converges only when the map contracts; a quasi-Newton root
    find does not care. It builds a low-rank approximation to ``(I - J)^{-1}``
    from the secant pairs it has seen and takes Newton-like steps, so it can
    locate fixed points that Picard walks away from -- including unstable ones.
    This is what production DEQ implementations use, and it is the reason the
    contraction requirement is properly a property of the *solver*, not of
    implicit differentiation.

    Stores ``2m`` vectors of the state's size: still O(1) in iteration count.
    """
    s = s0.clone()
    shape = s.shape
    n = s.numel()
    gx = (step_fn(s) - s).reshape(-1)

    Us = torch.zeros(m, n, dtype=s.dtype, device=s.device)   # low-rank -U V^T
    Vs = torch.zeros(m, n, dtype=s.dtype, device=s.device)
    n_hist = 0
    info = SolverInfo()
    tok_res = torch.full(s.shape[1:3], float("inf"), device=s.device)

    for k in range(max_iter):
        # direction: -B g, with B = -I + sum U V^T  (B approximates (I-J)^{-1})
        d = -gx
        if n_hist:
            d = d + Us[:n_hist].t() @ (Vs[:n_hist] @ gx)
        d = -d

        alpha = 1.0
        s_new = (s.reshape(-1) + alpha * d).reshape(shape)
        g_new = (step_fn(s_new) - s_new).reshape(-1)
        if line_search:
            for _ in range(4):                    # simple backtracking
                if g_new.norm() < gx.norm() or not torch.isfinite(g_new).all():
                    break
                alpha *= 0.5
                s_new = (s.reshape(-1) + alpha * d).reshape(shape)
                g_new = (step_fn(s_new) - s_new).reshape(-1)
        if not torch.isfinite(g_new).all():
            break

        ds = (s_new - s).reshape(-1)
        dg = g_new - gx
        with torch.no_grad():
            tok_res = (s_new - s).norm(dim=-1) / s.norm(dim=-1).clamp_min(1e-8)
            tok_res = tok_res.amax(dim=0)
        s, gx = s_new, g_new

        # "Good" Broyden update of the inverse, kept low-rank:
        #   B <- B + (ds - B dg) (B^T ds)^T / ((B^T ds) . dg)
        # The right factor is B^T ds, not ds -- with B = -I + sum u_i v_i^T that
        # is -ds + V^T (U ds), which is what makes the secant condition hold.
        Bdg = -dg
        if n_hist:
            Bdg = Bdg + Us[:n_hist].t() @ (Vs[:n_hist] @ dg)
        BTds = -ds
        if n_hist:
            BTds = BTds + Vs[:n_hist].t() @ (Us[:n_hist] @ ds)
        denom = BTds @ dg
        if denom.abs() > 1e-12:
            u = (ds - Bdg) / denom
            idx = n_hist % m
            Us[idx], Vs[idx] = u, BTds
            n_hist = min(n_hist + 1, m)

        info.n_iters = k + 1
        rel = float(gx.norm() / (s.reshape(-1).norm() + 1e-8))
        info.rel_residual = rel
        info.residual_trace.append(rel)
        if rel < tol:
            break

    info.converged_frac = float((tok_res < (token_tol or tol)).float().mean())
    info.token_converged = tok_res < (token_tol or tol)
    return s, info


@torch.no_grad()
def anderson_forward(step_fn: Callable[[torch.Tensor], torch.Tensor],
                     s0: torch.Tensor, max_iter: int, tol: float,
                     m: int = 6, beta: float = 1.0, lam_reg: float = 1e-4,
                     token_tol: Optional[float] = None):
    """Anderson-accelerated forward iteration. Cheaper than Broyden and still
    converges on maps where plain Picard oscillates."""
    shape = s0.shape
    n = s0.numel()
    X = torch.zeros(m, n, dtype=s0.dtype, device=s0.device)
    F = torch.zeros(m, n, dtype=s0.dtype, device=s0.device)
    X[0] = s0.reshape(-1)
    F[0] = step_fn(s0).reshape(-1)
    x = F[0]
    info = SolverInfo()
    tok_res = torch.full(s0.shape[1:3], float("inf"), device=s0.device)

    for k in range(1, max_iter + 1):
        fx = step_fn(x.reshape(shape)).reshape(-1)
        prev = x
        n_hist = min(k + 1, m)
        idx = k % m
        X[idx], F[idx] = x, fx
        G = F[:n_hist] - X[:n_hist]
        H = G @ G.t()
        H = H + lam_reg * torch.eye(n_hist, dtype=G.dtype, device=G.device) * (
            H.diagonal().mean().abs() + 1e-8)
        ones = torch.ones(n_hist, 1, dtype=G.dtype, device=G.device)
        try:
            a = torch.linalg.solve(H, ones)
        except Exception:
            a = ones
        a = (a / (a.sum() + 1e-12)).squeeze(-1)
        x = beta * (a @ F[:n_hist]) + (1 - beta) * (a @ X[:n_hist])
        if not torch.isfinite(x).all():
            x = fx
        s_new, s_old = x.reshape(shape), prev.reshape(shape)
        tok_res = ((s_new - s_old).norm(dim=-1)
                   / s_old.norm(dim=-1).clamp_min(1e-8)).amax(dim=0)
        info.n_iters = k
        info.rel_residual = float(tok_res.max())
        info.residual_trace.append(info.rel_residual)
        if info.rel_residual < tol:
            break

    info.converged_frac = float((tok_res < (token_tol or tol)).float().mean())
    info.token_converged = tok_res < (token_tol or tol)
    return x.reshape(shape), info


def gmres_solve(vjp: Callable[[torch.Tensor], torch.Tensor], g: torch.Tensor,
                max_iter: int = 20, tol: float = 1e-4,
                mask: Optional[torch.Tensor] = None, restart: int = 20):
    """Solve ``(I - J^T) lambda = g`` with restarted GMRES.

    The Neumann series is a *stationary* iteration: it converges iff the
    spectral radius is below 1, and slowly when it is close to it. GMRES is a
    Krylov method -- it minimises the residual over the whole Krylov subspace
    and converges whenever ``I - J^T`` is invertible, whatever the spectral
    radius. Swapping it in removes the contraction requirement from the backward
    pass entirely, at the same cost of one VJP per iteration.
    """
    shape = g.shape
    b = g.reshape(-1)

    def A(v):
        out = v.reshape(shape) - vjp(v.reshape(shape))
        if mask is not None:
            out = out * mask
        return out.reshape(-1)

    x = torch.zeros_like(b)
    bnorm = b.norm()
    if bnorm < 1e-12:
        return g, 0, 0.0

    total = 0
    rel = 1.0
    for _ in range(max(1, -(-max_iter // restart))):
        r = b - A(x)
        beta = r.norm()
        rel = float(beta / bnorm)
        if rel < tol:
            break
        k_max = min(restart, max_iter - total)
        if k_max <= 0:
            break
        Q = torch.zeros(k_max + 1, b.numel(), dtype=b.dtype, device=b.device)
        H = torch.zeros(k_max + 1, k_max, dtype=b.dtype, device=b.device)
        Q[0] = r / beta
        k_used = 0
        for j in range(k_max):
            w = A(Q[j])
            total += 1
            for i in range(j + 1):                  # modified Gram-Schmidt
                H[i, j] = torch.dot(Q[i], w)
                w = w - H[i, j] * Q[i]
            H[j + 1, j] = w.norm()
            k_used = j + 1
            if H[j + 1, j] < 1e-12:
                break
            Q[j + 1] = w / H[j + 1, j]
        e1 = torch.zeros(k_used + 1, dtype=b.dtype, device=b.device)
        e1[0] = beta
        y, *_ = torch.linalg.lstsq(H[:k_used + 1, :k_used], e1.unsqueeze(-1))
        x = x + Q[:k_used].t() @ y.squeeze(-1)
        rel = float((b - A(x)).norm() / bnorm)
        if rel < tol or total >= max_iter:
            break

    out = x.reshape(shape)
    if mask is not None:
        out = out * mask
    if not torch.isfinite(out).all():
        return g, total, float("inf")
    return out, total, rel


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

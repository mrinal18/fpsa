"""Auditable, per-example fixed-point and adjoint solves.

Maps must be deterministic and batch-separable. No claim of global contraction
is made. Forward residuals describe the exact returned tensor, and every map/VJP
call is counted. A finite unroll is explicitly not an equilibrium gradient.
"""
from dataclasses import dataclass, field
from typing import Callable
import torch

Tensor = torch.Tensor


@dataclass
class SolverConfig:
    max_iter: int = 48
    tol: float = 1e-4
    damping: float = 0.8
    history: int = 6
    ridge: float = 1e-4
    backward_max_iter: int = 64
    backward_tol: float = 1e-5
    restart: int = 16

    def __post_init__(self):
        if min(self.max_iter, self.history, self.backward_max_iter, self.restart) < 1:
            raise ValueError("Iteration counts must be positive")
        if min(self.tol, self.backward_tol) <= 0 or self.ridge < 0:
            raise ValueError("Invalid tolerances/ridge")
        if not 0 < self.damping <= 1:
            raise ValueError("damping must be in (0, 1]")


@dataclass
class SolveInfo:
    iterations: int = 0
    nfe: int = 0
    residual: Tensor | None = None  # per sample, maximum across token rows
    converged: Tensor | None = None
    sample_iterations: Tensor | None = None
    trace: list = field(default_factory=list)
    backward_vjps: int = 0
    backward_map_evals: int = 0
    backward_residual: float | None = None
    is_equilibrium: bool = True

    def as_dict(self):
        return dict(iterations=self.iterations, nfe=self.nfe,
                    max_residual=float(self.residual.max()),
                    converged_fraction=float(self.converged.float().mean()),
                    backward_vjps=self.backward_vjps,
                    backward_map_evals=self.backward_map_evals,
                    backward_residual=self.backward_residual,
                    is_equilibrium=self.is_equilibrium)


class ConvergenceError(RuntimeError):
    def __init__(self, message: str, info: SolveInfo):
        super().__init__(message)
        self.info = info


def residual(phi: Tensor, state: Tensor) -> Tensor:
    if phi.shape != state.shape or state.ndim != 3:
        raise ValueError("Expected equally shaped (batch, tokens, channels) tensors")
    dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
    phi, state = phi.to(dtype), state.to(dtype)
    if not torch.isfinite(phi).all() or not torch.isfinite(state).all():
        raise FloatingPointError("Non-finite fixed-point state")
    return ((phi - state).norm(dim=-1) /
            (phi.norm(dim=-1) + state.norm(dim=-1) + 1e-8)).amax(dim=1)


@torch.no_grad()
def anderson(phi: Callable[[Tensor], Tensor], initial: Tensor,
             cfg: SolverConfig) -> tuple[Tensor, SolveInfo]:
    """Anderson mixing with best-true-residual return, one system per example."""
    if initial.ndim != 3 or initial.dtype not in (torch.float32, torch.float64):
        raise ValueError("Solve in float32/float64 with shape (B,N,D), not BF16")
    b = initial.shape[0]
    x = initial.clone()
    best = x.clone()
    best_r = torch.full((b,), float('inf'), device=x.device, dtype=x.dtype)
    done = torch.zeros(b, dtype=torch.bool, device=x.device)
    counts = torch.zeros(b, dtype=torch.long, device=x.device)
    xs, fs = [], []
    info = SolveInfo()
    for k in range(cfg.max_iter):
        f = phi(x)
        info.nfe += 1
        r = residual(f, x)
        better = r < best_r
        best = torch.where(better[:, None, None], x, best)
        best_r = torch.minimum(r, best_r)
        counts += (~done).long()
        done |= r <= cfg.tol
        info.iterations = k + 1
        info.trace.append(float(r.max()))
        if bool(done.all()):
            break
        xs.append(x.flatten(1))
        fs.append(f.flatten(1))
        xs, fs = xs[-cfg.history:], fs[-cfg.history:]
        if len(xs) == 1:
            mixed = f
        else:
            X, F = torch.stack(xs, 1), torch.stack(fs, 1)
            g = (F - X).double()
            gram = g @ g.transpose(1, 2)
            scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
            eye = torch.eye(len(xs), device=x.device, dtype=torch.float64)
            system = gram + cfg.ridge * scale[:, None, None] * eye
            ones = torch.ones(b, len(xs), 1, device=x.device, dtype=torch.float64)
            a, status = torch.linalg.solve_ex(system, ones)
            denom = a.sum(1, keepdim=True)
            bad = (status != 0) | ~torch.isfinite(a).all((1, 2)) | (denom[:, 0, 0].abs() < 1e-12)
            a[bad], denom[bad] = 1.0, float(len(xs))
            a = (a / denom).to(x.dtype)
            mixed = (a * F).sum(1).reshape_as(x)
        candidate = x + cfg.damping * (mixed - x)
        fallback = x + cfg.damping * (f - x)
        valid = torch.isfinite(candidate).flatten(1).all(1)
        candidate = torch.where(valid[:, None, None], candidate, fallback)
        x = torch.where(done[:, None, None], x, candidate)
    # Include the last update in the best-state comparison, then validate the
    # exact per-example mixture returned. These two real calls count as NFEs.
    f = phi(x)
    info.nfe += 1
    r = residual(f, x)
    best = torch.where((r < best_r)[:, None, None], x, best)
    f = phi(best)
    info.nfe += 1
    info.residual = residual(f, best)
    info.converged = info.residual <= cfg.tol
    info.sample_iterations = counts
    return best, info


@torch.no_grad()
def gmres(vjp: Callable[[Tensor], Tensor], rhs: Tensor, *, max_iter=64,
          tol=1e-5, restart=16) -> tuple[Tensor, int, float]:
    """Solve (I-J^T)x=b. Independent bases; true residual check EACH column.

    This deliberately favors numerical auditing over peak speed: a column uses
    one operator call and one true-residual verification call. All VJPs count.
    Small least-squares systems use float64 SVD pseudoinverses (CUDA-compatible,
    including rank-deficient/happy-breakdown systems). Bases stay in FP32/64.
    """
    if rhs.ndim != 3 or rhs.dtype not in (torch.float32, torch.float64):
        raise ValueError("GMRES expects a float32/64 (B,N,D) RHS")
    if max_iter < 1 or restart < 1 or tol <= 0:
        raise ValueError("Invalid GMRES budget/tolerance")
    if not torch.isfinite(rhs).all():
        raise FloatingPointError("Non-finite adjoint RHS")
    shape = rhs.shape
    b = rhs.flatten(1)
    bnorm = b.double().norm(dim=1)
    nonzero = bnorm > 0
    denom = bnorm.clamp_min(torch.finfo(torch.float64).tiny)
    x = torch.zeros_like(b)
    best = x.clone()
    best_rel = nonzero.double()
    calls = 0

    def A(v):
        nonlocal calls
        calls += 1
        jv = vjp(v.reshape(shape))
        if jv.shape != shape:
            raise ValueError("VJP changed shape")
        result = v - jv.flatten(1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite adjoint operator")
        return result

    done = ~nonzero
    it = 0
    while it < max_iter and not bool(done.all()):
        r = b - A(x)
        beta = r.double().norm(dim=1)
        rel = torch.where(nonzero, beta / denom, 0.)
        done |= rel <= tol
        width = min(restart, max_iter - it, b.shape[1])
        V = torch.zeros(b.shape[0], width + 1, b.shape[1], device=b.device, dtype=b.dtype)
        H = torch.zeros(b.shape[0], width + 1, width, device=b.device, dtype=torch.float64)
        V[:, 0] = torch.where(done[:, None], 0., r / beta.clamp_min(1e-30).to(b.dtype)[:, None])
        cycle_base = x.clone()
        cycle_done = done.clone()
        e = torch.zeros(b.shape[0], width + 1, 1, device=b.device, dtype=torch.float64)
        e[:, 0, 0] = beta
        for j in range(width):
            active = ~cycle_done
            if not bool(active.any()):
                break
            w = A(V[:, j])
            it += 1
            # Two-pass modified Gram-Schmidt; multiply in double for dot products.
            for _ in range(2):
                for i in range(j + 1):
                    p = (V[:, i].double() * w.double()).sum(-1)
                    H[:, i, j] += p
                    w -= p.to(w.dtype)[:, None] * V[:, i]
            h = w.double().norm(dim=1)
            H[:, j + 1, j] = h
            breakdown = h <= 1e-12 * H[:, :j+1, j].abs().amax(1).clamp_min(1.)
            V[:, j + 1] = torch.where((active & ~breakdown)[:, None],
                                      w / h.clamp_min(1e-30).to(w.dtype)[:, None], 0.)
            hsmall = H[:, :j+2, :j+1]
            coeff = torch.linalg.pinv(hsmall, rtol=1e-12) @ e[:, :j+2]
            trial = cycle_base + (coeff.to(b.dtype) * V[:, :j+1]).sum(1)
            trial = torch.where(active[:, None], trial, x)
            true = b - A(trial)
            rel = torch.where(nonzero, true.double().norm(dim=1) / denom, 0.)
            improved = rel < best_rel
            best = torch.where(improved[:, None], trial, best)
            best_rel = torch.minimum(best_rel, rel)
            x = torch.where(active[:, None], trial, x)
            solved = rel <= tol
            done |= solved
            cycle_done |= solved | breakdown
        x = best.clone()
    # Verify returned solution; never report a recursively estimated residual.
    true = b - A(best) if nonzero.any() else b
    rel = torch.where(nonzero, true.double().norm(dim=1) / denom, 0.)
    return best.reshape(shape), calls, float(rel.max())


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fixed, mapped, phi, cfg, info):
        ctx.save_for_backward(fixed)
        ctx.phi, ctx.cfg, ctx.info = phi, cfg, info
        return fixed.clone()

    @staticmethod
    def backward(ctx, grad):
        (fixed,) = ctx.saved_tensors
        with torch.enable_grad():
            state = fixed.detach().requires_grad_(True)
            mapped = ctx.phi(state)
        ctx.info.backward_map_evals += 1

        def vjp(v):
            if not mapped.requires_grad:
                return torch.zeros_like(v)
            result = torch.autograd.grad(mapped, state, v, retain_graph=True,
                                         allow_unused=True)[0]
            return torch.zeros_like(v) if result is None else result

        lam, calls, rel = gmres(vjp, grad, max_iter=ctx.cfg.backward_max_iter,
                               tol=ctx.cfg.backward_tol, restart=ctx.cfg.restart)
        ctx.info.backward_vjps = calls
        ctx.info.backward_residual = rel
        if not torch.isfinite(lam).all() or rel > ctx.cfg.backward_tol:
            raise ConvergenceError(f"Adjoint residual {rel:.3e} > {ctx.cfg.backward_tol:.3e}", ctx.info)
        return None, lam, None, None, None


def solve(phi, initial, cfg: SolverConfig, *, mode='implicit', strict=True):
    """Attach an equilibrium or finite-unroll gradient with unchanged forward semantics."""
    if mode not in {'implicit', 'one_step', 'unroll'}:
        raise ValueError("Unknown gradient mode")
    if mode == 'unroll':
        x = initial
        for _ in range(cfg.max_iter):
            x = x + cfg.damping * (phi(x) - x)
        with torch.no_grad():
            r = residual(phi(x), x)
        return x, SolveInfo(iterations=cfg.max_iter, nfe=cfg.max_iter+1,
                            residual=r, converged=r <= cfg.tol,
                            sample_iterations=torch.full((x.shape[0],), cfg.max_iter,
                                                         device=x.device),
                            is_equilibrium=False)
    fixed, info = anderson(phi, initial, cfg)
    if strict and not bool(info.converged.all()):
        raise ConvergenceError(f"Forward residual {float(info.residual.max()):.3e} > {cfg.tol:.3e}", info)
    if not torch.is_grad_enabled():
        return fixed, info
    if not strict and not bool(info.converged.all()):
        raise ConvergenceError("Cannot attach equilibrium gradient to an unconverged state", info)
    mapped = phi(fixed)
    info.nfe += 1
    if mode == 'one_step':
        return fixed + (mapped - mapped.detach()), info
    return _Adjoint.apply(fixed, mapped, phi, cfg, info), info

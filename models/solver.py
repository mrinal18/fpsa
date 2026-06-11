"""Fixed-point solver + backward modes, with mandatory instrumentation.

Forward: damped Picard iteration (the prototype's solver). Anderson can be
added later behind the same interface.

Backward modes:
  bptt        : full unrolled backprop through every iteration (reference).
  phantom1    : 1-step / Jacobian-free gradient (HRM-style): backprop through
                the LAST iteration only. == neumann with 0 refinement terms.
  neumann     : phantom gradient refined by `neumann_steps` Neumann terms,
                lam <- J^T lam + g  (truncated (I - J^T)^{-1} g). This is the
                prototype's Neumann-RBP adjoint.

Every solve returns SolveStats: per-iteration mean/max relative residual
||z_{k+1}-z_k|| / ||z_k||, iterations run, fraction of tokens converged.
A run that doesn't log these is not a valid run.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn.utils.parametrize as parametrize


@dataclass
class SolveStats:
    rel_residual_mean: List[float] = field(default_factory=list)  # per iteration
    rel_residual_max: List[float] = field(default_factory=list)
    iterations: int = 0
    converged_frac: float = 0.0   # fraction of (batch, token) below tol at exit
    backward_residual: List[float] = field(default_factory=list)  # adjoint loop
    jac_sigma_est: Optional[torch.Tensor] = None  # differentiable sigma_max(J at z*)
    exit_residual_t: Optional[torch.Tensor] = None  # differentiable mean exit residual


def _rel_res(z_next, z):
    num = (z_next - z).norm(dim=-1)
    den = z.norm(dim=-1).clamp(min=1e-8)
    return num / den  # (B, N)


def fixed_point_solve(
    module,                      # FPSABlock (or any module exposing .f / .precompute)
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    z0: Optional[torch.Tensor] = None,
    max_iter: int = 16,
    tol: float = 1e-3,
    backward: str = "neumann",
    neumann_steps: int = 5,
    neumann_tol: float = 1e-4,
    jac_reg: bool = False,
    jac_power_iters: int = 2,
    freeze_sn: bool = True,
    stats_out: Optional[SolveStats] = None,
):
    """Solve z* = f(z*; x) and set up the requested backward. Returns (z*, stats)."""
    stats = stats_out if stats_out is not None else SolveStats()
    z = x if z0 is None else z0

    # Freeze spectral-norm power iteration within the solve so f is fixed.
    # freeze_sn=False reproduces the prototype bug: power iteration advances
    # on EVERY f call, so f mutates between FPI iterations and the f used for
    # the backward differs from the f that produced z*.
    import contextlib
    ctx = parametrize.cached() if freeze_sn else contextlib.nullcontext()
    with ctx:
        # Populate parametrization caches (lazy, on first access) WITH grad:
        # if first access happened inside the no_grad solve, the cached W/sigma
        # would be a constant and implicit-mode backward would silently return
        # zero gradient for spectral-normed weights (verified in debug_adjoint).
        if torch.is_grad_enabled():
            for sub in module.modules():
                if parametrize.is_parametrized(sub):
                    sub.weight
        static = module.precompute(x)

        if backward == "bptt":
            for k in range(max_iter):
                z_next = module.f(z, x, static, attn_mask)
                rr = _rel_res(z_next, z)
                stats.rel_residual_mean.append(rr.mean().item())
                stats.rel_residual_max.append(rr.max().item())
                z = z_next
                stats.iterations = k + 1
                if stats.rel_residual_max[-1] < tol:
                    break
            stats.converged_frac = float(
                (rr < tol).float().mean().item()) if max_iter > 0 else 1.0
            return z, stats

        # --- implicit-style: no-grad solve, then graph on final step(s) ---
        with torch.no_grad():
            for k in range(max_iter):
                z_next = module.f(z, x, static, attn_mask)
                rr = _rel_res(z_next, z)
                stats.rel_residual_mean.append(rr.mean().item())
                stats.rel_residual_max.append(rr.max().item())
                z = z_next
                stats.iterations = k + 1
                if stats.rel_residual_max[-1] < tol:
                    break
            stats.converged_frac = float((rr < tol).float().mean().item())

        z_star = z.detach()
        if not torch.is_grad_enabled():   # inference: no backward to set up
            return z_star, stats
        if backward == "phantom1":
            # gradient of one application of f at z*; J-free 1-step (HRM-like)
            return module.f(z_star, x, static, attn_mask), stats

        if backward == "neumann":
            # Separate graphs: f_vjp's graph serves the adjoint VJPs; the hook
            # lives on z_out (a distinct application of f), so VJP calls can't
            # re-fire the hook (that recursion OOMs).
            z0 = z_star.clone().requires_grad_(True)
            f_vjp = module.f(z0, x, static, attn_mask)

            if jac_reg:
                # Differentiable estimate of sigma_max(J) at z* via power
                # iteration on J^T (sigma_max(J^T) == sigma_max(J)). The
                # iteration map must be contractive (sigma < 1) for Picard
                # and the Neumann adjoint to converge; penalizing this
                # estimate keeps training inside the contractive regime
                # (Jacobian regularization, Bai et al. 2021 style but
                # targeting the spectral norm, not the Frobenius proxy).
                u = torch.randn_like(z_star)
                u = u / (u.norm() + 1e-12)
                for _ in range(jac_power_iters):
                    u = torch.autograd.grad(f_vjp, z0, u, retain_graph=True)[0]
                    u = u / (u.norm() + 1e-12)
                Ju = torch.autograd.grad(f_vjp, z0, u,
                                         retain_graph=True, create_graph=True)[0]
                stats.jac_sigma_est = Ju.norm()

            z_out = module.f(z_star, x, static, attn_mask)
            if jac_reg:
                # differentiable exit residual: direct handle on convergence
                # for guard-v2 escape when the sigma estimate is blind
                rr_t = (z_out - z_star).norm(dim=-1) / z_star.norm(dim=-1).clamp_min(1e-8)
                stats.exit_residual_t = rr_t.mean()

            def hook(grad):
                lam = grad
                for _ in range(neumann_steps):
                    vjp = torch.autograd.grad(
                        f_vjp, z0, lam, retain_graph=True)[0]
                    lam_new = vjp + grad
                    rel = ((lam_new - lam).norm() / (lam.norm() + 1e-12)).item()
                    stats.backward_residual.append(rel)
                    lam = lam_new
                    if rel < neumann_tol:
                        break
                return lam

            z_out.register_hook(hook)
            return z_out, stats

    raise ValueError(f"unknown backward mode {backward}")

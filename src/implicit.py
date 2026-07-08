"""Generic fixed-point solving and implicit differentiation utilities.

This module factors the Neumann-series implicit gradient ("phantom gradient" /
Neumann-RBP) out of the FPSA attention module so any fixed-point layer can use
it: the spatial FPSA model, the BERT FPSA attention, and the Implicit TRM.

Forward:  solve  z* = f(z*)  by damped iteration under no_grad.
Backward: given dL/dz*, solve the adjoint equation

    lambda = J_f(z*)^T lambda + dL/dz*      =>   lambda = (I - J^T)^{-1} dL/dz*

by fixed-point iteration (truncated Neumann series). Parameter/input gradients
then flow through ONE differentiable application of f at z*, which receives
lambda as its incoming gradient:  dL/dtheta = lambda^T df/dtheta|_{z*}.

Masked implicit differentiation: positions (tokens / samples) whose forward
iteration did not converge are excluded from the adjoint solve — both the
source term and the J^T products are zeroed there — which is the exact
gradient of the restricted equilibrium problem on the converged subset
(paper, Appendix D).
"""

from typing import Callable, Dict, Optional, Tuple

import torch


class _NeumannImplicitGrad(torch.autograd.Function):
    """Refine the phantom gradient with a truncated Neumann adjoint solve.

    forward(z_graph, z_star, f, steps, tol, mask) passes z_graph through
    unchanged. backward replaces the incoming gradient g = dL/dz_graph with

        lambda_N = sum_{t=0..N} (M J_f^T M)^t (M g)

    where M is the (optional) convergence mask. With N = 0 this reduces to the
    plain 1-step phantom gradient.
    """

    @staticmethod
    def forward(
        ctx,
        z_graph: torch.Tensor,
        z_star: torch.Tensor,
        f: Callable[[torch.Tensor], torch.Tensor],
        steps: int,
        tol: float,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx.f = f
        ctx.steps = steps
        ctx.tol = tol
        if mask is None:
            ctx.save_for_backward(z_star)
            ctx.has_mask = False
        else:
            ctx.save_for_backward(z_star, mask)
            ctx.has_mask = True
        return z_graph

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.has_mask:
            z_star, mask = ctx.saved_tensors
            # Broadcast mask (e.g. (B, N) or (B, N, 1)) over the hidden dim.
            while mask.dim() < grad_output.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.to(grad_output.dtype)
        else:
            (z_star,) = ctx.saved_tensors
            mask = None

        g = grad_output if mask is None else grad_output * mask

        if ctx.steps == 0:
            # Phantom gradient (still masked when a mask is given).
            return g, None, None, None, None, None

        z_var = z_star.detach().requires_grad_(True)
        with torch.enable_grad():
            z_next = ctx.f(z_var)

        # tol <= 0 disables the per-step relative-change check, which costs a
        # host sync (.item()) per adjoint iteration on GPU.
        check_tol = ctx.tol > 0
        lam = g
        for _ in range(ctx.steps):
            vjp = torch.autograd.grad(
                outputs=z_next,
                inputs=z_var,
                grad_outputs=lam,
                retain_graph=True,
                create_graph=False,
            )[0]
            if mask is not None:
                vjp = vjp * mask
            lam_new = vjp + g
            if check_tol:
                rel = (lam_new - lam).norm() / (lam.norm() + 1e-12)
                lam = lam_new
                if rel.item() < ctx.tol:
                    break
            else:
                lam = lam_new

        return lam, None, None, None, None, None


def attach_implicit_grad(
    z_star_detached: torch.Tensor,
    f: Callable[[torch.Tensor], torch.Tensor],
    *,
    adjoint_steps: int = 10,
    adjoint_tol: float = 1e-4,
    conv_mask: Optional[torch.Tensor] = None,
    min_conv_frac: float = 0.1,
) -> torch.Tensor:
    """Attach an implicit-gradient backward graph to a converged state.

    Args:
        z_star_detached: converged state, detached from any graph.
        f: differentiable update map; must close over its other inputs (x,
           masks, ...) so those receive gradients through the single call.
           IMPORTANT: the closure is re-invoked at backward time for the
           adjoint VJPs, so everything it captures (e.g. a dropout mask) must
           still be alive and identical then — bind such state explicitly in
           the closure rather than reading it from mutable module attributes.
        adjoint_steps: Neumann terms for the adjoint solve. 0 = phantom grad.
        adjoint_tol: relative-change stopping tolerance for the adjoint solve
           (<= 0 disables the check and always runs adjoint_steps terms).
        conv_mask: optional boolean/float mask of converged positions,
           broadcastable to z_star's leading dims. Non-converged positions are
           excluded from the adjoint solve (masked implicit differentiation).
           Applied in phantom mode too (the source term is masked).
        min_conv_frac: if fewer than this fraction of positions converged, the
           mask is dropped to avoid gradient starvation early in training.

    Returns:
        z_star with gradients: one differentiable application of f at z*,
        wrapped so the incoming gradient is refined by the Neumann solve.
    """
    if conv_mask is not None and conv_mask.float().mean() < min_conv_frac:
        conv_mask = None

    z_graph = f(z_star_detached)
    if adjoint_steps > 0 or conv_mask is not None:
        z_graph = _NeumannImplicitGrad.apply(
            z_graph, z_star_detached, f, adjoint_steps, adjoint_tol, conv_mask
        )
    return z_graph


@torch.no_grad()
def fixed_point_solve(
    f: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    *,
    tol: float = 1e-3,
    max_iter: int = 16,
    stepsize: float = 1.0,
    stepsize_decay: float = 1.0,
    decay_patience: int = 0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Damped fixed-point iteration z <- s*f(z) + (1-s)*z under no_grad.

    Any damping already inside f is composed with the outer stepsize s; both
    preserve the fixed-point set. When `decay_patience > 0` the per-sample
    stepsize is multiplied by `stepsize_decay` whenever the residual has not
    improved for `decay_patience` consecutive steps (patience-based damping, as
    in FPRM's FPOPT solver) — this suppresses period-2 oscillations that plain
    iteration cannot escape.

    Residuals are relative per token: ||z_new - z|| / (||z|| + eps) over the
    last dim; per-sample residual is the max over tokens.

    Returns (z_star, info) where info has:
        steps:          int tensor, iterations actually run
        residual:       (B, N) final per-token relative residual
        converged:      (B, N) bool, residual < tol
        sample_residual:(B,) max-over-tokens residual
        stepsize:       (B,) final per-sample stepsize
    """
    z = z0
    B = z.shape[0]
    device = z.device

    step = torch.full((B,), float(stepsize), device=device, dtype=torch.float32)
    best = torch.full((B,), float("inf"), device=device)
    patience = torch.full((B,), float(decay_patience), device=device)

    residual = torch.full(z.shape[:-1], float("inf"), device=device)
    sample_res = residual.flatten(1).max(dim=1).values if residual.dim() > 1 else residual
    steps_run = 0
    for k in range(max_iter):
        z_new = f(z)
        if stepsize != 1.0 or decay_patience > 0:
            s = step.view(-1, *([1] * (z.dim() - 1))).to(z.dtype)
            z_new = s * z_new + (1.0 - s) * z

        # Residuals in >= fp32: bf16 quantization floors relative residuals
        # around 2^-9, which would make tighter tolerances unreachable.
        num = (z_new - z).to(torch.promote_types(z.dtype, torch.float32)).norm(dim=-1)
        den = z.to(torch.promote_types(z.dtype, torch.float32)).norm(dim=-1).clamp_min(eps)
        residual = num / den                      # (B, N)
        sample_res = residual.flatten(1).max(dim=1).values if residual.dim() > 1 else residual

        z = z_new
        steps_run = k + 1

        if decay_patience > 0:
            improved = sample_res < best
            best = torch.minimum(sample_res, best)
            patience = torch.where(improved, torch.full_like(patience, float(decay_patience)), patience - 1)
            adapt = (patience <= 0) & (sample_res >= tol)
            patience = torch.where(adapt, torch.full_like(patience, float(decay_patience)), patience)
            step = torch.where(adapt, step * stepsize_decay, step)

        if bool((sample_res < tol).all()):
            break

    converged = residual < tol
    info = {
        "steps": torch.tensor(steps_run, device=device),
        "residual": residual,
        "converged": converged,
        "sample_residual": sample_res,
        "stepsize": step,
    }
    return z, info


def estimate_spectral_radius(
    f: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    *,
    n_iter: int = 50,
    seed: Optional[int] = None,
) -> float:
    """Estimate the spectral radius rho(J_f(z)) by power iteration with JVPs.

    rho(J) < 1 at the fixed point is the sharp condition for (i) local
    convergence of the forward iteration and (ii) convergence of the Neumann
    adjoint series. The 2-norm ||J||_2 (see estimate_lipschitz) upper-bounds
    rho but can exceed 1 for non-normal Jacobians even when the iteration
    converges.
    """
    if seed is not None:
        torch.manual_seed(seed)

    v = torch.randn_like(z)
    v = v / v.norm()
    growth = 0.0
    for _ in range(n_iter):
        _, jv = torch.func.jvp(f, (z.detach(),), (v,))
        growth = jv.norm().item()
        if growth == 0.0:
            return 0.0
        v = jv / growth
    return growth

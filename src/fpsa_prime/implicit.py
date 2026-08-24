"""Implicit differentiation for the single FPSA-Prime attention equilibrium."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .solvers import (
    SolverInfo,
    anderson_solve,
    gmres_solve,
    neumann_solve,
    picard_solve,
    token_fixed_point_residual,
)

BACKWARD_STATS = {
    "iters": 0,
    "relative_residual": 0.0,
    "calls": 0,
    "fallbacks": 0,
    "converged": True,
}


class _ImplicitFixedPoint(torch.autograd.Function):
    """Return the numerical fixed point and route its gradient through Phi."""

    @staticmethod
    def forward(
        ctx,
        fixed_value: torch.Tensor,
        differentiable_map_value: torch.Tensor,
        fixed_map,
        backward_solver: str,
        max_iter: int,
        tol: float,
        restart: int,
        require_convergence: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(fixed_value)
        ctx.fixed_map = fixed_map
        ctx.backward_solver = backward_solver
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.restart = restart
        ctx.require_convergence = require_convergence
        # Numerically return R*, not Phi(R*).  The latter differs by the solver
        # tolerance and should not silently perturb evaluation predictions.
        return fixed_value

    @staticmethod
    def backward(ctx, incoming: torch.Tensor):
        (fixed_value,) = ctx.saved_tensors
        with torch.enable_grad():
            state = fixed_value.detach().requires_grad_(True)
            mapped = ctx.fixed_map(state)

        def vjp(vector: torch.Tensor) -> torch.Tensor:
            return torch.autograd.grad(
                mapped,
                state,
                vector,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]

        if ctx.backward_solver == "gmres":
            adjoint, n_iters, rel = gmres_solve(
                vjp,
                incoming,
                max_iter=ctx.max_iter,
                tol=ctx.tol,
                restart=ctx.restart,
            )
        elif ctx.backward_solver == "neumann":
            adjoint, n_iters, rel = neumann_solve(
                vjp, incoming, max_iter=ctx.max_iter, tol=ctx.tol
            )
        else:
            raise ValueError(f"unknown backward solver {ctx.backward_solver!r}")

        BACKWARD_STATS["iters"] = n_iters
        BACKWARD_STATS["relative_residual"] = rel
        BACKWARD_STATS["calls"] += 1
        finite = bool(torch.isfinite(adjoint).all())
        converged = finite and rel <= ctx.tol
        BACKWARD_STATS["converged"] = converged
        if not converged and ctx.require_convergence:
            reason = "non-finite adjoint" if not finite else (
                f"adjoint relative residual {rel:.3e} exceeds tolerance {ctx.tol:.3e}"
            )
            raise RuntimeError(f"implicit backward did not converge: {reason}")
        if not finite:
            BACKWARD_STATS["fallbacks"] += 1
            adjoint = incoming

        # fixed_value is detached.  Returning the adjoint for the second input
        # sends it through the one differentiable evaluation Phi(R*), producing
        # both parameter gradients and the anchor/input implicit gradient.
        return None, adjoint, None, None, None, None, None, None


def _unrolled_solve(
    fixed_map: Callable[[torch.Tensor], torch.Tensor],
    initial: torch.Tensor,
    *,
    max_iter: int,
    damping: float,
    tol: float,
    record_trace: bool,
) -> tuple[torch.Tensor, SolverInfo]:
    state = initial
    trace = []
    for _ in range(max_iter):
        mapped = fixed_map(state)
        if record_trace:
            with torch.no_grad():
                trace.append(float(token_fixed_point_residual(mapped, state).mean()))
        state = state + damping * (mapped - state)
    with torch.no_grad():
        mapped = fixed_map(state)
        token_res = token_fixed_point_residual(mapped, state)
        sample_res = token_res.amax(dim=1)
    info = SolverInfo(
        n_iters=max_iter,
        n_function_evals=max_iter + 1,
        rel_residual=float(sample_res.max()),
        residual_trace=trace,
        per_sample_iters=torch.full(
            (state.shape[0],), max_iter, dtype=torch.int32, device=state.device
        ),
        per_token_residual=token_res,
        converged=sample_res < tol,
        converged_frac=float((sample_res < tol).float().mean()),
        final_damping=torch.full(
            (state.shape[0],), damping, dtype=state.dtype, device=state.device
        ),
    )
    return state, info


def solve_equilibrium(
    fixed_map: Callable[[torch.Tensor], torch.Tensor],
    initial: torch.Tensor,
    cfg,
    *,
    training: bool,
    max_iter: Optional[int] = None,
    record_trace: bool = False,
    require_convergence: Optional[bool] = None,
) -> tuple[torch.Tensor, SolverInfo]:
    budget = (cfg.max_iter if training else cfg.max_iter_eval) if max_iter is None else max_iter
    if budget <= 0:
        raise ValueError("max_iter must be positive")
    mode = cfg.grad_mode if training else "eval"

    if mode == "bptt":
        return _unrolled_solve(
            fixed_map,
            initial,
            max_iter=budget,
            damping=cfg.solver_damping,
            tol=cfg.fp_tol,
            record_trace=record_trace,
        )

    with torch.no_grad():
        if cfg.forward_solver == "picard":
            fixed, info = picard_solve(
                fixed_map,
                initial,
                max_iter=budget,
                tol=cfg.fp_tol,
                damping=cfg.solver_damping,
                min_damping=cfg.min_damping,
                damping_decay=cfg.damping_decay,
                stall_patience=cfg.stall_patience,
                record_trace=record_trace,
            )
        elif cfg.forward_solver == "anderson":
            fixed, info = anderson_solve(
                fixed_map,
                initial,
                max_iter=budget,
                tol=cfg.fp_tol,
                damping=cfg.solver_damping,
                history=cfg.anderson_m,
                beta=cfg.anderson_beta,
                lam_reg=cfg.anderson_lam,
                record_trace=record_trace,
            )
        else:
            raise ValueError(f"unknown forward solver {cfg.forward_solver!r}")
    fixed = fixed.detach()

    strict_forward = (
        cfg.require_convergence
        if require_convergence is None
        else require_convergence
    )
    if strict_forward and info.converged_frac < 1.0:
        raise RuntimeError(
            f"fixed-point solve did not converge: residual={info.rel_residual:.3e}, "
            f"converged_frac={info.converged_frac:.3f}"
        )
    if not (training and torch.is_grad_enabled()):
        return fixed, info

    mapped = fixed_map(fixed)
    if mode == "one_step":
        # Keep the numerical forward value equal to the solver output while
        # using one application of Phi as the phantom-gradient path.
        return fixed + (mapped - mapped.detach()), info
    if mode != "implicit":
        raise ValueError(f"unknown gradient mode {mode!r}")

    output = _ImplicitFixedPoint.apply(
        fixed,
        mapped,
        fixed_map,
        cfg.backward_solver,
        cfg.backward_max_iter,
        cfg.backward_tol,
        cfg.gmres_restart,
        cfg.require_backward_convergence,
    )
    return output, info

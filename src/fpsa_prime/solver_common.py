"""Shared data structures and residual helpers for FPSA-Prime solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch


@dataclass
class SolverInfo:
    n_iters: int = 0
    n_function_evals: int = 0
    rel_residual: float = float("inf")
    residual_trace: List[float] = field(default_factory=list)
    per_sample_iters: Optional[torch.Tensor] = None
    per_token_residual: Optional[torch.Tensor] = None
    converged: Optional[torch.Tensor] = None
    converged_frac: float = 0.0
    final_damping: Optional[torch.Tensor] = None


def token_fixed_point_residual(phi: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Scale-invariant fixed-point residual of shape ``(B, N)``.

    The symmetric denominator remains well behaved at the zero initialization.
    By the triangle inequality, the value lies in ``[0, 1]`` up to rounding.
    """
    if phi.shape != state.shape or phi.ndim != 3:
        raise ValueError("phi and state must have the same shape (B, N, D)")
    numerator = (phi - state).float().norm(dim=-1)
    denominator = phi.float().norm(dim=-1) + state.float().norm(dim=-1) + 1e-8
    return numerator / denominator


def _check_fixed_map_output(phi: torch.Tensor, state: torch.Tensor) -> None:
    if phi.shape != state.shape:
        raise ValueError("the fixed-point map changed the state shape")
    if not bool(torch.isfinite(phi).all()):
        raise FloatingPointError("the fixed-point map produced NaN or infinity")


def _finalize_solution(
    fixed_map: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    best_state: torch.Tensor,
    best_residual: torch.Tensor,
    info: SolverInfo,
    *,
    tol: float,
    iterations: torch.Tensor,
    damping: torch.Tensor,
) -> tuple[torch.Tensor, SolverInfo]:
    """Return the lowest-residual state seen independently for each sample."""
    phi = fixed_map(state)
    info.n_function_evals += 1
    _check_fixed_map_output(phi, state)
    current_token = token_fixed_point_residual(phi, state)
    current_sample = current_token.amax(dim=1)
    current_is_better = current_sample < best_residual
    selected = torch.where(
        current_is_better.view(-1, 1, 1), state, best_state
    )

    # Re-evaluate the selected state so diagnostics always describe the exact
    # tensor returned to the caller, including mixed per-sample best iterates.
    selected_phi = fixed_map(selected)
    info.n_function_evals += 1
    _check_fixed_map_output(selected_phi, selected)
    token_residual = token_fixed_point_residual(selected_phi, selected)
    sample_residual = token_residual.amax(dim=1)
    converged = sample_residual < tol
    info.rel_residual = float(sample_residual.max())
    info.per_sample_iters = iterations
    info.per_token_residual = token_residual
    info.converged = converged
    info.converged_frac = float(converged.float().mean())
    info.final_damping = damping.reshape(-1)
    return selected, info

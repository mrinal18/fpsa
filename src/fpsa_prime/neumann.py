"""Neumann adjoint solver for FPSA-Prime implicit differentiation."""

from __future__ import annotations

from typing import Callable

import torch


def neumann_solve(
    vjp: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, int, float]:
    """Truncated fixed-point/Neumann solve for the adjoint equation."""
    if max_iter <= 0 or tol <= 0:
        raise ValueError("Neumann iteration count and tolerance must be positive")
    estimate = rhs.clone()
    relative = float("inf")
    iteration = 0
    for iteration in range(1, max_iter + 1):
        updated = rhs + vjp(estimate)
        if not torch.isfinite(updated).all():
            return rhs, iteration, float("inf")
        relative = float(
            (updated - estimate).norm() / estimate.norm().clamp_min(1e-12)
        )
        estimate = updated
        if relative < tol:
            break
    return estimate, iteration, relative

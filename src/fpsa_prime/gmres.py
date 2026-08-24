"""Per-sample restarted GMRES for FPSA-Prime implicit differentiation."""

from __future__ import annotations

from typing import Callable

import torch


def gmres_solve(
    vjp: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iter: int,
    tol: float,
    restart: int = 20,
) -> tuple[torch.Tensor, int, float]:
    """Solve ``(I - J^T) lambda = rhs`` with per-sample restarted GMRES.

    The Krylov basis and least-squares coefficients are separate for every batch
    item. A truncated backward solve therefore cannot make one example's
    gradient depend on which unrelated examples happened to share its batch.
    One batched VJP advances all per-sample Krylov systems simultaneously.
    """
    if max_iter <= 0 or tol <= 0 or restart <= 0:
        raise ValueError("GMRES iteration counts and tolerance must be positive")
    if rhs.ndim != 3:
        raise ValueError("FPSA-Prime GMRES expects rhs shape (B, N, D)")

    batch = rhs.shape[0]
    state_shape = rhs.shape[1:]
    dimension = rhs[0].numel()
    original_dtype = rhs.dtype
    work_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    vector_rhs = rhs.reshape(batch, dimension).to(work_dtype)

    def operator(vectors: torch.Tensor) -> torch.Tensor:
        shaped = vectors.to(original_dtype).reshape(batch, *state_shape)
        output = shaped - vjp(shaped)
        return output.reshape(batch, dimension).to(work_dtype)

    estimate = torch.zeros_like(vector_rhs)
    rhs_norm = vector_rhs.norm(dim=-1)
    zero_rhs = rhs_norm < 1e-12
    done = zero_rhs.clone()
    relative = torch.where(
        zero_rhs, torch.zeros_like(rhs_norm), torch.ones_like(rhs_norm)
    )
    total = 0

    while total < max_iter and not bool(done.all()):
        residual = vector_rhs - operator(estimate)
        beta = residual.norm(dim=-1)
        relative = torch.where(
            zero_rhs,
            torch.zeros_like(beta),
            beta / rhs_norm.clamp_min(1e-12),
        )
        done |= relative < tol
        active = ~done
        if not bool(active.any()):
            break

        width = min(restart, max_iter - total)
        basis = torch.zeros(
            batch,
            width + 1,
            dimension,
            dtype=vector_rhs.dtype,
            device=vector_rhs.device,
        )
        hessenberg = torch.zeros(
            batch,
            width + 1,
            width,
            dtype=vector_rhs.dtype,
            device=vector_rhs.device,
        )
        basis[:, 0] = torch.where(
            active[:, None], residual / beta.clamp_min(1e-12)[:, None], 0.0
        )
        used = 0

        for column in range(width):
            candidate = operator(basis[:, column])
            total += 1
            for row in range(column + 1):
                projection = (basis[:, row] * candidate).sum(dim=-1)
                hessenberg[:, row, column] = projection
                candidate = candidate - projection[:, None] * basis[:, row]
            norm = candidate.norm(dim=-1)
            hessenberg[:, column + 1, column] = norm
            used = column + 1
            can_extend = active & (norm >= 1e-12)
            basis[:, column + 1] = torch.where(
                can_extend[:, None], candidate / norm.clamp_min(1e-12)[:, None], 0.0
            )
            if not bool(can_extend.any()):
                break

        target = torch.zeros(
            batch,
            used + 1,
            1,
            dtype=vector_rhs.dtype,
            device=vector_rhs.device,
        )
        target[:, 0, 0] = beta
        least_squares = torch.linalg.lstsq(
            hessenberg[:, : used + 1, :used], target
        ).solution.squeeze(-1)
        update = (
            basis[:, :used] * least_squares.unsqueeze(-1)
        ).sum(dim=1)
        estimate = torch.where(active[:, None], estimate + update, estimate)

        residual = vector_rhs - operator(estimate)
        residual_norm = residual.norm(dim=-1)
        relative = torch.where(
            zero_rhs,
            torch.zeros_like(residual_norm),
            residual_norm / rhs_norm.clamp_min(1e-12),
        )
        done |= relative < tol

    output = estimate.to(original_dtype).reshape_as(rhs)
    if not bool(torch.isfinite(output).all()):
        return rhs, total, float("inf")
    return output, total, float(relative.max())

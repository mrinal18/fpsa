"""Numerically robust per-sample restarted GMRES for FPSA-Prime.

The adjoint system is

    (I - J_Phi(R*)^T) lambda = dL / dR*.

Each item in a training batch is an independent linear system.  The Krylov
basis, Hessenberg matrix, Givens rotations, stopping decision, and restart state
are therefore maintained per sample even though a single batched VJP advances
all still-active systems.
"""

from __future__ import annotations

from typing import Callable

import torch


def _batched_dot(left: torch.Tensor, right: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Stable per-row dot product without materialising a full double tensor."""
    return torch.sum(left * right, dim=-1, dtype=dtype)


def _batched_norm(value: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.sqrt(
        torch.sum(value * value, dim=-1, dtype=dtype).clamp_min(0.0)
    )


def _apply_cycle_update(
    estimate: torch.Tensor,
    basis: torch.Tensor,
    hessenberg: torch.Tensor,
    residual_rhs: torch.Tensor,
    mask: torch.Tensor,
    width: int,
) -> None:
    """Apply the current GMRES correction to selected batch items in place."""
    if width <= 0 or not bool(mask.any()):
        return
    h = hessenberg[mask, :width, :width]
    g = residual_rhs[mask, :width].unsqueeze(-1)
    diagonal = h.diagonal(dim1=-2, dim2=-1).abs()
    well_conditioned = torch.isfinite(h).all(dim=(-1, -2)) & (
        diagonal.amin(dim=-1) > 1e-12
    )

    coefficients = torch.zeros(
        h.shape[0], width, dtype=h.dtype, device=h.device
    )
    if bool(well_conditioned.any()):
        coefficients[well_conditioned] = torch.linalg.solve_triangular(
            h[well_conditioned], g[well_conditioned], upper=True
        ).squeeze(-1)
    if bool((~well_conditioned).any()):
        # Happy breakdown and finite-precision loss of rank are both legitimate
        # reasons for a tiny Hessenberg diagonal.  A small least-squares solve is
        # safer than dividing by that diagonal.
        bad_h = h[~well_conditioned]
        bad_g = g[~well_conditioned]
        solution = torch.linalg.lstsq(bad_h, bad_g).solution.squeeze(-1)
        coefficients[~well_conditioned] = solution

    update = (
        basis[mask, :width]
        * coefficients.to(basis.dtype).unsqueeze(-1)
    ).sum(dim=1)
    estimate[mask] = estimate[mask] + update


def gmres_solve(
    vjp: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iter: int,
    tol: float,
    restart: int = 20,
) -> tuple[torch.Tensor, int, float]:
    """Solve ``(I - J^T) lambda = rhs`` with restarted batched GMRES.

    This implementation differs from the original v0 solver in four important
    ways:

    * Givens rotations update the least-squares residual after every Arnoldi
      column, so a solution that reaches tolerance stops immediately instead of
      being overwritten by a longer, less stable basis.
    * Modified Gram-Schmidt is run twice (DGKS-style reorthogonalisation).
    * Convergence and happy breakdown are tracked independently per sample.
    * A true residual is recomputed at every restart before accepting a sample.

    One batched VJP advances all active systems, so the cost remains one VJP per
    global Arnoldi column rather than one VJP per sample.
    """
    if max_iter <= 0 or tol <= 0 or restart <= 0:
        raise ValueError("GMRES iteration counts and tolerance must be positive")
    if rhs.ndim != 3:
        raise ValueError("FPSA-Prime GMRES expects rhs shape (B, N, D)")
    if not bool(torch.isfinite(rhs).all()):
        return rhs, 0, float("inf")

    batch = rhs.shape[0]
    state_shape = rhs.shape[1:]
    dimension = rhs[0].numel()
    original_dtype = rhs.dtype
    vector_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    # The basis stays in the operator's practical dtype.  Scalar reductions and
    # the tiny Hessenberg systems use float64 when possible, which is where the
    # old implementation lost most of its accuracy.
    scalar_dtype = torch.float64 if vector_dtype != torch.float64 else torch.float64

    vector_rhs = rhs.reshape(batch, dimension).to(vector_dtype)

    def operator(vectors: torch.Tensor) -> torch.Tensor:
        shaped = vectors.to(original_dtype).reshape(batch, *state_shape)
        output = shaped - vjp(shaped)
        flat = output.reshape(batch, dimension).to(vector_dtype)
        if flat.shape != vectors.shape:
            raise RuntimeError("GMRES operator returned the wrong shape")
        return flat

    estimate = torch.zeros_like(vector_rhs)
    rhs_norm = _batched_norm(vector_rhs, dtype=scalar_dtype)
    zero_rhs = rhs_norm <= 1e-30
    done = zero_rhs.clone()
    relative = torch.where(
        zero_rhs, torch.zeros_like(rhs_norm), torch.full_like(rhs_norm, float("inf"))
    )
    total_iterations = 0

    while total_iterations < max_iter and not bool(done.all()):
        residual = vector_rhs - operator(estimate)
        residual_norm = _batched_norm(residual, dtype=scalar_dtype)
        relative = torch.where(
            zero_rhs,
            torch.zeros_like(residual_norm),
            residual_norm / rhs_norm.clamp_min(1e-30),
        )
        done |= relative <= tol
        cycle_active = ~done
        if not bool(cycle_active.any()):
            break

        width = min(restart, max_iter - total_iterations)
        basis = torch.zeros(
            batch,
            width + 1,
            dimension,
            dtype=vector_dtype,
            device=rhs.device,
        )
        hessenberg = torch.zeros(
            batch,
            width + 1,
            width,
            dtype=scalar_dtype,
            device=rhs.device,
        )
        cosines = torch.zeros(
            batch, width, dtype=scalar_dtype, device=rhs.device
        )
        sines = torch.zeros_like(cosines)
        residual_rhs = torch.zeros(
            batch, width + 1, dtype=scalar_dtype, device=rhs.device
        )
        residual_rhs[:, 0] = residual_norm
        basis[:, 0] = torch.where(
            cycle_active[:, None],
            residual / residual_norm.clamp_min(1e-30).to(vector_dtype)[:, None],
            torch.zeros_like(residual),
        )

        # A sample leaves cycle_active once its incremental residual estimate is
        # below tolerance or Arnoldi reaches happy breakdown.  Its correction is
        # applied immediately and verified with a true residual after the cycle.
        finished_in_cycle = torch.zeros_like(done)
        used_width = torch.zeros(batch, dtype=torch.int64, device=rhs.device)

        for column in range(width):
            active = cycle_active & ~finished_in_cycle
            if not bool(active.any()):
                break

            candidate = operator(basis[:, column])
            total_iterations += 1

            # Two-pass modified Gram-Schmidt.  Projections for inactive samples
            # are masked to zero so they do not accumulate numerical garbage.
            for _ in range(2):
                for row in range(column + 1):
                    projection = _batched_dot(
                        basis[:, row], candidate, dtype=scalar_dtype
                    )
                    projection = torch.where(
                        active, projection, torch.zeros_like(projection)
                    )
                    hessenberg[:, row, column] += projection
                    candidate = candidate - projection.to(vector_dtype)[:, None] * basis[:, row]

            next_norm = _batched_norm(candidate, dtype=scalar_dtype)
            hessenberg[:, column + 1, column] = torch.where(
                active, next_norm, torch.zeros_like(next_norm)
            )
            breakdown_scale = hessenberg[:, : column + 1, column].abs().amax(dim=1)
            happy_breakdown = active & (
                next_norm <= 1e-12 * breakdown_scale.clamp_min(1.0)
            )
            extend = active & ~happy_breakdown
            basis[:, column + 1] = torch.where(
                extend[:, None],
                candidate
                / next_norm.clamp_min(1e-30).to(vector_dtype)[:, None],
                torch.zeros_like(candidate),
            )

            # Apply all previous Givens rotations to the new Hessenberg column.
            for row in range(column):
                upper = hessenberg[:, row, column].clone()
                lower = hessenberg[:, row + 1, column].clone()
                c = cosines[:, row]
                s = sines[:, row]
                hessenberg[:, row, column] = c * upper + s * lower
                hessenberg[:, row + 1, column] = -s * upper + c * lower

            diagonal = hessenberg[:, column, column].clone()
            subdiagonal = hessenberg[:, column + 1, column].clone()
            radius = torch.hypot(diagonal, subdiagonal)
            safe_radius = radius.clamp_min(1e-30)
            cosine = diagonal / safe_radius
            sine = subdiagonal / safe_radius
            cosine = torch.where(active, cosine, torch.ones_like(cosine))
            sine = torch.where(active, sine, torch.zeros_like(sine))
            cosines[:, column] = cosine
            sines[:, column] = sine
            hessenberg[:, column, column] = cosine * diagonal + sine * subdiagonal
            hessenberg[:, column + 1, column] = 0.0

            old_g = residual_rhs[:, column].clone()
            next_g = residual_rhs[:, column + 1].clone()
            residual_rhs[:, column] = cosine * old_g + sine * next_g
            residual_rhs[:, column + 1] = -sine * old_g + cosine * next_g
            estimated_relative = residual_rhs[:, column + 1].abs() / rhs_norm.clamp_min(1e-30)

            used_width = torch.where(
                active,
                torch.full_like(used_width, column + 1),
                used_width,
            )
            newly_finished = active & (
                (estimated_relative <= tol) | happy_breakdown
            )
            if bool(newly_finished.any()):
                _apply_cycle_update(
                    estimate,
                    basis,
                    hessenberg,
                    residual_rhs,
                    newly_finished,
                    column + 1,
                )
                finished_in_cycle |= newly_finished

            if total_iterations >= max_iter:
                break

        # Apply the correction for samples that consumed the available restart
        # width without an early stop.  Every such sample has the same final
        # global column, but used_width is retained for defensive correctness.
        unfinished = cycle_active & ~finished_in_cycle
        if bool(unfinished.any()):
            unique_widths = torch.unique(used_width[unfinished])
            for used in unique_widths.tolist():
                if used <= 0:
                    continue
                subset = unfinished & (used_width == used)
                _apply_cycle_update(
                    estimate,
                    basis,
                    hessenberg,
                    residual_rhs,
                    subset,
                    int(used),
                )

        # Never trust only the recursively updated least-squares estimate in
        # float32.  Verify the actual linear-system residual before accepting a
        # sample and before starting a new restart cycle.
        residual = vector_rhs - operator(estimate)
        residual_norm = _batched_norm(residual, dtype=scalar_dtype)
        relative = torch.where(
            zero_rhs,
            torch.zeros_like(residual_norm),
            residual_norm / rhs_norm.clamp_min(1e-30),
        )
        finite = torch.isfinite(relative) & torch.isfinite(estimate).all(dim=1)
        done |= finite & (relative <= tol)
        if not bool(finite.all()):
            break

    output = estimate.to(original_dtype).reshape_as(rhs)
    if not bool(torch.isfinite(output).all()):
        return rhs, total_iterations, float("inf")
    return output, total_iterations, float(relative.max())

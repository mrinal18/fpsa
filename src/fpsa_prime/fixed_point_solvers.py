"""Forward fixed-point solvers for FPSA-Prime."""

from __future__ import annotations

from typing import Callable

import torch

from .solver_common import (
    SolverInfo,
    _check_fixed_map_output,
    _finalize_solution,
    token_fixed_point_residual,
)


@torch.no_grad()
def picard_solve(
    fixed_map: Callable[[torch.Tensor], torch.Tensor],
    initial: torch.Tensor,
    *,
    max_iter: int,
    tol: float,
    damping: float,
    min_damping: float,
    damping_decay: float,
    stall_patience: int,
    record_trace: bool = False,
) -> tuple[torch.Tensor, SolverInfo]:
    """Damped Picard iteration with per-sample adaptive damping."""
    if initial.ndim != 3:
        raise ValueError("FPSA-Prime solvers expect state shape (B, N, D)")
    if max_iter <= 0 or tol <= 0 or stall_patience <= 0:
        raise ValueError("max_iter, tol, and stall_patience must be positive")
    if not 0 < min_damping <= damping <= 1 or not 0 < damping_decay <= 1:
        raise ValueError("invalid damping configuration")

    state = initial.clone()
    batch = state.shape[0]
    alpha = torch.full(
        (batch, 1, 1), damping, device=state.device, dtype=state.dtype
    )
    best_residual = torch.full((batch,), float("inf"), device=state.device)
    best_state = state.clone()
    stalls = torch.zeros(batch, dtype=torch.int64, device=state.device)
    done = torch.zeros(batch, dtype=torch.bool, device=state.device)
    iterations = torch.zeros(batch, dtype=torch.int32, device=state.device)
    info = SolverInfo()

    for step in range(max_iter):
        running = ~done
        phi = fixed_map(state)
        info.n_function_evals += 1
        _check_fixed_map_output(phi, state)
        token_residual = token_fixed_point_residual(phi, state)
        sample_residual = token_residual.amax(dim=1)
        newly_done = running & (sample_residual < tol)
        active = running & ~newly_done
        iterations += running.to(torch.int32)

        better = running & (sample_residual < best_residual)
        best_state = torch.where(better.view(batch, 1, 1), state, best_state)
        improved = active & (sample_residual < best_residual * (1.0 - 1e-3))
        best_residual = torch.where(
            better, sample_residual, best_residual
        )
        stalls = torch.where(improved, torch.zeros_like(stalls), stalls)
        stalls = torch.where(active & ~improved, stalls + 1, stalls)
        decay = active & (stalls >= stall_patience)
        if bool(decay.any()):
            alpha = torch.where(
                decay.view(batch, 1, 1),
                torch.clamp(alpha * damping_decay, min=min_damping),
                alpha,
            )
            stalls = torch.where(decay, torch.zeros_like(stalls), stalls)

        candidate = state + alpha * (phi - state)
        state = torch.where(active.view(batch, 1, 1), candidate, state)
        done |= newly_done

        info.n_iters = step + 1
        info.rel_residual = float(sample_residual.max())
        if record_trace:
            info.residual_trace.append(float(sample_residual.mean()))
        if bool(done.all()):
            break

    state, info = _finalize_solution(
        fixed_map,
        state,
        best_state,
        best_residual,
        info,
        tol=tol,
        iterations=iterations,
        damping=alpha,
    )
    return state, info


@torch.no_grad()
def anderson_solve(
    fixed_map: Callable[[torch.Tensor], torch.Tensor],
    initial: torch.Tensor,
    *,
    max_iter: int,
    tol: float,
    damping: float,
    history: int,
    beta: float,
    lam_reg: float,
    record_trace: bool = False,
) -> tuple[torch.Tensor, SolverInfo]:
    """Per-sample Anderson acceleration.

    Each batch item gets its own least-squares coefficients. Flattening the
    entire batch into one Anderson problem would couple independent examples and
    make convergence depend on which other samples happened to share the batch.
    """
    if initial.ndim != 3:
        raise ValueError("FPSA-Prime solvers expect state shape (B, N, D)")
    if max_iter <= 0 or tol <= 0 or history <= 0:
        raise ValueError("max_iter, tol, and history must be positive")
    if not 0 < damping <= 1 or not 0 <= beta <= 1 or lam_reg < 0:
        raise ValueError("invalid Anderson configuration")

    batch = initial.shape[0]
    state_shape = initial.shape[1:]
    dim = initial[0].numel()
    memory = min(history, max_iter)
    x = initial.reshape(batch, dim).clone()
    x_history = torch.zeros(
        batch, memory, dim, dtype=initial.dtype, device=initial.device
    )
    f_history = torch.zeros_like(x_history)
    done = torch.zeros(batch, dtype=torch.bool, device=initial.device)
    iterations = torch.zeros(batch, dtype=torch.int32, device=initial.device)
    best_residual = torch.full((batch,), float("inf"), device=initial.device)
    best_state = initial.clone()
    info = SolverInfo()

    for step in range(max_iter):
        running = ~done
        state = x.reshape(batch, *state_shape)
        phi_state = fixed_map(state)
        info.n_function_evals += 1
        _check_fixed_map_output(phi_state, state)
        phi = phi_state.reshape(batch, dim)
        token_residual = token_fixed_point_residual(phi_state, state)
        sample_residual = token_residual.amax(dim=1)
        better = running & (sample_residual < best_residual)
        best_state = torch.where(
            better.view(batch, 1, 1), state, best_state
        )
        best_residual = torch.where(
            better, sample_residual, best_residual
        )
        newly_done = running & (sample_residual < tol)
        active = running & ~newly_done
        iterations += running.to(torch.int32)

        slot = step % memory
        x_history[:, slot] = x
        f_history[:, slot] = phi
        n_history = min(step + 1, memory)

        if n_history == 1:
            mixed = phi
        else:
            # Once the ring buffer wraps, its rows are no longer chronological,
            # but the constrained least-squares solution is permutation-invariant.
            x_hist = x_history[:, :n_history]
            f_hist = f_history[:, :n_history]
            residual_hist = (f_hist - x_hist).float()
            gram = residual_hist @ residual_hist.transpose(1, 2)
            scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).abs() + 1e-8
            eye = torch.eye(
                n_history, dtype=gram.dtype, device=gram.device
            ).unsqueeze(0)
            system = gram + lam_reg * scale.view(batch, 1, 1) * eye
            ones = torch.ones(
                batch, n_history, 1, dtype=gram.dtype, device=gram.device
            )
            coefficients, solve_info = torch.linalg.solve_ex(system, ones)
            denominator = coefficients.sum(dim=1, keepdim=True)
            failed = (
                (solve_info != 0)
                | ~torch.isfinite(coefficients).all(dim=(1, 2))
                | (denominator.squeeze(-1).squeeze(-1).abs() < 1e-12)
            )
            if bool(failed.any()):
                coefficients[failed] = 1.0
                denominator[failed] = float(n_history)
            coefficients = coefficients / denominator
            weights = coefficients.squeeze(-1).unsqueeze(-1).to(f_hist.dtype)
            mixed_f = (weights * f_hist).sum(dim=1)
            mixed_x = (weights * x_hist).sum(dim=1)
            mixed = beta * mixed_f + (1.0 - beta) * mixed_x

        candidate = x + damping * (mixed - x)
        picard_candidate = x + damping * (phi - x)
        finite_candidate = torch.isfinite(candidate).all(dim=1)
        candidate = torch.where(
            finite_candidate.view(batch, 1), candidate, picard_candidate
        )
        x = torch.where(active.view(batch, 1), candidate, x)
        done |= newly_done

        info.n_iters = step + 1
        info.rel_residual = float(sample_residual.max())
        if record_trace:
            info.residual_trace.append(float(sample_residual.mean()))
        if bool(done.all()):
            break

    state = x.reshape(batch, *state_shape)
    final_damping = torch.full(
        (batch,), damping, dtype=initial.dtype, device=initial.device
    )
    state, info = _finalize_solution(
        fixed_map,
        state,
        best_state,
        best_residual,
        info,
        tol=tol,
        iterations=iterations,
        damping=final_damping,
    )
    return state, info

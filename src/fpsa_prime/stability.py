"""Local stability diagnostics and soft regularisation for FPSA-Prime."""

from __future__ import annotations

from typing import Callable

import torch


def _normalize_per_sample(vector: torch.Tensor) -> torch.Tensor:
    norm = vector.flatten(1).norm(dim=1).clamp_min(1e-12)
    return vector / norm.view(-1, 1, 1)


def _deterministic_probe_like(state: torch.Tensor) -> torch.Tensor:
    dimension = state[0].numel()
    index = torch.arange(dimension, device=state.device, dtype=torch.float32)
    probe = torch.sin(index * 0.017) + torch.cos(index * 0.031)
    probe = probe.to(state.dtype).reshape(1, *state.shape[1:])
    if state.shape[0] > 1:
        offsets = torch.arange(
            state.shape[0], device=state.device, dtype=state.dtype
        ).view(-1, 1, 1)
        probe = probe.expand_as(state) + 0.01 * torch.sin(offsets + probe)
    else:
        probe = probe.expand_as(state)
    return _normalize_per_sample(probe)


def _finite_difference_jvp(
    fixed_map: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    vector: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    coordinates = state[0].numel()
    rms = state.flatten(1).norm(dim=1) / (coordinates**0.5)
    step = epsilon * rms.clamp_min(1.0)
    scaled = step.view(-1, 1, 1)
    positive = fixed_map(state + scaled * vector)
    negative = fixed_map(state - scaled * vector)
    return (positive - negative) / (2.0 * scaled)


def local_jacobian_spectral_penalty(
    fixed_map: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    *,
    target: float,
    power_steps: int = 1,
    epsilon: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Approximate ``sigma_max(dPhi/dR)`` and return a one-sided penalty.

    The right singular-vector probe is estimated with power iteration on
    ``J^T J``. Probe updates are detached, while the final finite-difference
    JVP remains differentiable with respect to the recurrent attention
    parameters. This avoids a full second-order graph while targeting local
    feedback gain rather than globally capping Q/K/O projection norms.
    """
    if state.ndim != 3:
        raise ValueError("stability state must have shape (B, N, D)")
    if target <= 0 or power_steps <= 0 or epsilon <= 0:
        raise ValueError("target, power_steps, and epsilon must be positive")

    point = state.detach()
    vector = _deterministic_probe_like(point)

    for _ in range(power_steps):
        with torch.no_grad():
            jv = _finite_difference_jvp(
                fixed_map, point, vector, epsilon
            ).detach()
        with torch.enable_grad():
            variable = point.detach().requires_grad_(True)
            mapped = fixed_map(variable)
            jtjv = torch.autograd.grad(
                mapped,
                variable,
                jv,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
        vector = _normalize_per_sample(jtjv.detach())

    # Keep this final pair of map evaluations in the graph so the hinge penalty
    # updates the recurrent operator. The equilibrium state and probe direction
    # are treated as constants for this local regulariser.
    final_jv = _finite_difference_jvp(fixed_map, point, vector, epsilon)
    estimate = final_jv.flatten(1).norm(dim=1)
    penalty_per_sample = torch.relu(estimate - target).square()
    return {
        "loss": penalty_per_sample.mean(),
        "estimate": estimate,
        "max_estimate": estimate.max(),
    }

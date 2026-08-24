"""Mathematical and architectural invariants for FPSA-Prime."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fpsa_prime import build_model  # noqa: E402
from src.fpsa_prime.attention import DualBankFixedPointAttention  # noqa: E402
from src.fpsa_prime.config import FPSAPrimeConfig  # noqa: E402
from src.fpsa_prime.implicit import solve_equilibrium  # noqa: E402
from src.fpsa_prime.losses import (  # noqa: E402
    attractor_margin_loss,
    sudoku_discrete_violations,
    sudoku_energy,
)
from src.fpsa_prime.solvers import anderson_solve, gmres_solve, picard_solve  # noqa: E402
from src.fpsa_prime.structures import (  # noqa: E402
    MAZE_RELATIONS,
    SUDOKU_RELATIONS,
    local_attention_bias,
    maze_relation_ids,
    prepend_global_attention_bias,
    prepend_global_slots,
    sudoku_relation_ids,
)


def _flat_grad(model):
    return torch.cat(
        [
            parameter.grad.reshape(-1)
            for _, parameter in sorted(model.named_parameters())
            if parameter.grad is not None
        ]
    )


def _tiny_kwargs(**extra):
    values = dict(
        vocab_size=10,
        out_vocab_size=10,
        max_seq_len=9,
        hidden_size=24,
        num_heads=4,
        use_input_mlp=False,
        use_output_mlp=False,
        use_verifier_head=False,
        position_encoding="rope",
        forward_solver="picard",
        solver_damping=0.8,
        fp_tol=1e-7,
        max_iter=100,
        max_iter_eval=100,
        backward_max_iter=80,
        backward_tol=1e-7,
        output_init_std=0.01,
        require_convergence=False,
        require_backward_convergence=True,
    )
    values.update(extra)
    return values


def test_solver_damping_does_not_change_the_fixed_point():
    torch.manual_seed(0)
    matrix = torch.randn(8, 8) * 0.03
    bias = torch.randn(2, 4, 2)

    def fixed_map(state):
        return (state.reshape(2, -1) @ matrix.T).reshape_as(state) + bias

    initial = torch.zeros_like(bias)
    full, full_info = picard_solve(
        fixed_map,
        initial,
        max_iter=300,
        tol=1e-8,
        damping=1.0,
        min_damping=1.0,
        damping_decay=1.0,
        stall_patience=10,
    )
    damped, damped_info = picard_solve(
        fixed_map,
        initial,
        max_iter=300,
        tol=1e-8,
        damping=0.25,
        min_damping=0.25,
        damping_decay=1.0,
        stall_patience=10,
    )
    assert full_info.rel_residual < 1e-7
    assert damped_info.rel_residual < 1e-7
    assert torch.allclose(full, damped, atol=2e-6, rtol=2e-6)


def test_anderson_solves_each_batch_item_independently():
    torch.manual_seed(1)
    matrix = torch.randn(6, 6) * 0.04
    bias = torch.randn(3, 2, 3)

    def fixed_map(state):
        return (state.reshape(3, -1) @ matrix.T).reshape_as(state) + bias

    initial = torch.zeros_like(bias)
    together, info = anderson_solve(
        fixed_map,
        initial,
        max_iter=50,
        tol=1e-7,
        damping=1.0,
        history=5,
        beta=1.0,
        lam_reg=1e-4,
    )
    separate = []
    for index in range(3):
        single, _ = anderson_solve(
            lambda state, index=index: (
                state.reshape(1, -1) @ matrix.T
            ).reshape_as(state)
            + bias[index : index + 1],
            initial[index : index + 1],
            max_iter=50,
            tol=1e-7,
            damping=1.0,
            history=5,
            beta=1.0,
            lam_reg=1e-4,
        )
        separate.append(single)
    separate = torch.cat(separate, dim=0)
    assert info.rel_residual < 1e-6
    assert torch.allclose(together, separate, atol=2e-5, rtol=2e-5)


def test_gmres_keeps_truncated_batch_adjoints_independent():
    torch.manual_seed(3)
    jacobians = torch.randn(3, 6, 6) * 0.08
    rhs = torch.randn(3, 2, 3)

    def batched_vjp(vector):
        flat = vector.reshape(3, 6)
        return torch.einsum("bij,bj->bi", jacobians.transpose(1, 2), flat).reshape_as(vector)

    together, _, _ = gmres_solve(
        batched_vjp, rhs, max_iter=3, tol=1e-12, restart=3
    )
    separate = []
    for index in range(3):
        matrix = jacobians[index]

        def single_vjp(vector, matrix=matrix):
            flat = vector.reshape(1, 6)
            return (flat @ matrix).reshape_as(vector)

        solved, _, _ = gmres_solve(
            single_vjp,
            rhs[index : index + 1],
            max_iter=3,
            tol=1e-12,
            restart=3,
        )
        separate.append(solved)
    separate = torch.cat(separate, dim=0)
    assert torch.allclose(together, separate, atol=2e-6, rtol=2e-6)


def test_strict_backward_rejects_an_inexact_adjoint():
    torch.manual_seed(12)
    anchor = torch.randn(1, 1, 2, dtype=torch.float64, requires_grad=True)
    matrix = torch.tensor(
        [[0.2, 0.1], [0.0, 0.3]], dtype=torch.float64
    )
    initial = torch.zeros_like(anchor)
    cfg = SimpleNamespace(
        max_iter=200,
        max_iter_eval=200,
        grad_mode="implicit",
        forward_solver="picard",
        fp_tol=1e-11,
        solver_damping=1.0,
        min_damping=1.0,
        damping_decay=1.0,
        stall_patience=10,
        anderson_m=4,
        anderson_beta=1.0,
        anderson_lam=1e-6,
        require_convergence=True,
        backward_solver="gmres",
        backward_max_iter=1,
        backward_tol=1e-12,
        gmres_restart=1,
        require_backward_convergence=True,
    )
    fixed, _ = solve_equilibrium(
        lambda state: state @ matrix.T + anchor,
        initial,
        cfg,
        training=True,
    )
    try:
        (fixed * torch.tensor([[[1.0, 2.0]]], dtype=fixed.dtype)).sum().backward()
    except RuntimeError as error:
        assert "implicit backward did not converge" in str(error)
    else:
        raise AssertionError("strict backward accepted an inexact adjoint")


def test_attractor_margin_repels_high_energy_fixed_points():
    scale = torch.tensor(0.5, requires_grad=True)
    residual = torch.tensor([[[0.2]], [[0.4]]])
    energy = torch.tensor([0.0, 1.0])
    result = attractor_margin_loss(
        lambda state: scale * state + 0.1,
        residual,
        energy,
        positive_threshold=0.1,
        negative_threshold=0.9,
        negative_margin=0.3,
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert scale.grad is not None and torch.isfinite(scale.grad)
    assert scale.grad.abs() > 0


def test_implicit_gradient_matches_exact_linear_fixed_point():
    torch.manual_seed(2)
    anchor = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
    gain = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    initial = torch.zeros_like(anchor)
    cfg = SimpleNamespace(
        max_iter=200,
        max_iter_eval=200,
        grad_mode="implicit",
        forward_solver="picard",
        fp_tol=1e-11,
        solver_damping=1.0,
        min_damping=1.0,
        damping_decay=1.0,
        stall_patience=10,
        anderson_m=4,
        anderson_beta=1.0,
        anderson_lam=1e-6,
        require_convergence=True,
        backward_solver="gmres",
        backward_max_iter=20,
        backward_tol=1e-11,
        gmres_restart=10,
        require_backward_convergence=True,
    )
    fixed, info = solve_equilibrium(
        lambda state: gain * state + anchor,
        initial,
        cfg,
        training=True,
    )
    loss = fixed.square().sum()
    loss.backward()

    exact = anchor.detach() / (1.0 - gain.detach())
    expected_anchor_grad = 2.0 * exact / (1.0 - gain.detach())
    expected_gain_grad = 2.0 * exact.square().sum() / (1.0 - gain.detach())
    assert info.rel_residual < 1e-10
    assert torch.allclose(anchor.grad, expected_anchor_grad, atol=1e-9, rtol=1e-9)
    assert torch.allclose(gain.grad, expected_gain_grad, atol=1e-9, rtol=1e-9)


def test_gmres_long_restart_matches_direct_solve_and_stops_early():
    """Production-shaped regression for the v0 long-basis failure.

    The old solver reached a good residual at a short Krylov depth, continued to
    the full restart width, and returned a much worse solution.  A larger budget
    must not destroy an already-converged iterate.
    """
    torch.manual_seed(123)
    batch, dimension = 6, 32
    jacobians = torch.zeros(batch, dimension, dimension)
    for index in range(batch):
        jacobians[index].diagonal().fill_(0.10)
        jacobians[index].diagonal(1).fill_(0.25)
        jacobians[index] += 0.003 * torch.randn(dimension, dimension)
    rhs = torch.randn(batch, 4, 8)

    def vjp(vector):
        flat = vector.reshape(batch, dimension)
        return torch.einsum(
            "bij,bj->bi", jacobians.transpose(1, 2), flat
        ).reshape_as(vector)

    short, short_iters, short_rel = gmres_solve(
        vjp, rhs, max_iter=40, tol=1e-6, restart=20
    )
    long, long_iters, long_rel = gmres_solve(
        vjp, rhs, max_iter=200, tol=1e-6, restart=20
    )
    system = (
        torch.eye(dimension).expand(batch, dimension, dimension)
        - jacobians.transpose(1, 2)
    )
    exact = torch.linalg.solve(
        system, rhs.reshape(batch, dimension, 1)
    ).reshape_as(rhs)

    relative_error = (
        (short - exact).reshape(batch, -1).norm(dim=1)
        / exact.reshape(batch, -1).norm(dim=1).clamp_min(1e-12)
    )
    assert short_rel < 1e-6
    assert long_rel < 1e-6
    assert short_iters < 20
    assert long_iters == short_iters
    assert relative_error.max() < 2e-6
    assert torch.allclose(short, long, atol=2e-6, rtol=2e-6)


def test_gmres_long_batched_solve_matches_individual_solves():
    torch.manual_seed(124)
    batch, dimension = 5, 24
    jacobians = torch.randn(batch, dimension, dimension) * 0.025
    jacobians += 0.08 * torch.eye(dimension).unsqueeze(0)
    rhs = torch.randn(batch, 3, 8)

    def batched_vjp(vector):
        flat = vector.reshape(batch, dimension)
        return torch.einsum(
            "bij,bj->bi", jacobians.transpose(1, 2), flat
        ).reshape_as(vector)

    together, _, together_rel = gmres_solve(
        batched_vjp, rhs, max_iter=40, tol=1e-7, restart=12
    )
    separate = []
    for index in range(batch):
        matrix = jacobians[index]

        def single_vjp(vector, matrix=matrix):
            flat = vector.reshape(1, dimension)
            return (flat @ matrix).reshape_as(vector)

        solved, _, rel = gmres_solve(
            single_vjp,
            rhs[index : index + 1],
            max_iter=40,
            tol=1e-7,
            restart=12,
        )
        assert rel < 1e-7
        separate.append(solved)
    separate = torch.cat(separate, dim=0)
    assert together_rel < 1e-7
    assert torch.allclose(together, separate, atol=3e-6, rtol=3e-6)


def test_local_stability_penalty_tracks_linear_spectral_norm():
    from src.fpsa_prime.stability import local_jacobian_spectral_penalty

    gain = torch.tensor(0.70, requires_grad=True)
    state = torch.randn(3, 4, 5)
    result = local_jacobian_spectral_penalty(
        lambda value: gain * value,
        state,
        target=0.60,
        power_steps=2,
        epsilon=1e-3,
    )
    assert torch.allclose(
        result["estimate"], torch.full((3,), 0.70), atol=2e-4, rtol=2e-4
    )
    assert result["loss"] > 0
    result["loss"].backward()
    assert gain.grad is not None and gain.grad > 0

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

def test_sudoku_energy_prefers_a_valid_solution():
    solution = torch.tensor(
        [
            5, 3, 4, 6, 7, 8, 9, 1, 2,
            6, 7, 2, 1, 9, 5, 3, 4, 8,
            1, 9, 8, 3, 4, 2, 5, 6, 7,
            8, 5, 9, 7, 6, 1, 4, 2, 3,
            4, 2, 6, 8, 5, 3, 7, 9, 1,
            7, 1, 3, 9, 2, 4, 8, 5, 6,
            9, 6, 1, 5, 3, 7, 2, 8, 4,
            2, 8, 7, 4, 1, 9, 6, 3, 5,
            3, 4, 5, 2, 8, 6, 1, 7, 9,
        ]
    ).unsqueeze(0)
    puzzle = solution.clone()
    puzzle[:, ::3] = 0
    valid_logits = torch.full((1, 81, 10), -12.0)
    valid_logits.scatter_(-1, solution.unsqueeze(-1), 12.0)
    invalid = solution.clone()
    invalid[:, 0] = invalid[:, 1]
    invalid_logits = torch.full((1, 81, 10), -12.0)
    invalid_logits.scatter_(-1, invalid.unsqueeze(-1), 12.0)
    assert sudoku_energy(valid_logits, puzzle) < sudoku_energy(
        invalid_logits, puzzle
    )
    assert sudoku_discrete_violations(solution, puzzle).item() == 0
    assert sudoku_discrete_violations(invalid, puzzle).item() > 0

def _saved_tensor_numel(arch: str, iterations: int) -> int:
    torch.manual_seed(8)
    model = build_model(
        arch,
        **_tiny_kwargs(
            max_iter=iterations,
            max_iter_eval=iterations,
            fp_tol=1e-10,
        ),
    )
    model.train()
    inputs = torch.randint(0, 10, (2, 9))
    targets = torch.randint(0, 10, (2, 9))
    saved = 0

    def pack(tensor):
        nonlocal saved
        saved += tensor.numel()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        logits = model(inputs)["logits"]
        assert isinstance(logits, torch.Tensor)
        loss = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
        loss.backward()
    return saved

def test_implicit_saved_activation_size_is_constant_in_iteration_depth():
    implicit_short = _saved_tensor_numel("fpsa_prime", 4)
    implicit_long = _saved_tensor_numel("fpsa_prime", 24)
    bptt_short = _saved_tensor_numel("fpsa_prime_bptt", 4)
    bptt_long = _saved_tensor_numel("fpsa_prime_bptt", 24)
    assert implicit_short == implicit_long
    assert bptt_long > 2 * bptt_short
    assert bptt_long > 2 * implicit_long

def test_particle_inference_requires_an_energy_source():
    model = build_model("fpsa_prime", **_tiny_kwargs())
    model.eval()
    inputs = torch.randint(0, 10, (2, 9))
    try:
        model.forward_particles(
            inputs,
            num_particles=2,
            init_std=0.1,
        )
    except ValueError as error:
        assert "energy_fn" in str(error)
    else:
        raise AssertionError("particles were ranked without an energy source")

def test_particle_inference_requires_eval_mode_and_uses_task_energy():
    torch.manual_seed(11)
    model = build_model("fpsa_prime", **_tiny_kwargs())
    inputs = torch.randint(0, 10, (2, 9))
    try:
        model.forward_particles(
            inputs,
            num_particles=2,
            init_std=0.01,
            energy_fn=lambda logits, _: -logits[..., 1].mean(dim=1),
        )
    except RuntimeError as error:
        assert "model.eval" in str(error)
    else:
        raise AssertionError("particle inference was allowed in training mode")

    model.eval()
    output = model.forward_particles(
        inputs,
        num_particles=3,
        init_std=0.01,
        energy_fn=lambda logits, _: -logits[..., 1].mean(dim=1),
        max_iter=20,
    )
    assert output["all_logits"].shape == (2, 3, 9, 10)
    assert output["energies"].shape == (2, 3)
    assert output["valid_particles"].shape == (2, 3)
    assert output["best_particle"].shape == (2,)
    assert output["valid_particles"].any(dim=1).all()
    best_energy = output["selection_energy"].gather(
        1, output["best_particle"].unsqueeze(1)
    ).squeeze(1)
    assert torch.isfinite(best_energy).all()
    assert torch.isfinite(output["logits"]).all()

def test_eval_is_deterministic():
    torch.manual_seed(9)
    model = build_model("fpsa_prime", **_tiny_kwargs())
    model.eval()
    inputs = torch.randint(0, 10, (2, 9))
    with torch.no_grad():
        first = model(inputs)["logits"]
        second = model(inputs)["logits"]
    assert isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
    assert torch.equal(first, second)

def test_one_training_step_is_finite():
    torch.manual_seed(10)
    model = build_model(
        "fpsa_prime",
        **_tiny_kwargs(fp_tol=1e-5, max_iter=40, backward_tol=1e-5),
    )
    model.train()
    inputs = torch.randint(0, 10, (3, 9))
    targets = torch.randint(0, 10, (3, 9))
    output = model(inputs)
    logits = output["logits"]
    assert isinstance(logits, torch.Tensor)
    loss = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
    loss.backward()
    gradient = _flat_grad(model)
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0

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

def test_implicit_gradient_matches_deep_unrolling():
    torch.manual_seed(4)
    inputs = torch.randint(0, 10, (2, 9))
    targets = torch.randint(0, 10, (2, 9))

    implicit = build_model("fpsa_prime", **_tiny_kwargs(grad_mode="implicit"))
    unrolled = build_model("fpsa_prime_bptt", **_tiny_kwargs(grad_mode="bptt"))
    unrolled.load_state_dict(implicit.state_dict())

    implicit.train()
    unrolled.train()
    implicit.zero_grad(set_to_none=True)
    unrolled.zero_grad(set_to_none=True)
    implicit_output = implicit(inputs)["logits"]
    unrolled_output = unrolled(inputs)["logits"]
    assert isinstance(implicit_output, torch.Tensor)
    assert isinstance(unrolled_output, torch.Tensor)
    loss_implicit = F.cross_entropy(
        implicit_output.reshape(-1, 10), targets.reshape(-1)
    )
    loss_unrolled = F.cross_entropy(
        unrolled_output.reshape(-1, 10), targets.reshape(-1)
    )
    loss_implicit.backward()
    loss_unrolled.backward()
    gradient_implicit = _flat_grad(implicit)
    gradient_unrolled = _flat_grad(unrolled)
    cosine = F.cosine_similarity(
        gradient_implicit, gradient_unrolled, dim=0
    ).item()
    relative = (
        (gradient_implicit - gradient_unrolled).norm()
        / gradient_unrolled.norm().clamp_min(1e-12)
    ).item()
    assert implicit.last_info.rel_residual < 1e-6
    assert cosine > 0.999, cosine
    assert relative < 0.03, relative

def test_forward_returns_the_numerical_fixed_point_not_one_more_map_step():
    torch.manual_seed(5)
    model = build_model("fpsa_prime", **_tiny_kwargs())
    model.train()
    inputs = torch.randint(0, 10, (2, 9))
    output = model(inputs)
    context = model.last_context
    residual = output["residual"]
    assert context is not None and isinstance(residual, torch.Tensor)
    mapped = model.attention.fixed_map(residual.detach(), context)
    assert torch.allclose(residual.detach(), mapped, atol=2e-5, rtol=2e-5)

def test_one_step_ablation_preserves_the_numerical_forward_value():
    torch.manual_seed(6)
    implicit = build_model("fpsa_prime", **_tiny_kwargs())
    one_step = build_model("fpsa_prime_one_step", **_tiny_kwargs())
    one_step.load_state_dict(implicit.state_dict())
    inputs = torch.randint(0, 10, (2, 9))
    implicit.train()
    one_step.train()
    output_implicit = implicit(inputs)
    output_one = one_step(inputs)
    assert torch.equal(output_implicit["residual"], output_one["residual"])
    assert torch.equal(output_implicit["logits"], output_one["logits"])

def test_structural_relations_masks_and_global_slots_are_valid():
    sudoku = sudoku_relation_ids()
    assert sudoku.shape == (81, 81)
    assert int(sudoku.diagonal().unique()) == SUDOKU_RELATIONS["self"]
    assert sudoku[0, 1] == SUDOKU_RELATIONS["row_box"]
    assert sudoku[0, 9] == SUDOKU_RELATIONS["column_box"]
    assert sudoku[0, 10] == SUDOKU_RELATIONS["box"]
    assert sudoku[0, 4] == SUDOKU_RELATIONS["row"]
    assert sudoku[0, 36] == SUDOKU_RELATIONS["column"]

    maze = maze_relation_ids(3, 4)
    assert maze[4, 0] == MAZE_RELATIONS["up"]
    assert maze[0, 4] == MAZE_RELATIONS["down"]
    assert maze[1, 0] == MAZE_RELATIONS["left"]
    assert maze[0, 1] == MAZE_RELATIONS["right"]
    assert maze[0, 5] == MAZE_RELATIONS["other"]
    allowed = tuple(
        MAZE_RELATIONS[name]
        for name in ("self", "up", "down", "left", "right")
    )
    bias = local_attention_bias(
        maze,
        num_heads=4,
        allowed_relations=allowed,
        num_global_heads=1,
    )
    assert bias.shape == (1, 4, 12, 12)
    assert torch.isneginf(bias[0, 0, 0, 5])
    assert bias[0, -1, 0, 5] == 0

    padded_relation = prepend_global_slots(
        maze, 2, slot_relation=MAZE_RELATIONS["slot"]
    )
    padded_bias = prepend_global_attention_bias(bias, 2)
    assert padded_relation.shape == (14, 14)
    assert padded_bias.shape == (1, 4, 14, 14)
    assert padded_relation[0, -1] == MAZE_RELATIONS["slot"]
    assert padded_bias[0, 0, 0, -1] == 0
    assert torch.isneginf(padded_bias[0, 0, 2, 7])

def test_fully_masked_attention_row_is_rejected():
    cfg = FPSAPrimeConfig(**_tiny_kwargs())
    attention = DualBankFixedPointAttention(cfg)
    anchor = torch.randn(1, 9, cfg.hidden_size)
    bias = torch.zeros(9, 9)
    bias[3] = float("-inf")
    try:
        attention.prepare(anchor, attention_bias=bias)
    except ValueError as error:
        assert "fully masked" in str(error)
    else:
        raise AssertionError("fully masked attention row was accepted")

def test_batched_relation_bias_is_differentiable():
    torch.manual_seed(7)
    cfg = FPSAPrimeConfig(**_tiny_kwargs(num_relation_types=3))
    attention = DualBankFixedPointAttention(cfg)
    anchor = torch.randn(2, 9, cfg.hidden_size)
    relation_ids = torch.randint(0, 3, (2, 9, 9), dtype=torch.long)
    context = attention.prepare(anchor, relation_ids=relation_ids)
    output = attention.fixed_map(torch.zeros_like(anchor), context)
    output.square().mean().backward()
    assert attention.relation_bias is not None
    assert attention.relation_bias.grad is not None
    assert torch.isfinite(attention.relation_bias.grad).all()

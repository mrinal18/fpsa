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


def test_safe_defaults_require_forward_and_backward_convergence():
    config = FPSAPrimeConfig()
    assert config.require_convergence
    assert config.require_backward_convergence


def test_invalid_runtime_enum_is_rejected():
    try:
        FPSAPrimeConfig(value_mode="unknown")  # type: ignore[arg-type]
    except ValueError as error:
        assert "value_mode" in str(error)
    else:
        raise AssertionError("invalid value_mode was accepted")


def test_strict_forward_rejects_an_unconverged_state():
    model = build_model(
        "fpsa_prime",
        **_tiny_kwargs(
            max_iter=1,
            max_iter_eval=1,
            fp_tol=1e-12,
            require_convergence=True,
        ),
    )
    inputs = torch.randint(0, 10, (1, 9))
    try:
        model(inputs)
    except RuntimeError as error:
        assert "did not converge" in str(error)
    else:
        raise AssertionError("strict equilibrium mode accepted an unconverged solve")


def test_only_attention_is_recurrent():
    model = build_model("fpsa_prime", **_tiny_kwargs())
    recurrent_parameters = {name for name, _ in model.attention.named_parameters()}
    assert recurrent_parameters
    assert not any(
        "mlp" in name.lower() or "conv" in name.lower()
        for name in recurrent_parameters
    )
    assert model.input_mlp is None and model.output_mlp is None


def test_value_bank_presets_are_wired_correctly():
    dual = build_model("fpsa_prime", **_tiny_kwargs()).attention
    fixed = build_model("fpsa_fixed_v", **_tiny_kwargs()).attention
    dynamic = build_model("fpsa_dynamic_v", **_tiny_kwargs()).attention
    assert dual.W_V_evidence is not None and dual.W_V_scratch is not None
    assert fixed.W_V_evidence is not None and fixed.W_V_scratch is None
    assert dynamic.W_V_evidence is None and dynamic.W_V_scratch is not None


def test_evidence_values_are_frozen_while_scratch_values_evolve():
    torch.manual_seed(0)
    cfg = FPSAPrimeConfig(**_tiny_kwargs())
    attention = DualBankFixedPointAttention(cfg)
    anchor = torch.randn(2, 9, cfg.hidden_size)
    context = attention.prepare(anchor)
    assert context.evidence_values is not None
    frozen_before = context.evidence_values.clone()
    residual_zero = torch.zeros_like(anchor)
    residual_random = torch.randn_like(anchor)
    output_zero = attention.fixed_map(residual_zero, context)
    output_random = attention.fixed_map(residual_random, context)
    assert torch.equal(context.evidence_values, frozen_before)
    assert not torch.allclose(output_zero, output_random)


def test_default_hero_is_parameter_matched_and_full_model_is_explicit():
    control_params = 135_558
    hero = build_model(
        "fpsa_prime",
        vocab_size=4,
        out_vocab_size=2,
        max_seq_len=49,
        hidden_size=128,
        num_heads=8,
        num_relation_types=len(MAZE_RELATIONS),
    )
    decoder = build_model(
        "fpsa_prime_decoder_mlp",
        vocab_size=4,
        out_vocab_size=2,
        max_seq_len=49,
        hidden_size=128,
        num_heads=8,
        num_relation_types=len(MAZE_RELATIONS),
    )
    full = build_model(
        "fpsa_prime_full",
        vocab_size=4,
        out_vocab_size=2,
        max_seq_len=49,
        hidden_size=128,
        num_heads=8,
        num_relation_types=len(MAZE_RELATIONS),
    )
    assert hero.cfg.use_input_mlp and not hero.cfg.use_output_mlp
    assert not decoder.cfg.use_input_mlp and decoder.cfg.use_output_mlp
    assert full.cfg.use_input_mlp and full.cfg.use_output_mlp
    assert abs(hero.n_params() - control_params) / control_params < 0.01
    assert hero.n_params() == decoder.n_params()
    assert full.n_params() > 1.45 * control_params


def test_legacy_bptt_alias_maps_to_explicit_forward_and_backward_modes():
    config = FPSAPrimeConfig(grad_mode="bptt")
    assert config.forward_mode == "fixed_unroll"
    assert config.backward_mode == "bptt"

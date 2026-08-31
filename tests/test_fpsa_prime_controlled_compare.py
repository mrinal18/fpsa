"""Regression tests for the controlled-comparison runtime contract."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.fpsa_prime.controlled_compare import evaluate, parser


def test_controlled_compare_uses_explicit_strict_evaluation_headroom():
    args = parser().parse_args(
        ["--family", "prime", "--arch", "fpsa_prime"]
    )
    assert args.max_iter == 16
    assert args.max_iter_eval == 64
    assert args.stability_power_steps == 4
    assert not args.allow_nonconvergence


class _DummyPrime:
    def __init__(self):
        self.cfg = SimpleNamespace(max_iter_eval=64)
        self.seen = []

    def eval(self):
        return self

    def __call__(
        self,
        tokens,
        *,
        relation_ids=None,
        attention_bias=None,
        max_iter=None,
        require_convergence=None,
    ):
        self.seen.append((max_iter, require_convergence))
        batch, length = tokens.shape
        logits = torch.zeros(batch, length, 2)
        logits[..., 0] = 1.0
        info = SimpleNamespace(
            rel_residual=8e-5,
            n_iters=11,
            n_function_evals=13,
            converged_frac=1.0,
            converged=torch.ones(batch, dtype=torch.bool),
            per_sample_iters=torch.arange(1, batch + 1),
        )
        return {"logits": logits, "info": info}


def test_runtime_evaluation_reports_tail_solver_statistics():
    model = _DummyPrime()
    tokens = torch.zeros(4, 5, dtype=torch.long)
    targets = torch.zeros_like(tokens)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    loader = DataLoader(TensorDataset(tokens, targets, mask), batch_size=4)

    metrics = evaluate(
        "prime",
        model,
        loader,
        device=torch.device("cpu"),
        max_iter=64,
        require_convergence=True,
    )

    assert model.seen == [(64, True)]
    assert metrics["exact_match"] == 100.0
    assert metrics["max_residual"] == 8e-5
    assert metrics["min_batch_converged_fraction"] == 1.0
    assert metrics["p50_sample_iterations"] == 2.5
    assert metrics["p90_sample_iterations"] > 3.0
    assert metrics["max_sample_iterations"] == 4.0


def test_runtime_evaluation_keeps_strict_nonconvergence_fail_fast():
    class FailingPrime(_DummyPrime):
        def __call__(self, *args, **kwargs):
            raise RuntimeError(
                "fixed-point solve did not converge: residual=1.663e-03, "
                "converged_frac=0.938"
            )

    model = FailingPrime()
    tokens = torch.zeros(2, 5, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(
            tokens,
            torch.zeros_like(tokens),
            torch.ones_like(tokens, dtype=torch.bool),
        ),
        batch_size=2,
    )
    try:
        evaluate(
            "prime",
            model,
            loader,
            device=torch.device("cpu"),
            max_iter=64,
            require_convergence=True,
        )
    except RuntimeError as error:
        message = str(error)
        assert "max_iter_eval=64" in message
        assert "not accepted as an equilibrium" in message
    else:
        raise AssertionError("strict evaluation accepted an unconverged state")

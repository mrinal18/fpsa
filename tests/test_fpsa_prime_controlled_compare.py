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
    assert args.train_max_iter_ceiling == 96
    assert args.eval_max_iter_ceiling == 256
    assert args.stability_power_steps == 4
    assert not args.disable_cap_escalation
    assert not args.allow_nonconvergence


class _DummyPrime:
    def __init__(self):
        self.cfg = SimpleNamespace(
            max_iter=24,
            max_iter_eval=64,
            require_convergence=True,
        )
        self.training = False
        self.seen = []

    def eval(self):
        self.training = False
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


def _loader(batch: int = 4, length: int = 5):
    tokens = torch.zeros(batch, length, dtype=torch.long)
    targets = torch.zeros_like(tokens)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    return DataLoader(TensorDataset(tokens, targets, mask), batch_size=batch)


def test_runtime_evaluation_reports_tail_solver_statistics():
    model = _DummyPrime()
    metrics = evaluate(
        "prime",
        model,
        _loader(),
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
    assert metrics["max_solver_cap"] == 64.0
    assert metrics["batches_with_cap_escalation"] == 0.0


def test_runtime_evaluation_retries_the_same_fixed_point_with_a_larger_cap():
    class TailPrime(_DummyPrime):
        def __call__(self, tokens, **kwargs):
            cap = int(kwargs["max_iter"])
            self.seen.append((cap, kwargs.get("require_convergence")))
            if cap < 128:
                raise RuntimeError(
                    "fixed-point solve did not converge: residual=1.663e-03, "
                    "converged_frac=0.938"
                )
            batch, length = tokens.shape
            logits = torch.zeros(batch, length, 2)
            logits[..., 0] = 1.0
            info = SimpleNamespace(
                rel_residual=8e-5,
                n_iters=37,
                n_function_evals=39,
                converged_frac=1.0,
                converged=torch.ones(batch, dtype=torch.bool),
                per_sample_iters=torch.full((batch,), 37),
            )
            return {"logits": logits, "info": info}

    model = TailPrime()
    model._runtime_eval_max_iter_ceiling = 256
    model._runtime_cap_escalation = True
    metrics = evaluate(
        "prime",
        model,
        _loader(),
        device=torch.device("cpu"),
        max_iter=64,
        require_convergence=True,
    )

    assert model.seen == [(64, True), (128, True)]
    assert metrics["exact_match"] == 100.0
    assert metrics["max_solver_cap"] == 128.0
    assert metrics["batches_with_cap_escalation"] == 1.0
    assert metrics["mean_cap_retries"] == 1.0
    assert metrics["max_residual"] < 1e-4


def test_runtime_evaluation_keeps_strict_nonconvergence_fail_fast_at_ceiling():
    class FailingPrime(_DummyPrime):
        def __call__(self, *args, **kwargs):
            self.seen.append(
                (kwargs.get("max_iter"), kwargs.get("require_convergence"))
            )
            raise RuntimeError(
                "fixed-point solve did not converge: residual=1.663e-03, "
                "converged_frac=0.938"
            )

    model = FailingPrime()
    model._runtime_eval_max_iter_ceiling = 256
    model._runtime_cap_escalation = True
    try:
        evaluate(
            "prime",
            model,
            _loader(batch=2),
            device=torch.device("cpu"),
            max_iter=64,
            require_convergence=True,
        )
    except RuntimeError as error:
        message = str(error)
        assert "initial max_iter_eval=64" in message
        assert "retry ceiling=256" in message
        assert "not accepted as an equilibrium" in message
        assert model.seen == [
            (64, True),
            (128, True),
            (256, True),
        ]
    else:
        raise AssertionError("strict evaluation accepted an unconverged state")

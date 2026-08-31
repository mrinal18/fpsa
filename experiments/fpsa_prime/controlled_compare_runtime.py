"""Runtime-safe wrapper for the shared FPSA-Prime comparison harness.

The training fixed-point cap and the evaluation cap serve different purposes.
Training controls optimization cost. Evaluation must give every example enough
iterations to satisfy the declared equilibrium tolerance; otherwise a handful
of hard examples can terminate a run even when most of the batch converged.

This wrapper keeps strict convergence enabled. It never accepts or scores an
unconverged state as an equilibrium. When the initial numerical budget is too
small, it retries the *same fixed-point equation* with a larger explicit cap,
logs that escalation, and reports the realized NFE and iteration tails. If the
largest configured ceiling still fails, the run stops and writes a diagnostic
checkpoint instead of silently loosening the tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from . import controlled_compare_core as base

_ACTIVE_MODEL = None


def _is_forward_nonconvergence(error: BaseException) -> bool:
    return "fixed-point solve did not converge" in str(error)


def _cap_schedule(initial: int, ceiling: int) -> list[int]:
    if initial <= 0 or ceiling <= 0:
        raise ValueError("solver iteration caps must be positive")
    if ceiling < initial:
        raise ValueError("solver retry ceiling cannot be below the initial cap")
    caps = [initial]
    while caps[-1] < ceiling:
        caps.append(min(ceiling, 2 * caps[-1]))
    return caps


def _forward(
    family: str,
    model,
    tokens: torch.Tensor,
    *,
    relation_ids=None,
    attention_bias=None,
    max_iter: Optional[int] = None,
    require_convergence: Optional[bool] = None,
):
    """Run one forward pass, escalating only the numerical cap when necessary."""
    if family != "prime":
        return model(tokens, max_iter=max_iter)

    training = bool(model.training)
    configured_initial = int(
        model.cfg.max_iter if training else model.cfg.max_iter_eval
    )
    initial = configured_initial if max_iter is None else int(max_iter)
    ceiling = int(
        getattr(
            model,
            "_runtime_train_max_iter_ceiling"
            if training
            else "_runtime_eval_max_iter_ceiling",
            initial,
        )
    )
    escalation_enabled = bool(
        getattr(model, "_runtime_cap_escalation", True)
    )
    if not escalation_enabled:
        ceiling = initial

    strict = (
        bool(model.cfg.require_convergence)
        if require_convergence is None
        else bool(require_convergence)
    )
    caps = _cap_schedule(initial, ceiling) if strict else [initial]
    last_error: Optional[RuntimeError] = None

    if training:
        model._runtime_training_step = int(
            getattr(model, "_runtime_training_step", 0)
        ) + 1

    for attempt, cap in enumerate(caps):
        try:
            output = model(
                tokens,
                relation_ids=relation_ids,
                attention_bias=attention_bias,
                max_iter=cap,
                require_convergence=strict,
            )
            info = output.get("info")
            if info is not None:
                setattr(info, "runtime_initial_cap", initial)
                setattr(info, "runtime_used_cap", cap)
                setattr(info, "runtime_retry_count", attempt)
                setattr(info, "runtime_cap_escalated", attempt > 0)
            model._runtime_last_initial_cap = initial
            model._runtime_last_used_cap = cap
            model._runtime_last_retry_count = attempt
            model._runtime_total_retry_count = int(
                getattr(model, "_runtime_total_retry_count", 0)
            ) + attempt
            if attempt > 0:
                print(
                    json.dumps(
                        {
                            "event": "forward_cap_escalation_succeeded",
                            "phase": "train" if training else "eval",
                            "step": int(
                                getattr(model, "_runtime_training_step", 0)
                            ),
                            "initial_cap": initial,
                            "used_cap": cap,
                            "retry_count": attempt,
                            "residual": float(
                                getattr(info, "rel_residual", float("nan"))
                            ),
                            "converged_fraction": float(
                                getattr(info, "converged_frac", float("nan"))
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return output
        except RuntimeError as error:
            if not strict or not _is_forward_nonconvergence(error):
                raise
            last_error = error
            if cap >= ceiling:
                break
            print(
                json.dumps(
                    {
                        "event": "forward_cap_escalation_retry",
                        "phase": "train" if training else "eval",
                        "step": int(
                            getattr(model, "_runtime_training_step", 0)
                        ),
                        "failed_cap": cap,
                        "next_cap": min(ceiling, 2 * cap),
                        "error": str(error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    assert last_error is not None
    raise RuntimeError(
        "strict FPSA-Prime forward failed after numerical cap escalation: "
        f"phase={'train' if training else 'eval'}, initial_cap={initial}, "
        f"ceiling={ceiling}, step={int(getattr(model, '_runtime_training_step', 0))}. "
        "The state was not accepted as an equilibrium. This is now a real "
        "convergence failure rather than an arbitrary short-cap failure. "
        f"Last error: {last_error}"
    ) from last_error


@torch.no_grad()
def evaluate(
    family: str,
    model,
    loader,
    *,
    device: torch.device,
    relation_ids=None,
    attention_bias=None,
    max_iter: Optional[int] = None,
    require_convergence: bool = True,
) -> dict[str, float]:
    """Evaluate with strict equilibrium checks and tail diagnostics."""
    model.eval()
    exact = total = token_ok = token_total = 0
    residual_sum = iteration_sum = convergence_sum = nfe_sum = 0.0
    worst_residual = 0.0
    minimum_converged_fraction = 1.0
    converged_samples = 0
    sample_iterations: list[torch.Tensor] = []
    used_caps: list[float] = []
    retry_counts: list[float] = []
    batches = 0

    for batch_index, (x, y, mask) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        try:
            output = _forward(
                family,
                model,
                x,
                relation_ids=relation_ids,
                attention_bias=attention_bias,
                max_iter=max_iter,
                require_convergence=require_convergence,
            )
        except RuntimeError as error:
            if not _is_forward_nonconvergence(error) and (
                "strict FPSA-Prime forward failed" not in str(error)
            ):
                raise
            budget = max_iter
            if budget is None:
                budget = int(getattr(model.cfg, "max_iter_eval", 0))
            ceiling = int(
                getattr(model, "_runtime_eval_max_iter_ceiling", budget)
            )
            raise RuntimeError(
                "strict evaluation failed on batch "
                f"{batch_index} with initial max_iter_eval={budget} and "
                f"retry ceiling={ceiling}. The state was not accepted as an "
                "equilibrium. Inspect the stability estimate and residual "
                "trajectory; do not use --allow_nonconvergence for the reported "
                f"comparison. Original error: {error}"
            ) from error

        logits = output["logits"]
        prediction = logits.argmax(-1)
        correct = (prediction == y) | ~mask
        exact += int(correct.all(dim=1).sum())
        total += x.shape[0]
        token_ok += int((correct & mask).sum())
        token_total += int(mask.sum())

        info = output["info"]
        batch_residual = float(info.rel_residual)
        batch_convergence = float(getattr(info, "converged_frac", 1.0))
        residual_sum += batch_residual
        iteration_sum += float(info.n_iters)
        convergence_sum += batch_convergence
        nfe_sum += float(getattr(info, "n_function_evals", info.n_iters))
        worst_residual = max(worst_residual, batch_residual)
        minimum_converged_fraction = min(
            minimum_converged_fraction, batch_convergence
        )
        used_caps.append(
            float(getattr(info, "runtime_used_cap", max_iter or 0))
        )
        retry_counts.append(
            float(getattr(info, "runtime_retry_count", 0))
        )

        converged = getattr(info, "converged", None)
        if isinstance(converged, torch.Tensor):
            converged_samples += int(converged.sum())
        else:
            converged_samples += int(round(batch_convergence * x.shape[0]))
        per_sample_iters = getattr(info, "per_sample_iters", None)
        if isinstance(per_sample_iters, torch.Tensor):
            sample_iterations.append(per_sample_iters.detach().float().cpu())
        batches += 1

    denominator = max(1, batches)
    result = {
        "exact_match": 100.0 * exact / max(1, total),
        "token_accuracy": 100.0 * token_ok / max(1, token_total),
        "mean_residual": residual_sum / denominator,
        "max_residual": worst_residual,
        "mean_iterations": iteration_sum / denominator,
        "mean_function_evals": nfe_sum / denominator,
        "mean_converged_fraction": convergence_sum / denominator,
        "min_batch_converged_fraction": minimum_converged_fraction,
        "converged_samples": float(converged_samples),
        "total_samples": float(total),
        "mean_solver_cap": sum(used_caps) / max(1, len(used_caps)),
        "max_solver_cap": max(used_caps) if used_caps else 0.0,
        "batches_with_cap_escalation": float(
            sum(value > 0 for value in retry_counts)
        ),
        "mean_cap_retries": sum(retry_counts) / max(1, len(retry_counts)),
    }
    if sample_iterations:
        iterations = torch.cat(sample_iterations)
        result.update(
            {
                "p50_sample_iterations": float(
                    torch.quantile(iterations, 0.50)
                ),
                "p90_sample_iterations": float(
                    torch.quantile(iterations, 0.90)
                ),
                "p99_sample_iterations": float(
                    torch.quantile(iterations, 0.99)
                ),
                "max_sample_iterations": float(iterations.max()),
            }
        )
    return result


def _save_failure_checkpoint(args, model, error: BaseException) -> None:
    """Preserve a failed run for diagnosis instead of discarding all progress."""
    output = getattr(args, "output", "")
    if not output:
        return
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    failure_path = destination.with_suffix(".failure.json")
    checkpoint_path = destination.with_suffix(".failure.pt")
    step = int(getattr(model, "_runtime_training_step", 0))
    failure_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "step": step,
                "error_type": type(error).__name__,
                "error": str(error),
                "last_initial_cap": int(
                    getattr(model, "_runtime_last_initial_cap", 0)
                ),
                "last_used_cap": int(
                    getattr(model, "_runtime_last_used_cap", 0)
                ),
                "total_cap_retries": int(
                    getattr(model, "_runtime_total_retry_count", 0)
                ),
                "args": vars(args),
            },
            indent=2,
        )
    )
    torch.save(
        {
            "step": step,
            "config": model.cfg.to_dict(),
            "state_dict": model.state_dict(),
            "error": str(error),
        },
        checkpoint_path,
    )


def run(args) -> dict:
    """Run the shared trainer with explicit strict solver ceilings."""
    global _ACTIVE_MODEL

    if args.max_iter_eval is None:
        args.max_iter_eval = max(64, int(args.max_iter))
    if args.max_iter_eval < args.max_iter:
        raise ValueError(
            "max_iter_eval must be at least max_iter for a strict equilibrium "
            "comparison"
        )
    if args.train_max_iter_ceiling < args.max_iter:
        raise ValueError("train_max_iter_ceiling must be >= max_iter")
    if args.eval_max_iter_ceiling < args.max_iter_eval:
        raise ValueError("eval_max_iter_ceiling must be >= max_iter_eval")

    print(
        "solver_protocol="
        + json.dumps(
            {
                "train_initial_cap": int(args.max_iter),
                "train_retry_ceiling": int(args.train_max_iter_ceiling),
                "eval_initial_cap": int(args.max_iter_eval),
                "eval_retry_ceiling": int(args.eval_max_iter_ceiling),
                "cap_escalation": not bool(args.disable_cap_escalation),
                "fp_tol": float(args.fp_tol),
                "strict_forward": not bool(args.allow_nonconvergence),
                "note": (
                    "Cap escalation changes only numerical effort, not the "
                    "fixed-point equation. Realized NFEs and retry counts are "
                    "reported."
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    original_build = base._build_model
    original_forward = base._forward
    original_evaluate = base.evaluate

    def build_with_runtime(*build_args, **build_kwargs):
        global _ACTIVE_MODEL
        model = original_build(*build_args, **build_kwargs)
        model._runtime_train_max_iter_ceiling = int(
            args.train_max_iter_ceiling
        )
        model._runtime_eval_max_iter_ceiling = int(
            args.eval_max_iter_ceiling
        )
        model._runtime_cap_escalation = not bool(
            args.disable_cap_escalation
        )
        model._runtime_training_step = 0
        model._runtime_total_retry_count = 0
        _ACTIVE_MODEL = model
        return model

    base._build_model = build_with_runtime
    base._forward = _forward
    base.evaluate = evaluate
    try:
        result = base.run(args)
    except BaseException as error:
        if _ACTIVE_MODEL is not None:
            _save_failure_checkpoint(args, _ACTIVE_MODEL, error)
        raise
    finally:
        base._build_model = original_build
        base._forward = original_forward
        base.evaluate = original_evaluate

    result["solver_protocol"] = {
        "train_initial_cap": int(args.max_iter),
        "train_retry_ceiling": int(args.train_max_iter_ceiling),
        "eval_initial_cap": int(args.max_iter_eval),
        "eval_retry_ceiling": int(args.eval_max_iter_ceiling),
        "cap_escalation": not bool(args.disable_cap_escalation),
        "fp_tol": float(args.fp_tol),
        "strict_forward": not bool(args.allow_nonconvergence),
        "total_cap_retries": int(
            getattr(_ACTIVE_MODEL, "_runtime_total_retry_count", 0)
        ),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    return result


def parser():
    parser = base.parser()
    parser.set_defaults(max_iter_eval=64, stability_power_steps=4)
    parser.add_argument(
        "--train_max_iter_ceiling",
        type=int,
        default=96,
        help=(
            "largest strict training cap after deterministic retries; the "
            "default schedule from max_iter=24 is 24, 48, 96"
        ),
    )
    parser.add_argument(
        "--eval_max_iter_ceiling",
        type=int,
        default=256,
        help=(
            "largest strict evaluation cap after deterministic retries; the "
            "default schedule from max_iter_eval=64 is 64, 128, 256"
        ),
    )
    parser.add_argument(
        "--disable_cap_escalation",
        action="store_true",
        help="disable numerical cap retries and fail at the initial cap",
    )
    return parser


__all__ = ["evaluate", "parser", "run"]

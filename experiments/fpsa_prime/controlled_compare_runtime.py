"""Runtime-safe wrapper for the shared FPSA-Prime comparison harness.

The training fixed-point cap and the evaluation cap serve different purposes.
Training controls optimization cost. Evaluation must give every example enough
iterations to satisfy the declared equilibrium tolerance; otherwise a handful
of hard examples can terminate a run even when most of the batch converged.

This wrapper keeps strict convergence enabled. It does not accept or score an
unconverged state as an equilibrium. Instead it gives evaluation an explicit,
larger cap and reports tail convergence statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from . import controlled_compare_core as base


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
    if family == "prime":
        return model(
            tokens,
            relation_ids=relation_ids,
            attention_bias=attention_bias,
            max_iter=max_iter,
            require_convergence=require_convergence,
        )
    return model(tokens, max_iter=max_iter)


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
            if "fixed-point solve did not converge" not in str(error):
                raise
            budget = max_iter
            if budget is None:
                budget = int(getattr(model.cfg, "max_iter_eval", 0))
            raise RuntimeError(
                "strict evaluation failed on batch "
                f"{batch_index} with max_iter_eval={budget}. "
                "The state was not accepted as an equilibrium. Increase the "
                "explicit evaluation cap, inspect the stability estimate, or "
                "treat the run as a convergence failure. Original error: "
                f"{error}"
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


def _save_failure_checkpoint(args, model, step: int, error: BaseException) -> None:
    """Preserve a failed run for diagnosis instead of discarding all progress."""
    output = getattr(args, "output", "")
    if not output:
        return
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    failure_path = destination.with_suffix(".failure.json")
    checkpoint_path = destination.with_suffix(".failure.pt")
    failure_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "step": step,
                "error_type": type(error).__name__,
                "error": str(error),
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
    """Run the shared trainer with an explicit equilibrium evaluation budget."""
    if args.max_iter_eval is None:
        args.max_iter_eval = max(64, int(args.max_iter))
    if args.max_iter_eval < args.max_iter:
        raise ValueError(
            "max_iter_eval must be at least max_iter for a strict equilibrium "
            "comparison"
        )

    print(
        "evaluation_protocol="
        + json.dumps(
            {
                "train_max_iter": int(args.max_iter),
                "eval_max_iter": int(args.max_iter_eval),
                "fp_tol": float(args.fp_tol),
                "strict_forward": not bool(args.allow_nonconvergence),
                "note": (
                    "The evaluation cap is a ceiling, not the realized cost; "
                    "mean and tail NFEs are reported."
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    original_evaluate = base.evaluate
    base.evaluate = evaluate
    try:
        result = base.run(args)
    finally:
        base.evaluate = original_evaluate

    result["evaluation_protocol"] = {
        "train_max_iter": int(args.max_iter),
        "eval_max_iter": int(args.max_iter_eval),
        "fp_tol": float(args.fp_tol),
        "strict_forward": not bool(args.allow_nonconvergence),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    return result


def parser():
    parser = base.parser()
    # A fixed-point model should be evaluated at a strict equilibrium.  Sixty-
    # four is only a ceiling: adaptive solvers stop earlier and realized NFEs
    # are included in the output.  The training cap remains independently set.
    parser.set_defaults(max_iter_eval=64, stability_power_steps=4)
    return parser


__all__ = ["evaluate", "parser", "run"]

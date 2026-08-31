"""Portable Colab verification wrapper for FPSA-Prime.

The corrected GMRES solver is a correctness requirement. Reproducing a
particular numerical failure of the legacy v0 solver is only a diagnostic: its
severity depends on device, PyTorch/LAPACK/cuSOLVER version, batch shape, and
floating-point reduction order. This module keeps the strong positive checks
while recording, rather than asserting, the legacy degradation.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import torch

from . import verification_core as base
from src.fpsa_prime.solvers import gmres_solve

_LAST_GMRES_METADATA: dict[str, object] = {}


def gmres_check(
    output_dir: Path,
    device: torch.device,
    *,
    quick: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate fixed GMRES and treat legacy instability as a soft diagnostic."""
    global _LAST_GMRES_METADATA

    diagnostic = base.build_maze_diagnostic(
        device, batch=4 if quick else 32
    )
    rhs = diagnostic["rhs"]
    vjp = diagnostic["vjp"]
    assert isinstance(rhs, torch.Tensor)
    assert callable(vjp)

    rows = []
    for budget in [4, 8, 12, 15, 20, 40]:
        for name, solver in (
            ("legacy v0", base.legacy_gmres_solve),
            ("fixed", gmres_solve),
        ):
            solution, iterations, reported = solver(
                vjp, rhs, max_iter=budget, tol=1e-5, restart=20
            )
            rows.append(
                {
                    "solver": name,
                    "budget": budget,
                    "iterations_used": iterations,
                    "reported_residual": reported,
                    "true_residual": base._true_adjoint_residual(
                        solution, rhs, vjp
                    ),
                }
            )
    table = pd.DataFrame(rows)

    fixed_rows = table[table.solver == "fixed"].sort_values("budget")
    fixed_at_40 = fixed_rows[fixed_rows.budget == 40].iloc[0]
    if float(fixed_at_40.true_residual) > 1.05e-5:
        raise AssertionError(
            "fixed GMRES missed the adjoint tolerance: "
            f"true residual={float(fixed_at_40.true_residual):.3e}"
        )
    if int(fixed_at_40.iterations_used) >= 15:
        raise AssertionError(
            "fixed GMRES did not stop at the short solution: "
            f"iterations={int(fixed_at_40.iterations_used)}"
        )

    converged = fixed_rows[fixed_rows.true_residual <= 1.05e-5]
    if converged.empty:
        raise AssertionError("fixed GMRES never reached the requested tolerance")
    first_converged_budget = int(converged.budget.min())
    post_convergence = fixed_rows[
        fixed_rows.budget >= first_converged_budget
    ]
    if float(post_convergence.true_residual.max()) > 1.05e-5:
        raise AssertionError(
            "fixed GMRES lost a previously converged solution when the maximum "
            "iteration budget increased"
        )

    legacy_12 = float(
        table[
            (table.solver == "legacy v0") & (table.budget == 12)
        ].true_residual.iloc[0]
    )
    legacy_15 = float(
        table[
            (table.solver == "legacy v0") & (table.budget == 15)
        ].true_residual.iloc[0]
    )
    legacy_ratio = legacy_15 / max(legacy_12, 1e-30)
    legacy_degradation_observed = legacy_ratio > 100.0
    if not legacy_degradation_observed:
        warnings.warn(
            "The legacy v0 GMRES did not reproduce the seeded >100x long-basis "
            "degradation on this runtime. This is device/library dependent and "
            "is not a failure of the corrected solver. Inspect the recorded "
            "table and fixed-solver checks instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    _LAST_GMRES_METADATA = {
        "fixed_passed": True,
        "fixed_tolerance": 1e-5,
        "fixed_first_converged_budget": first_converged_budget,
        "fixed_iterations_at_budget_40": int(fixed_at_40.iterations_used),
        "fixed_true_residual_at_budget_40": float(
            fixed_at_40.true_residual
        ),
        "legacy_degradation_ratio_budget15_over_budget12": legacy_ratio,
        "legacy_degradation_over_100x_observed": legacy_degradation_observed,
        "legacy_check_is_diagnostic_only": True,
        "note": (
            "Legacy degradation is sensitive to accelerator and linear-algebra "
            "library details. The portable pass criterion is that corrected "
            "GMRES reaches tolerance, stops early, and retains the converged "
            "solution as its maximum budget grows."
        ),
    }

    plt.figure(figsize=(8, 4.5))
    for name, group in table.groupby("solver"):
        plt.plot(
            group.budget,
            group.true_residual,
            marker="o",
            label=name,
        )
    plt.axhline(1e-5, linestyle="--", label="backward tolerance")
    plt.yscale("log")
    plt.xlabel("Maximum Arnoldi iterations")
    plt.ylabel("True relative adjoint residual")
    plt.title("GMRES verification on an actual FPSA VJP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gmres_regression.png", dpi=160)
    plt.close()
    return table, diagnostic


def run_verification(**kwargs) -> dict[str, object]:
    """Run the existing harness with portable GMRES pass criteria."""
    original = base.gmres_check
    base.gmres_check = gmres_check
    try:
        summary = base.run_verification(**kwargs)
    finally:
        base.gmres_check = original

    summary["gmres_diagnostic"] = dict(_LAST_GMRES_METADATA)
    destination = Path(kwargs.get("output_dir", "results/verification"))
    (destination / "fpsa_prime_verification_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


if __name__ == "__main__":
    args = base.parser().parse_args()
    result = run_verification(
        output_dir=args.output_dir,
        device=args.device,
        quick=not args.full,
        run_tests=not args.skip_tests,
        run_training=args.run_training,
    )
    print(json.dumps(result, indent=2))


__all__ = ["gmres_check", "run_verification"]

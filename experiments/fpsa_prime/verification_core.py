"""Numerical verification harness used by the FPSA-Prime Colab notebook.

The checks in this module validate solver and protocol invariants. They do not
constitute a benchmark claim. The expensive multi-seed comparison remains in
``controlled_compare.py``.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.fpsa_prime.utils import masked_loss
from src.fpsa_prime import build_model as build_prime_model
from src.fpsa_prime.implicit import BACKWARD_STATS
from src.fpsa_prime.solvers import gmres_solve
from src.fpsa_prime.stability import local_jacobian_spectral_penalty
from src.fpsa_prime.structures import (
    MAZE_RELATIONS,
    local_attention_bias,
    maze_relation_ids,
)
from src.fpsa_r import build_model as build_block_model


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return selected


def run_repository_tests() -> str:
    completed = subprocess.run(
        f"{sys.executable} -m pytest tests/test_fpsa_prime_*.py -q",
        shell=True,
        text=True,
        capture_output=True,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"FPSA-Prime correctness suite failed:\n{output}")
    return output


def parameter_check(output_dir: Path) -> tuple[pd.DataFrame, float]:
    maze_common = dict(
        vocab_size=4,
        out_vocab_size=2,
        hidden_size=128,
        num_heads=8,
    )
    prime_common = dict(
        **maze_common,
        max_seq_len=49,
        num_relation_types=len(MAZE_RELATIONS),
    )
    models = {
        "FPSA-Prime hero": build_prime_model("fpsa_prime", **prime_common),
        "FPSA decoder-MLP control": build_prime_model(
            "fpsa_prime_decoder_mlp", **prime_common
        ),
        "FPSA two-MLP capacity control": build_prime_model(
            "fpsa_prime_full", **prime_common
        ),
        "FPSA fixed-V": build_prime_model("fpsa_fixed_v", **prime_common),
        "FPSA dynamic-V": build_prime_model("fpsa_dynamic_v", **prime_common),
        "Block-DEQ control": build_block_model(
            "deq_block",
            **maze_common,
            seq_len=49,
            n_block_layers=1,
            expansion=2.0,
            conv_type="conv2d",
            causal=False,
        ),
    }
    table = pd.DataFrame(
        [
            {
                "model": name,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
            for name, model in models.items()
        ]
    ).sort_values("parameters")
    hero = int(table.loc[table.model == "FPSA-Prime hero", "parameters"].iloc[0])
    block = int(table.loc[table.model == "Block-DEQ control", "parameters"].iloc[0])
    gap = abs(hero - block) / block
    if gap >= 0.01:
        raise AssertionError(f"hero/block parameter gap is {100 * gap:.3f}%")

    ordered = table.sort_values("parameters", ascending=False)
    plt.figure(figsize=(10, 4.5))
    plt.bar(ordered.model, ordered.parameters)
    plt.ylabel("Trainable parameters")
    plt.title("FPSA-Prime capacity controls")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_counts.png", dpi=160)
    plt.close()
    return table, gap


def legacy_gmres_solve(
    vjp: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iter: int,
    tol: float,
    restart: int = 20,
) -> tuple[torch.Tensor, int, float]:
    """Original v0 per-sample GMRES, retained only as a regression oracle."""
    batch = rhs.shape[0]
    state_shape = rhs.shape[1:]
    dimension = rhs[0].numel()
    original_dtype = rhs.dtype
    work_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    vector_rhs = rhs.reshape(batch, dimension).to(work_dtype)

    def operator(vectors: torch.Tensor) -> torch.Tensor:
        shaped = vectors.to(original_dtype).reshape(batch, *state_shape)
        output = shaped - vjp(shaped)
        return output.reshape(batch, dimension).to(work_dtype)

    estimate = torch.zeros_like(vector_rhs)
    rhs_norm = vector_rhs.norm(dim=-1)
    zero_rhs = rhs_norm < 1e-12
    done = zero_rhs.clone()
    relative = torch.where(
        zero_rhs, torch.zeros_like(rhs_norm), torch.ones_like(rhs_norm)
    )
    total = 0
    while total < max_iter and not bool(done.all()):
        residual = vector_rhs - operator(estimate)
        beta = residual.norm(dim=-1)
        relative = torch.where(
            zero_rhs,
            torch.zeros_like(beta),
            beta / rhs_norm.clamp_min(1e-12),
        )
        done |= relative < tol
        active = ~done
        if not bool(active.any()):
            break

        width = min(restart, max_iter - total)
        basis = torch.zeros(
            batch,
            width + 1,
            dimension,
            dtype=work_dtype,
            device=rhs.device,
        )
        hessenberg = torch.zeros(
            batch,
            width + 1,
            width,
            dtype=work_dtype,
            device=rhs.device,
        )
        basis[:, 0] = torch.where(
            active[:, None], residual / beta.clamp_min(1e-12)[:, None], 0.0
        )
        used = 0
        for column in range(width):
            candidate = operator(basis[:, column])
            total += 1
            for row in range(column + 1):
                projection = (basis[:, row] * candidate).sum(dim=-1)
                hessenberg[:, row, column] = projection
                candidate = candidate - projection[:, None] * basis[:, row]
            norm = candidate.norm(dim=-1)
            hessenberg[:, column + 1, column] = norm
            used = column + 1
            can_extend = active & (norm >= 1e-12)
            basis[:, column + 1] = torch.where(
                can_extend[:, None],
                candidate / norm.clamp_min(1e-12)[:, None],
                0.0,
            )
            if not bool(can_extend.any()):
                break

        target = torch.zeros(
            batch, used + 1, 1, dtype=work_dtype, device=rhs.device
        )
        target[:, 0, 0] = beta
        coefficients = torch.linalg.lstsq(
            hessenberg[:, : used + 1, :used], target
        ).solution.squeeze(-1)
        update = (basis[:, :used] * coefficients.unsqueeze(-1)).sum(dim=1)
        estimate = torch.where(active[:, None], estimate + update, estimate)

        residual = vector_rhs - operator(estimate)
        residual_norm = residual.norm(dim=-1)
        relative = torch.where(
            zero_rhs,
            torch.zeros_like(residual_norm),
            residual_norm / rhs_norm.clamp_min(1e-12),
        )
        done |= relative < tol

    output = estimate.to(original_dtype).reshape_as(rhs)
    if not bool(torch.isfinite(output).all()):
        return rhs, total, float("inf")
    return output, total, float(relative.max())


def _true_adjoint_residual(
    solution: torch.Tensor,
    rhs: torch.Tensor,
    vjp: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    residual = rhs - (solution - vjp(solution))
    relative = residual.flatten(1).norm(dim=1) / rhs.flatten(1).norm(
        dim=1
    ).clamp_min(1e-12)
    return float(relative.max())


def build_maze_diagnostic(
    device: torch.device,
    *,
    batch: int,
) -> dict[str, object]:
    torch.manual_seed(0)
    length, hidden, heads = 49, 128, 8
    relations = maze_relation_ids(7, 7, device=device)
    allowed = tuple(
        MAZE_RELATIONS[name]
        for name in ("self", "up", "down", "left", "right")
    )
    attention_bias = local_attention_bias(
        relations,
        num_heads=heads,
        allowed_relations=allowed,
        num_global_heads=2,
    ).to(device)
    model = build_prime_model(
        "fpsa_prime",
        vocab_size=4,
        out_vocab_size=2,
        max_seq_len=length,
        hidden_size=hidden,
        num_heads=heads,
        num_relation_types=len(MAZE_RELATIONS),
        forward_solver="picard",
        max_iter=64,
        max_iter_eval=64,
        fp_tol=1e-7,
        solver_damping=0.8,
        stability_weight=0.0,
        require_convergence=False,
        require_backward_convergence=False,
    ).to(device).eval()

    tokens = torch.randint(0, 4, (batch, length), device=device)
    targets = torch.randint(0, 2, (batch, length), device=device)
    score_mask = torch.ones_like(tokens, dtype=torch.bool)
    with torch.no_grad():
        output = model(
            tokens,
            relation_ids=relations,
            attention_bias=attention_bias,
            max_iter=64,
        )
    fixed = output["residual"].detach()
    encoded = output["encoded"].detach()
    residual_variable = fixed.clone().requires_grad_(True)
    logits = model.decode_residual(encoded, residual_variable)
    loss = masked_loss(logits, targets, score_mask)
    (rhs,) = torch.autograd.grad(loss, residual_variable)

    context = model._prepare_context(encoded, attention_bias, relations)
    state_variable = fixed.clone().requires_grad_(True)
    mapped = model.attention.fixed_map(state_variable, context)

    def vjp(vector: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            mapped, state_variable, vector, retain_graph=True
        )[0]

    return {
        "model": model,
        "relations": relations,
        "attention_bias": attention_bias,
        "tokens": tokens,
        "output": output,
        "rhs": rhs,
        "vjp": vjp,
    }


def gmres_check(
    output_dir: Path,
    device: torch.device,
    *,
    quick: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    diagnostic = build_maze_diagnostic(device, batch=4 if quick else 32)
    rhs = diagnostic["rhs"]
    vjp = diagnostic["vjp"]
    assert isinstance(rhs, torch.Tensor)
    assert callable(vjp)

    rows = []
    for budget in [4, 8, 12, 15, 20, 40]:
        for name, solver in (
            ("legacy v0", legacy_gmres_solve),
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
                    "true_residual": _true_adjoint_residual(
                        solution, rhs, vjp
                    ),
                }
            )
    table = pd.DataFrame(rows)
    fixed_at_40 = table[
        (table.solver == "fixed") & (table.budget == 40)
    ].iloc[0]
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
    if fixed_at_40.true_residual > 1.05e-5:
        raise AssertionError("fixed GMRES missed the adjoint tolerance")
    if fixed_at_40.iterations_used >= 15:
        raise AssertionError("fixed GMRES did not stop at the short solution")
    if legacy_15 <= 100 * legacy_12:
        raise AssertionError("seeded old-GMRES degradation was not reproduced")

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
    plt.title("Long-basis regression on an actual FPSA VJP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gmres_regression.png", dpi=160)
    plt.close()
    return table, diagnostic


def stability_check(
    output_dir: Path,
    diagnostic: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, float]]:
    device = diagnostic["tokens"].device
    gain = torch.tensor(0.70, device=device, requires_grad=True)
    linear_state = torch.randn(3, 4, 5, device=device)
    linear = local_jacobian_spectral_penalty(
        lambda value: gain * value,
        linear_state,
        target=0.60,
        power_steps=2,
        epsilon=1e-3,
    )
    linear["loss"].backward()
    if not torch.allclose(
        linear["estimate"],
        torch.full((3,), 0.70, device=device),
        atol=3e-4,
        rtol=3e-4,
    ):
        raise AssertionError("stability estimator failed the known linear map")
    if gain.grad is None or not bool(gain.grad > 0):
        raise AssertionError("stability penalty gradient has the wrong sign")

    model = diagnostic["model"]
    tokens = diagnostic["tokens"][:2]
    relations = diagnostic["relations"]
    bias = diagnostic["attention_bias"]
    with torch.no_grad():
        output = model(
            tokens,
            relation_ids=relations,
            attention_bias=bias,
            max_iter=64,
        )
    context = model._prepare_context(output["encoded"].detach(), bias, relations)
    context = model._detached_context(context)
    fixed_map = lambda state: model.attention.fixed_map(state, context)

    rows = []
    for power_steps in [1, 2, 4, 8]:
        result = local_jacobian_spectral_penalty(
            fixed_map,
            output["residual"].detach(),
            target=0.95,
            power_steps=power_steps,
            epsilon=1e-3,
        )
        rows.append(
            {
                "power_steps": power_steps,
                "mean_sigma": float(result["estimate"].detach().mean()),
                "max_sigma": float(result["estimate"].detach().max()),
                "penalty": float(result["loss"].detach()),
            }
        )
    table = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4.2))
    plt.plot(table.power_steps, table.mean_sigma, marker="o", label="mean")
    plt.plot(table.power_steps, table.max_sigma, marker="o", label="max")
    plt.axhline(0.95, linestyle="--", label="example target")
    plt.xlabel("Power iterations on J^T J")
    plt.ylabel("Estimated local spectral norm")
    plt.title("Stability-estimate convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stability_power_sweep.png", dpi=160)
    plt.close()
    linear_summary = {
        "known_sigma": 0.70,
        "estimated_sigma_mean": float(linear["estimate"].detach().mean()),
        "penalty": float(linear["loss"].detach()),
        "gradient_wrt_gain": float(gain.grad),
    }
    return table, linear_summary


def fixed_unroll_check(device: torch.device) -> dict[str, object]:
    torch.manual_seed(21)
    model = build_prime_model(
        "fpsa_prime_bptt",
        vocab_size=10,
        out_vocab_size=10,
        max_seq_len=9,
        hidden_size=24,
        num_heads=4,
        use_input_mlp=False,
        use_output_mlp=False,
        position_encoding="rope",
        max_iter=7,
        max_iter_eval=7,
        fp_tol=1e-12,
        require_convergence=True,
    ).to(device)
    inputs = torch.randint(0, 10, (2, 9), device=device)
    model.train()
    training = model(inputs)
    model.eval()
    with torch.no_grad():
        evaluation = model(inputs)
    residual_difference = float(
        (
            training["residual"].detach() - evaluation["residual"]
        ).abs().max()
    )
    logit_difference = float(
        (training["logits"].detach() - evaluation["logits"]).abs().max()
    )
    if training["info"].n_iters != evaluation["info"].n_iters:
        raise AssertionError("fixed-unroll train/eval iteration counts differ")
    if residual_difference != 0.0 or logit_difference != 0.0:
        raise AssertionError("fixed-unroll train/eval forward values differ")
    return {
        "forward_mode": model.cfg.forward_mode,
        "backward_mode": model.cfg.backward_mode,
        "train_iterations": training["info"].n_iters,
        "eval_iterations": evaluation["info"].n_iters,
        "max_residual_difference": residual_difference,
        "max_logit_difference": logit_difference,
    }


def _bfs_path(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    queue = deque([start])
    previous = {start: None}
    height, width = grid.shape
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = previous[current]
            return path[::-1]
        row, column = current
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = row + dr, column + dc
            if (
                0 <= neighbor[0] < height
                and 0 <= neighbor[1] < width
                and grid[neighbor] == 1
                and neighbor not in previous
            ):
                previous[neighbor] = current
                queue.append(neighbor)
    return None


def _make_tiny_maze_dataset(
    samples: int,
    *,
    size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    while len(inputs) < samples:
        grid = (rng.rand(size, size) > 0.28).astype(np.int64)
        free = np.argwhere(grid == 1)
        if len(free) < 4:
            continue
        selected = rng.choice(len(free), 2, replace=False)
        start, goal = tuple(free[selected[0]]), tuple(free[selected[1]])
        path = _bfs_path(grid, start, goal)
        if path is None or len(path) < 3:
            continue
        puzzle = grid.copy()
        puzzle[start], puzzle[goal] = 2, 3
        answer = np.zeros_like(grid)
        for cell in path:
            answer[cell] = 1
        inputs.append(puzzle.reshape(-1))
        targets.append(answer.reshape(-1))
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def tiny_training_smoke(
    output_dir: Path,
    device: torch.device,
    *,
    quick: bool,
) -> pd.DataFrame:
    torch.manual_seed(31)
    maze_size = 5
    train_x, train_y = _make_tiny_maze_dataset(
        256, size=maze_size, seed=31
    )
    test_x, test_y = _make_tiny_maze_dataset(
        64, size=maze_size, seed=3131
    )
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    relations = maze_relation_ids(maze_size, maze_size, device=device)
    allowed = tuple(
        MAZE_RELATIONS[name]
        for name in ("self", "up", "down", "left", "right")
    )
    bias = local_attention_bias(
        relations,
        num_heads=4,
        allowed_relations=allowed,
        num_global_heads=1,
    ).to(device)
    model = build_prime_model(
        "fpsa_prime",
        vocab_size=4,
        out_vocab_size=2,
        max_seq_len=maze_size * maze_size,
        hidden_size=48,
        num_heads=4,
        num_relation_types=len(MAZE_RELATIONS),
        forward_solver="anderson",
        max_iter=24,
        max_iter_eval=24,
        fp_tol=1e-4,
        solver_damping=0.8,
        backward_max_iter=40,
        backward_tol=1e-5,
        gmres_restart=20,
        stability_weight=0.0,
        require_convergence=True,
        require_backward_convergence=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-2
    )
    steps = 30 if quick or device.type == "cpu" else 80
    records = []
    for step in range(1, steps + 1):
        indices = torch.randint(0, len(train_x), (16,), device=device)
        x_batch, y_batch = train_x[indices], train_y[indices]
        score_mask = torch.ones_like(x_batch, dtype=torch.bool)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            x_batch, relation_ids=relations, attention_bias=bias
        )
        loss = masked_loss(output["logits"], y_batch, score_mask)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 5 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                evaluation = model(
                    test_x, relation_ids=relations, attention_bias=bias
                )
            prediction = evaluation["logits"].argmax(-1)
            records.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "token_accuracy": float(
                        (prediction == test_y).float().mean()
                    ),
                    "exact_match": float(
                        (prediction == test_y).all(dim=1).float().mean()
                    ),
                    "forward_iterations": evaluation["info"].n_iters,
                    "forward_residual": evaluation["info"].rel_residual,
                    "backward_iterations": BACKWARD_STATS["iters"],
                    "backward_residual": BACKWARD_STATS[
                        "relative_residual"
                    ],
                    "gradient_norm": float(gradient_norm),
                }
            )
    table = pd.DataFrame(records)
    numeric = table.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        raise AssertionError("tiny training produced a non-finite value")
    if table.backward_residual.max() > 1.05e-5:
        raise AssertionError("tiny training missed the backward tolerance")
    if table.forward_residual.max() > 1.05e-4:
        raise AssertionError("tiny training missed the forward tolerance")

    plt.figure(figsize=(8, 4.2))
    plt.plot(table.step, table.loss, marker="o")
    plt.xlabel("Training step")
    plt.ylabel("Stablemax loss")
    plt.title("Tiny implicit-training smoke test")
    plt.tight_layout()
    plt.savefig(output_dir / "tiny_training_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.2))
    plt.plot(
        table.step,
        table.forward_residual,
        marker="o",
        label="forward FP",
    )
    plt.plot(
        table.step,
        table.backward_residual,
        marker="o",
        label="adjoint",
    )
    plt.axhline(1e-4, linestyle="--", label="forward tolerance")
    plt.axhline(1e-5, linestyle=":", label="backward tolerance")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Relative residual")
    plt.title("Solver residuals during the smoke run")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tiny_training_residuals.png", dpi=160)
    plt.close()
    return table


def run_verification(
    *,
    output_dir: str | Path = "results/verification",
    device: str = "auto",
    quick: bool = True,
    run_tests: bool = True,
    run_training: bool = False,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected_device = resolve_device(device)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    test_output = run_repository_tests() if run_tests else "skipped"
    parameters, parameter_gap = parameter_check(destination)
    gmres, diagnostic = gmres_check(
        destination, selected_device, quick=quick
    )
    stability, linear_stability = stability_check(destination, diagnostic)
    fixed_unroll = fixed_unroll_check(selected_device)
    training = (
        tiny_training_smoke(destination, selected_device, quick=quick)
        if run_training
        else None
    )

    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    summary = {
        "environment": {
            "git_sha": git_sha,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(selected_device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if selected_device.type == "cuda"
                else None
            ),
        },
        "tests": test_output,
        "parameter_counts": parameters.to_dict(orient="records"),
        "hero_block_parameter_gap_fraction": parameter_gap,
        "gmres": gmres.to_dict(orient="records"),
        "linear_stability": linear_stability,
        "stability_power_sweep": stability.to_dict(orient="records"),
        "fixed_unroll": fixed_unroll,
        "tiny_training": (
            None if training is None else training.to_dict(orient="records")
        ),
        "plots": sorted(path.name for path in destination.glob("*.png")),
    }
    (destination / "fpsa_prime_verification_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="results/verification")
    p.add_argument("--device", default="auto")
    p.add_argument("--full", action="store_true")
    p.add_argument("--skip_tests", action="store_true")
    p.add_argument("--run_training", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    result = run_verification(
        output_dir=args.output_dir,
        device=args.device,
        quick=not args.full,
        run_tests=not args.skip_tests,
        run_training=args.run_training,
    )
    print(json.dumps(result, indent=2))

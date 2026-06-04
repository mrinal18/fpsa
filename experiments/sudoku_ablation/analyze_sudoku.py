#!/usr/bin/env python3
"""Analyze and visualize FPSA Value-stream ablation results on Sudoku.

Loads JSON result files for each variant from a results directory and
produces four publication-quality comparison plots:
  1. Board accuracy grouped bar chart
  2. FPI convergence curves (residual norms, log-scale)
  3. Effective rank evolution across FPI iterations
  4. Training curves (loss, cell acc, board acc, mean steps)

Usage:
    python analyze_sudoku.py --results-dir results/sudoku
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS: List[str] = ["fixed", "evolving", "blended", "fixed_ffn", "vanilla"]

COLOR_PALETTE: Dict[str, str] = {
    "fixed": "#e74c3c",
    "evolving": "#3498db",
    "blended": "#2ecc71",
    "fixed_ffn": "#9b59b6",
    "vanilla": "#95a5a6",
}

DPI: int = 150


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all available ``{variant}_results.json`` files.

    Parameters
    ----------
    results_dir : str
        Path to the directory containing result JSON files.

    Returns
    -------
    dict
        Mapping from variant name to its parsed JSON content.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        path = os.path.join(results_dir, f"{variant}_results.json")
        if os.path.isfile(path):
            with open(path, "r") as f:
                results[variant] = json.load(f)
    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _apply_style() -> None:
    """Set a clean, publication-quality matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        # Fallback for older matplotlib versions.
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")


def plot_board_accuracy(
    results: Dict[str, Dict[str, Any]], save_path: str
) -> None:
    """Grouped bar chart of final board accuracy per variant.

    Parameters
    ----------
    results : dict
        Loaded results keyed by variant name.
    save_path : str
        Destination PNG path.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    variants = [v for v in VARIANTS if v in results]
    accuracies = [
        results[v]["eval_metrics"]["board_accuracy"] * 100 for v in variants
    ]
    colors = [COLOR_PALETTE[v] for v in variants]

    x = np.arange(len(variants))
    bars = ax.bar(x, accuracies, color=colors, width=0.6, edgecolor="white")

    # Annotate bars with the accuracy value.
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=11)
    ax.set_ylabel("Board Accuracy (%)", fontsize=12)
    ax.set_title("Board Accuracy by Value-Stream Variant (Sudoku)", fontsize=14)
    ax.set_ylim(0, min(max(accuracies) + 10, 105))

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_convergence_curves(
    results: Dict[str, Dict[str, Any]], save_path: str
) -> None:
    """Line plot of mean residual norm vs FPI iteration (log-scale Y).

    Residual norms are averaged across layers for each iteration step.

    Parameters
    ----------
    results : dict
        Loaded results keyed by variant name.
    save_path : str
        Destination PNG path.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for variant in VARIANTS:
        if variant not in results:
            continue
        residual_norms = results[variant]["convergence_trace"]["residual_norms"]
        # residual_norms: list[list[float]] — per layer, norms per iteration
        # Average across layers for each iteration index.
        norms_array = np.array(residual_norms)  # (num_layers, num_iters)
        mean_norms = norms_array.mean(axis=0)  # (num_iters,)
        iterations = np.arange(1, len(mean_norms) + 1)

        ax.plot(
            iterations,
            mean_norms,
            label=variant,
            color=COLOR_PALETTE[variant],
            linewidth=2,
            marker="o",
            markersize=4,
        )

    ax.set_yscale("log")
    ax.set_xlabel("FPI Iteration $k$", fontsize=12)
    ax.set_ylabel("Mean Residual Norm", fontsize=12)
    ax.set_title("FPI Convergence Curves", fontsize=14)
    ax.legend(fontsize=10, frameon=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_rank_evolution(
    results: Dict[str, Dict[str, Any]], save_path: str
) -> None:
    """Line plot of effective rank vs FPI iteration (averaged across layers).

    Parameters
    ----------
    results : dict
        Loaded results keyed by variant name.
    save_path : str
        Destination PNG path.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for variant in VARIANTS:
        if variant not in results:
            continue
        effective_ranks = results[variant]["convergence_trace"]["effective_ranks"]
        # effective_ranks: list[list[float]] — per layer, ranks per iteration
        ranks_array = np.array(effective_ranks)  # (num_layers, num_iters)
        mean_ranks = ranks_array.mean(axis=0)  # (num_iters,)
        iterations = np.arange(1, len(mean_ranks) + 1)

        ax.plot(
            iterations,
            mean_ranks,
            label=variant,
            color=COLOR_PALETTE[variant],
            linewidth=2,
            marker="s",
            markersize=4,
        )

    ax.set_xlabel("FPI Iteration $k$", fontsize=12)
    ax.set_ylabel("Effective Rank", fontsize=12)
    ax.set_title("Effective Rank vs Iteration", fontsize=14)
    ax.legend(fontsize=10, frameon=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curves(
    results: Dict[str, Dict[str, Any]], save_path: str
) -> None:
    """2×2 subplot grid of training metrics over epochs.

    Subplots: loss, cell accuracy, board accuracy, mean steps.

    Parameters
    ----------
    results : dict
        Loaded results keyed by variant name.
    save_path : str
        Destination PNG path.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    metric_keys = ["loss", "cell_accuracy", "board_accuracy", "mean_steps"]
    titles = ["Training Loss", "Cell Accuracy", "Board Accuracy", "Mean Steps"]
    ylabels = ["Loss", "Accuracy", "Accuracy", "Steps"]

    for ax, key, title, ylabel in zip(axes.flat, metric_keys, titles, ylabels):
        for variant in VARIANTS:
            if variant not in results:
                continue
            history = results[variant]["train_history"]
            if key not in history:
                continue
            values = history[key]
            epochs = np.arange(1, len(values) + 1)
            ax.plot(
                epochs,
                values,
                label=variant,
                color=COLOR_PALETTE[variant],
                linewidth=1.5,
            )
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a formatted comparison table of final eval metrics to stdout.

    Parameters
    ----------
    results : dict
        Loaded results keyed by variant name.
    """
    header = f"{'Variant':<12} {'Cell Acc (%)':<14} {'Board Acc (%)':<15} {'Mean Steps':<12}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("  FPSA Value-Stream Ablation — Sudoku Results")
    print(sep)
    print(header)
    print(sep)

    for variant in VARIANTS:
        if variant not in results:
            continue
        em = results[variant]["eval_metrics"]
        cell_acc = em["cell_accuracy"] * 100
        board_acc = em["board_accuracy"] * 100
        mean_steps = em["mean_steps"]
        print(f"{variant:<12} {cell_acc:<14.2f} {board_acc:<15.2f} {mean_steps:<12.2f}")

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(results_dir: str) -> None:
    """Entry-point: load results, generate plots, print summary.

    Parameters
    ----------
    results_dir : str
        Path to the directory with ``{variant}_results.json`` files.
    """
    if not os.path.isdir(results_dir):
        print("No results found. Run train_sudoku.py first.")
        return

    results = load_results(results_dir)

    if not results:
        print("No results found. Run train_sudoku.py first.")
        return

    available = list(results.keys())
    print(f"Loaded results for variants: {available}")

    # Generate all plots -------------------------------------------------- #
    print("\nGenerating plots …")

    plot_board_accuracy(
        results,
        os.path.join(results_dir, "board_accuracy_comparison.png"),
    )

    # Only generate convergence / rank plots if trace data is present.
    has_traces = any("convergence_trace" in results[v] for v in results)
    if has_traces:
        plot_convergence_curves(
            results,
            os.path.join(results_dir, "convergence_curves.png"),
        )
        plot_rank_evolution(
            results,
            os.path.join(results_dir, "rank_evolution.png"),
        )
    else:
        print("  (Skipping convergence / rank plots — no trace data found.)")

    plot_training_curves(
        results,
        os.path.join(results_dir, "training_curves.png"),
    )

    # Print comparison table ---------------------------------------------- #
    print_comparison_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze FPSA Value-stream ablation results on Sudoku.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/sudoku",
        help="Directory containing {variant}_results.json files "
        "(default: results/sudoku).",
    )
    args = parser.parse_args()
    main(args.results_dir)

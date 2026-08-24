"""Verifier energies and attractor-shaping losses."""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .solvers import token_fixed_point_residual


def sudoku_energy(
    logits: torch.Tensor,
    puzzle: torch.Tensor,
    *,
    clue_weight: float = 1.0,
    constraint_weight: float = 1.0,
    entropy_weight: float = 0.05,
) -> torch.Tensor:
    """Differentiable Sudoku violation energy, one value per sample.

    Token ``0`` denotes a blank and digits are tokens ``1..9``.  The output may
    contain a blank logit, but the verifier renormalizes over the nine digits.
    """
    if logits.ndim != 3 or logits.shape[1] != 81 or logits.shape[-1] < 10:
        raise ValueError("Sudoku logits must have shape (B, 81, V>=10)")
    if puzzle.shape != logits.shape[:2]:
        raise ValueError("puzzle must have shape (B, 81)")
    probabilities = F.softmax(logits[..., 1:10].float(), dim=-1)
    grid = probabilities.view(-1, 9, 9, 9)

    row_count = grid.sum(dim=2)
    column_count = grid.sum(dim=1)
    boxes = grid.view(-1, 3, 3, 3, 3, 9).permute(0, 1, 3, 2, 4, 5)
    box_count = boxes.reshape(-1, 9, 9, 9).sum(dim=2)
    constraint = (
        (row_count - 1.0).square().mean(dim=(1, 2))
        + (column_count - 1.0).square().mean(dim=(1, 2))
        + (box_count - 1.0).square().mean(dim=(1, 2))
    )

    clues = puzzle > 0
    clue_digit = (puzzle - 1).clamp_min(0)
    clue_prob = probabilities.gather(-1, clue_digit.unsqueeze(-1)).squeeze(-1)
    clue_nll = -clue_prob.clamp_min(1e-8).log()
    clue_term = (clue_nll * clues).sum(dim=1) / clues.sum(dim=1).clamp_min(1)
    entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(-1).mean(-1)
    return clue_weight * clue_term + constraint_weight * constraint + entropy_weight * entropy


def sudoku_discrete_violations(prediction: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
    """Exact integer violation count for selecting particles at evaluation."""
    if prediction.shape != puzzle.shape or prediction.shape[-1] != 81:
        raise ValueError("prediction and puzzle must have shape (B, 81)")
    grid = prediction.view(-1, 9, 9)
    clue_mismatch = ((puzzle > 0) & (prediction != puzzle)).sum(dim=1)

    def unit_violations(units: torch.Tensor) -> torch.Tensor:
        # A valid unit contains each digit 1..9 exactly once.
        one_hot = F.one_hot(units.clamp(0, 9), num_classes=10)[..., 1:10]
        return (one_hot.sum(dim=-2) != 1).sum(dim=-1)

    rows = unit_violations(grid).sum(dim=1)
    columns = unit_violations(grid.transpose(1, 2)).sum(dim=1)
    boxes = grid.view(-1, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    box = unit_violations(boxes).sum(dim=1)
    invalid_digit = ((prediction < 1) | (prediction > 9)).sum(dim=1)
    return clue_mismatch + rows + columns + box + invalid_digit


def attractor_margin_loss(
    fixed_map: Callable[[torch.Tensor], torch.Tensor],
    residual: torch.Tensor,
    energy: torch.Tensor,
    *,
    positive_threshold: float,
    negative_threshold: float,
    negative_margin: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Make low-energy states fixed and repel high-energy states from fixedness.

    ``residual`` is normally detached before this function.  The loss then
    shapes the local vector field without backpropagating through how that state
    was discovered.
    """
    mapped = fixed_map(residual)
    fp_residual = token_fixed_point_residual(mapped, residual).mean(dim=1)
    positive = energy.detach() <= positive_threshold
    negative = energy.detach() >= negative_threshold
    zero = fp_residual.new_zeros(())
    positive_loss = fp_residual[positive].square().mean() if bool(positive.any()) else zero
    negative_loss = (
        F.relu(negative_margin - fp_residual[negative]).square().mean()
        if bool(negative.any())
        else zero
    )
    return {
        "loss": positive_loss + negative_loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "fixed_point_residual": fp_residual.detach(),
    }

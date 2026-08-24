"""Shared task adapters and evaluation helpers for FPSA-Prime experiments."""

from __future__ import annotations

import math
from typing import Optional

import torch

from src.fpsa_prime.structures import (
    MAZE_RELATIONS,
    SUDOKU_RELATIONS,
    local_attention_bias,
    maze_relation_ids,
    prepend_global_attention_bias,
    prepend_global_slots,
    sudoku_relation_ids,
)


def stablemax(logits: torch.Tensor) -> torch.Tensor:
    values = torch.where(logits >= 0, logits + 1.0, 1.0 / (1.0 - logits))
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def masked_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    log_probability = stablemax(logits.float()).clamp_min(1e-12).log()
    negative_log_likelihood = -log_probability.gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    weights = mask.float()
    return (negative_log_likelihood * weights).sum() / weights.sum().clamp_min(1.0)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def task_structure(
    task,
    *,
    heads: int,
    global_heads: int,
    global_slots: int,
    device: torch.device,
):
    if task.name == "sudoku":
        relations = sudoku_relation_ids(device=device)
        allowed = (
            SUDOKU_RELATIONS["self"],
            SUDOKU_RELATIONS["row"],
            SUDOKU_RELATIONS["column"],
            SUDOKU_RELATIONS["box"],
            SUDOKU_RELATIONS["row_box"],
            SUDOKU_RELATIONS["column_box"],
        )
        bias = local_attention_bias(
            relations,
            num_heads=heads,
            allowed_relations=allowed,
            num_global_heads=global_heads,
        )
        if global_slots:
            relations = prepend_global_slots(
                relations,
                global_slots,
                slot_relation=SUDOKU_RELATIONS["slot"],
            )
            bias = prepend_global_attention_bias(bias, global_slots)
        return relations, bias, len(SUDOKU_RELATIONS)

    if task.name == "maze":
        side = math.isqrt(task.seq_len)
        if side * side != task.seq_len:
            raise ValueError("the current Maze task adapter expects a square grid")
        relations = maze_relation_ids(side, side, device=device)
        allowed = (
            MAZE_RELATIONS["self"],
            MAZE_RELATIONS["up"],
            MAZE_RELATIONS["down"],
            MAZE_RELATIONS["left"],
            MAZE_RELATIONS["right"],
        )
        bias = local_attention_bias(
            relations,
            num_heads=heads,
            allowed_relations=allowed,
            num_global_heads=global_heads,
        )
        if global_slots:
            relations = prepend_global_slots(
                relations,
                global_slots,
                slot_relation=MAZE_RELATIONS["slot"],
            )
            bias = prepend_global_attention_bias(bias, global_slots)
        return relations, bias, len(MAZE_RELATIONS)

    return None, None, 0


@torch.no_grad()
def evaluate(
    model,
    loader,
    *,
    device: torch.device,
    relation_ids,
    attention_bias,
    max_iter: Optional[int] = None,
):
    model.eval()
    exact = total = token_ok = token_total = 0
    residual = iterations = convergence = 0.0
    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        output = model(
            x,
            relation_ids=relation_ids,
            attention_bias=attention_bias,
            max_iter=max_iter,
        )
        logits = output["logits"]
        assert isinstance(logits, torch.Tensor)
        prediction = logits.argmax(-1)
        correct = (prediction == y) | ~mask
        exact += int(correct.all(dim=1).sum())
        total += x.shape[0]
        token_ok += int((correct & mask).sum())
        token_total += int(mask.sum())
        info = output["info"]
        residual += info.rel_residual
        iterations += info.n_iters
        convergence += info.converged_frac
    batches = max(1, len(loader))
    return {
        "exact_match": 100.0 * exact / max(1, total),
        "token_accuracy": 100.0 * token_ok / max(1, token_total),
        "mean_residual": residual / batches,
        "mean_iterations": iterations / batches,
        "mean_converged_fraction": convergence / batches,
    }

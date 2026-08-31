"""Structural relation builders for grid reasoning tasks.

The relation tensor is indexed as ``relation[query, key]``.  Directional Maze
relations therefore describe where the key lies relative to the query.
"""

from __future__ import annotations

from typing import Iterable

import torch


SUDOKU_RELATIONS = {
    "other": 0,
    "self": 1,
    "row": 2,
    "column": 3,
    "box": 4,
    "row_box": 5,
    "column_box": 6,
    "slot": 7,
}

MAZE_RELATIONS = {
    "other": 0,
    "self": 1,
    "up": 2,
    "down": 3,
    "left": 4,
    "right": 5,
    "slot": 6,
}


def sudoku_relation_ids(
    size: int = 9,
    box_size: int = 3,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return pairwise Sudoku relations, preserving overlapping box relations."""
    if size <= 0 or box_size <= 0 or size % box_size:
        raise ValueError("size must be positive and divisible by box_size")
    positions = torch.arange(size * size, device=device)
    rows = positions // size
    columns = positions % size
    same_row = rows[:, None] == rows[None, :]
    same_column = columns[:, None] == columns[None, :]
    same_box = (
        (rows[:, None] // box_size == rows[None, :] // box_size)
        & (columns[:, None] // box_size == columns[None, :] // box_size)
    )
    same_cell = positions[:, None] == positions[None, :]

    relation = torch.full(
        (size * size, size * size),
        SUDOKU_RELATIONS["other"],
        dtype=torch.long,
        device=device,
    )
    relation = torch.where(
        same_box,
        torch.tensor(SUDOKU_RELATIONS["box"], device=device),
        relation,
    )
    relation = torch.where(
        same_row & ~same_box,
        torch.tensor(SUDOKU_RELATIONS["row"], device=device),
        relation,
    )
    relation = torch.where(
        same_column & ~same_box,
        torch.tensor(SUDOKU_RELATIONS["column"], device=device),
        relation,
    )
    relation = torch.where(
        same_row & same_box & ~same_cell,
        torch.tensor(SUDOKU_RELATIONS["row_box"], device=device),
        relation,
    )
    relation = torch.where(
        same_column & same_box & ~same_cell,
        torch.tensor(SUDOKU_RELATIONS["column_box"], device=device),
        relation,
    )
    relation = torch.where(
        same_cell,
        torch.tensor(SUDOKU_RELATIONS["self"], device=device),
        relation,
    )
    return relation


def maze_relation_ids(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return self and four directional neighbor relations for a grid."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    positions = torch.arange(height * width, device=device)
    query_row = (positions // width)[:, None]
    query_column = (positions % width)[:, None]
    key_row = (positions // width)[None, :]
    key_column = (positions % width)[None, :]

    relation = torch.full(
        (height * width, height * width),
        MAZE_RELATIONS["other"],
        dtype=torch.long,
        device=device,
    )
    relation = torch.where(
        (key_row == query_row - 1) & (key_column == query_column),
        torch.tensor(MAZE_RELATIONS["up"], device=device),
        relation,
    )
    relation = torch.where(
        (key_row == query_row + 1) & (key_column == query_column),
        torch.tensor(MAZE_RELATIONS["down"], device=device),
        relation,
    )
    relation = torch.where(
        (key_row == query_row) & (key_column == query_column - 1),
        torch.tensor(MAZE_RELATIONS["left"], device=device),
        relation,
    )
    relation = torch.where(
        (key_row == query_row) & (key_column == query_column + 1),
        torch.tensor(MAZE_RELATIONS["right"], device=device),
        relation,
    )
    relation.fill_diagonal_(MAZE_RELATIONS["self"])
    return relation


def local_attention_bias(
    relation_ids: torch.Tensor,
    *,
    num_heads: int,
    allowed_relations: Iterable[int],
    num_global_heads: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Mask local heads and leave the final global heads unrestricted.

    The result has shape ``(1, H, N, N)`` and lives on the same device as the
    relation tensor.  Every local query must retain at least one key, normally
    itself, or the softmax would be undefined.
    """
    if relation_ids.ndim != 2 or relation_ids.shape[0] != relation_ids.shape[1]:
        raise ValueError("relation_ids must be a square (N, N) tensor")
    if num_heads <= 0 or not 0 <= num_global_heads <= num_heads:
        raise ValueError("invalid head counts")
    allowed = torch.zeros_like(relation_ids, dtype=torch.bool)
    for relation in allowed_relations:
        allowed |= relation_ids == int(relation)
    if not bool(allowed.diagonal().all()):
        raise ValueError("every token must be allowed to attend to itself")
    length = relation_ids.shape[0]
    bias = torch.zeros(
        1,
        num_heads,
        length,
        length,
        dtype=dtype,
        device=relation_ids.device,
    )
    local_heads = num_heads - num_global_heads
    if local_heads:
        local = torch.zeros(
            length, length, dtype=dtype, device=relation_ids.device
        )
        local = local.masked_fill(~allowed, float("-inf"))
        bias[:, :local_heads] = local
    return bias


def prepend_global_slots(
    relation_ids: torch.Tensor,
    num_slots: int,
    *,
    slot_relation: int,
) -> torch.Tensor:
    """Pad a 2-D or batched 3-D relation tensor with slot interactions."""
    if num_slots < 0:
        raise ValueError("num_slots cannot be negative")
    if relation_ids.ndim not in (2, 3):
        raise ValueError("relation_ids must have shape (N,N) or (B,N,N)")
    if relation_ids.shape[-2] != relation_ids.shape[-1]:
        raise ValueError("relation_ids must be square")
    if num_slots == 0:
        return relation_ids
    length = relation_ids.shape[-1]
    leading = relation_ids.shape[:-2]
    output = torch.full(
        (*leading, length + num_slots, length + num_slots),
        int(slot_relation),
        dtype=relation_ids.dtype,
        device=relation_ids.device,
    )
    output[..., num_slots:, num_slots:] = relation_ids
    return output


def prepend_global_attention_bias(
    attention_bias: torch.Tensor,
    num_slots: int,
) -> torch.Tensor:
    """Pad an additive attention bias while leaving slot edges unrestricted."""
    if num_slots < 0:
        raise ValueError("num_slots cannot be negative")
    if attention_bias.ndim not in (2, 3, 4):
        raise ValueError("attention_bias must have 2, 3, or 4 dimensions")
    if attention_bias.shape[-2] != attention_bias.shape[-1]:
        raise ValueError("attention_bias must be square in its last two dimensions")
    if num_slots == 0:
        return attention_bias
    length = attention_bias.shape[-1]
    output = torch.zeros(
        *attention_bias.shape[:-2],
        length + num_slots,
        length + num_slots,
        dtype=attention_bias.dtype,
        device=attention_bias.device,
    )
    output[..., num_slots:, num_slots:] = attention_bias
    return output

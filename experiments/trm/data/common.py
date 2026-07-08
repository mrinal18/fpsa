"""Shared dataset format for the TRM experiments.

Datasets are directories with train/ and test/ subdirs, each containing
    dataset.json          (metadata)
    all__inputs.npy       (num_examples, seq_len) uint8/int32
    all__labels.npy       (num_examples, seq_len)
    all__group_indices.npy(num_groups+1,) — examples [g[i], g[i+1]) are
                          augmented variants of one original puzzle
Identical to the TRM reference layout (minus puzzle_identifiers, which are
constant for Sudoku/Maze), so datasets built with the reference repo also load.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np


@dataclass
class PuzzleDatasetMetadata:
    pad_id: int
    ignore_label_id: Optional[int]
    vocab_size: int
    seq_len: int
    total_groups: int
    sets: List[str]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotation, flip, and mirror."""
    if tid == 1:
        return np.rot90(arr, k=1)
    if tid == 2:
        return np.rot90(arr, k=2)
    if tid == 3:
        return np.rot90(arr, k=3)
    if tid == 4:
        return np.fliplr(arr)
    if tid == 5:
        return np.flipud(arr)
    if tid == 6:
        return arr.T
    if tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    return arr


def save_split(output_dir: str, set_name: str, inputs: np.ndarray, labels: np.ndarray,
               group_indices: np.ndarray, metadata: PuzzleDatasetMetadata) -> None:
    save_dir = os.path.join(output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(asdict(metadata), f)
    np.save(os.path.join(save_dir, "all__inputs.npy"), inputs)
    np.save(os.path.join(save_dir, "all__labels.npy"), labels)
    np.save(os.path.join(save_dir, "all__group_indices.npy"), group_indices)


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    """Validity-preserving Sudoku augmentation: random digit relabeling,
    optional transpose, band/stack and row/column permutations within bands.
    (Port of the TRM reference augmentation.)"""
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    transpose_flag = np.random.rand() < 0.5

    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)

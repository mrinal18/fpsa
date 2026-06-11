"""Sudoku-Extreme data: exact HRM/TRM protocol.

PROVENANCE: `shuffle_sudoku` and the encoding below are line-faithful ports of
SamsungSAILMontreal/TinyRecursiveModels/dataset/build_sudoku_dataset.py
(functionally identical to sapientinc/HRM's builder; verified by diff —
only whitespace + one metadata field differ).

Protocol facts (from their code, not from memory):
- CSV rows: source, q, a, rating; q/a are 81-char strings, '.' = blank.
- Canonical config: --subsample-size 1000 --num-aug 1000 (train);
  test set is NEVER augmented or subsampled.
- Encoding: cell value v in 0..9 stored as v+1 (so 1..10); pad_id = 0;
  vocab_size = 11; seq_len = 81. Loss uses ignore_label_id = 0.
- Metric: exact-match accuracy over all 81 cells, on the full test split.

ZERO-DRIFT PATH (preferred): run THEIR builder verbatim on the A100 box:
    python dataset/build_sudoku_dataset.py \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 --num-aug 1000
then point `load_trm_dataset` at the output dir. This module can also
rebuild equivalently (`build_dataset`, requires huggingface_hub) and can
generate synthetic puzzles for CPU unit tests.

NOTE faithfully reproduced: the upstream builder does NOT seed numpy, so
their dataset differs per build. `shuffle_sudoku(rng=None)` matches that
behavior; pass an rng for determinism in our own builds/tests.
"""
import os
from typing import Optional, Tuple

import numpy as np
import torch

SEQ_LEN = 81
VOCAB_SIZE = 11  # PAD + values 0..9 (stored +1)
PAD_ID = 0
IGNORE_LABEL_ID = 0


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray, rng=None):
    """Validity-preserving augmentation — exact port of TRM/HRM.

    digit permutation (blank 0 fixed) + optional transpose + band/row and
    stack/column permutations.
    """
    R = np.random if rng is None else rng
    digit_map = np.pad(R.permutation(np.arange(1, 10)), (1, 0))
    transpose_flag = R.rand() < 0.5 if rng is None else (R.random() < 0.5)

    bands = R.permutation(3)
    row_perm = np.concatenate([b * 3 + R.permutation(3) for b in bands])
    stacks = R.permutation(3)
    col_perm = np.concatenate([s * 3 + R.permutation(3) for s in stacks])

    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


# ---------------- loading their builder's output (zero drift) ----------------
def load_trm_dataset(output_dir: str, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load `all__inputs.npy` / `all__labels.npy` written by the HRM/TRM
    builder. Values are 1..10 (= cell value + 1). Returns int64 tensors
    of shape (N, 81)."""
    d = os.path.join(output_dir, split)
    inputs = np.load(os.path.join(d, "all__inputs.npy"))
    labels = np.load(os.path.join(d, "all__labels.npy"))
    assert inputs.shape[1] == SEQ_LEN and labels.shape[1] == SEQ_LEN
    assert inputs.min() >= 1 and inputs.max() <= 10
    assert labels.min() >= 2 and labels.max() <= 10  # solutions have no blanks
    return torch.from_numpy(inputs.astype(np.int64)), torch.from_numpy(labels.astype(np.int64))


# ---------------- equivalent rebuild (requires huggingface_hub) ----------------
def build_dataset(output_dir: str, subsample_size: Optional[int] = 1000,
                  num_aug: int = 1000, seed: Optional[int] = None,
                  source_repo: str = "sapientinc/sudoku-extreme"):
    """Replicates convert_subset for inputs/labels (the arrays we consume).
    Use their builder when possible; this exists for environments where
    cloning their repo is awkward. seed=None reproduces their unseeded
    behavior."""
    import csv
    from huggingface_hub import hf_hub_download

    rng = np.random.default_rng(seed) if seed is not None else None
    R = np.random if rng is None else rng

    for set_name in ["train", "test"]:
        inputs, labels = [], []
        with open(hf_hub_download(source_repo, f"{set_name}.csv",
                                  repo_type="dataset"), newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for source, q, a, rating in reader:
                assert len(q) == 81 and len(a) == 81
                inputs.append(np.frombuffer(q.replace('.', '0').encode(),
                                            dtype=np.uint8).reshape(9, 9) - ord('0'))
                labels.append(np.frombuffer(a.encode(),
                                            dtype=np.uint8).reshape(9, 9) - ord('0'))
        if set_name == "train" and subsample_size is not None and subsample_size < len(inputs):
            idx = (R.choice(len(inputs), size=subsample_size, replace=False)
                   if rng is None else rng.choice(len(inputs), size=subsample_size, replace=False))
            inputs = [inputs[i] for i in idx]
            labels = [labels[i] for i in idx]

        n_aug = num_aug if set_name == "train" else 0
        out_i, out_l = [], []
        for inp, lab in zip(inputs, labels):
            for a_idx in range(1 + n_aug):
                i2, l2 = (inp, lab) if a_idx == 0 else shuffle_sudoku(inp, lab, rng)
                out_i.append(i2)
                out_l.append(l2)
        arr_i = np.stack(out_i).reshape(len(out_i), -1).astype(np.int64) + 1
        arr_l = np.stack(out_l).reshape(len(out_l), -1).astype(np.int64) + 1
        d = os.path.join(output_dir, set_name)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "all__inputs.npy"), arr_i)
        np.save(os.path.join(d, "all__labels.npy"), arr_l)


# ---------------- synthetic puzzles for CPU unit tests ----------------
_BASE = np.array([[(i * 3 + i // 3 + j) % 9 + 1 for j in range(9)]
                  for i in range(9)], dtype=np.int64)


def synthetic_sudoku(n: int, n_givens: int = 30, seed: int = 0):
    """Valid (solution, puzzle) pairs for tests. Solutions are derived from
    the cyclic base grid via the official augmentation (validity-preserving
    by the property under test elsewhere — validity of outputs is asserted
    independently in tests via is_valid_solution)."""
    rng = np.random.default_rng(seed)
    puzzles, solutions = [], []
    for _ in range(n):
        _, sol = shuffle_sudoku(_BASE.copy(), _BASE.copy(), rng)
        mask = np.zeros(81, dtype=bool)
        mask[rng.choice(81, size=n_givens, replace=False)] = True
        puz = np.where(mask.reshape(9, 9), sol, 0)
        puzzles.append(puz)
        solutions.append(sol)
    return np.stack(puzzles), np.stack(solutions)


def is_valid_solution(grid: np.ndarray) -> bool:
    target = set(range(1, 10))
    for i in range(9):
        if set(grid[i, :]) != target or set(grid[:, i]) != target:
            return False
    for bi in range(3):
        for bj in range(3):
            if set(grid[bi*3:bi*3+3, bj*3:bj*3+3].flatten()) != target:
                return False
    return True


def is_consistent(puzzle: np.ndarray, solution: np.ndarray) -> bool:
    given = puzzle > 0
    return bool(np.all(solution[given] == puzzle[given]))


def encode(arr2d: np.ndarray) -> torch.Tensor:
    """(N,9,9) values 0..9 -> (N,81) tokens 1..10 (TRM encoding)."""
    return torch.from_numpy(arr2d.reshape(len(arr2d), -1).astype(np.int64) + 1)

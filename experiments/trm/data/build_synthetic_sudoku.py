"""Build a synthetic Sudoku dataset in the shared format — offline fallback.

Generates solved boards by randomized backtracking and masks cells. Useful
for smoke tests and environments without Hugging Face access. These puzzles
are much easier than Sudoku-Extreme (random masking, not minimal-clue), so
numbers are NOT comparable to the benchmark — use build_sudoku_extreme.py for
real runs.

Usage:
    python build_synthetic_sudoku.py --output-dir data/sudoku-synthetic-small \
        --num-train 500 --num-test 100 --n-remove 45 --num-aug 0
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PuzzleDatasetMetadata, save_split, shuffle_sudoku


def _is_valid(board, row, col, num):
    if num in board[row, :] or num in board[:, col]:
        return False
    br, bc = 3 * (row // 3), 3 * (col // 3)
    return num not in board[br:br + 3, bc:bc + 3]


def _solve(board, rng):
    empty = np.argwhere(board == 0)
    if len(empty) == 0:
        return True
    row, col = empty[0]
    candidates = np.arange(1, 10)
    rng.shuffle(candidates)
    for num in candidates:
        if _is_valid(board, row, col, num):
            board[row, col] = num
            if _solve(board, rng):
                return True
            board[row, col] = 0
    return False


def generate_solved_board(rng):
    board = np.zeros((9, 9), dtype=int)
    for i in range(3):
        board[3 * i:3 * i + 3, 3 * i:3 * i + 3] = rng.permutation(np.arange(1, 10)).reshape(3, 3)
    _solve(board, rng)
    return board


def build_split(set_name, n_puzzles, n_remove, num_aug, rng, output_dir):
    out_inputs, out_labels, group_indices = [], [], [0]
    for _ in range(n_puzzles):
        solution = generate_solved_board(rng)
        puzzle = solution.copy()
        holes = rng.choice(81, size=n_remove, replace=False)
        puzzle.reshape(-1)[holes] = 0

        n_aug = num_aug if set_name == "train" else 0
        for aug_idx in range(1 + n_aug):
            inp, out = (puzzle, solution) if aug_idx == 0 else shuffle_sudoku(puzzle, solution)
            out_inputs.append(inp.reshape(-1))
            out_labels.append(out.reshape(-1))
        group_indices.append(len(out_inputs))

    inputs_arr = np.stack(out_inputs).astype(np.uint8) + 1
    labels_arr = np.stack(out_labels).astype(np.uint8) + 1

    metadata = PuzzleDatasetMetadata(
        pad_id=0, ignore_label_id=0, vocab_size=11, seq_len=81,
        total_groups=len(group_indices) - 1, sets=["all"],
    )
    save_split(output_dir, set_name, inputs_arr, labels_arr,
               np.array(group_indices, dtype=np.int64), metadata)
    print(f"{set_name}: {len(inputs_arr)} examples -> {output_dir}/{set_name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="data/sudoku-synthetic-small")
    p.add_argument("--num-train", type=int, default=500)
    p.add_argument("--num-test", type=int, default=100)
    p.add_argument("--n-remove", type=int, default=45)
    p.add_argument("--num-aug", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    build_split("train", args.num_train, args.n_remove, args.num_aug, rng, args.output_dir)
    build_split("test", args.num_test, args.n_remove, 0, rng, args.output_dir)


if __name__ == "__main__":
    main()

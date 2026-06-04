"""
Sudoku puzzle generator and PyTorch Dataset for spatial reasoning models.

Generates valid 9x9 Sudoku boards from scratch using backtracking,
then creates masked puzzles suitable for training transformer-based solvers.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Board generation utilities
# ---------------------------------------------------------------------------

def _is_valid(board: np.ndarray, row: int, col: int, num: int) -> bool:
    """Check whether placing `num` at (row, col) violates Sudoku constraints."""
    # Row check
    if num in board[row, :]:
        return False
    # Column check
    if num in board[:, col]:
        return False
    # 3x3 box check
    box_r, box_c = 3 * (row // 3), 3 * (col // 3)
    if num in board[box_r:box_r + 3, box_c:box_c + 3]:
        return False
    return True


def _solve(board: np.ndarray, rng: np.random.RandomState) -> bool:
    """
    Solve the board in-place via backtracking with randomised candidate order.

    Parameters
    ----------
    board : np.ndarray, shape (9, 9)
        Partially filled board (0 = empty).
    rng : np.random.RandomState
        Random state used to shuffle candidate digits so that different
        calls produce different valid completions.

    Returns
    -------
    bool
        True if a valid completion was found, False otherwise.
    """
    # Find next empty cell
    empty = np.argwhere(board == 0)
    if len(empty) == 0:
        return True  # solved
    row, col = empty[0]

    candidates = np.arange(1, 10)
    rng.shuffle(candidates)

    for num in candidates:
        if _is_valid(board, row, col, num):
            board[row, col] = num
            if _solve(board, rng):
                return True
            board[row, col] = 0  # backtrack

    return False


def generate_solved_board(rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Generate a fully valid solved 9x9 Sudoku board.

    Strategy
    --------
    1. Fill the three main-diagonal 3x3 boxes independently (they share no
       row or column constraints with each other), using random permutations
       of 1-9.
    2. Solve the remaining cells via randomised backtracking.

    Parameters
    ----------
    rng : np.random.RandomState, optional
        Random state for reproducibility.  A new one is created if *None*.

    Returns
    -------
    np.ndarray, shape (9, 9), dtype int
        Completed Sudoku board with values 1-9.
    """
    if rng is None:
        rng = np.random.RandomState()

    board = np.zeros((9, 9), dtype=int)

    # Fill the three diagonal 3x3 boxes (indices 0, 1, 2 along the diagonal)
    for i in range(3):
        perm = rng.permutation(np.arange(1, 10)).reshape(3, 3)
        r, c = 3 * i, 3 * i
        board[r:r + 3, c:c + 3] = perm

    # Solve remaining cells
    _solve(board, rng)
    return board


def remove_cells(board: np.ndarray, n_remove: int,
                 rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Create a puzzle by blanking out `n_remove` random cells.

    Parameters
    ----------
    board : np.ndarray, shape (9, 9)
        A fully solved Sudoku board.
    n_remove : int
        Number of cells to mask (set to 0).
    rng : np.random.RandomState, optional
        Random state for reproducibility.

    Returns
    -------
    np.ndarray, shape (9, 9), dtype int
        Puzzle board with 0 representing masked (unknown) cells.
    """
    if rng is None:
        rng = np.random.RandomState()

    puzzle = board.copy()
    indices = rng.choice(81, size=n_remove, replace=False)
    rows, cols = np.divmod(indices, 9)
    puzzle[rows, cols] = 0
    return puzzle


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_board(board: np.ndarray) -> bool:
    """Return True if `board` is a fully valid solved Sudoku."""
    target = set(range(1, 10))
    for i in range(9):
        if set(board[i, :]) != target:
            return False
        if set(board[:, i]) != target:
            return False
    for br in range(3):
        for bc in range(3):
            block = board[3 * br:3 * br + 3, 3 * bc:3 * bc + 3]
            if set(block.ravel()) != target:
                return False
    return True


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SudokuDataset(Dataset):
    """
    Pre-generated Sudoku puzzle dataset.

    Each sample is a dict containing flat (81,) tensors that represent a
    single 9x9 Sudoku puzzle together with its solution and positional
    metadata.

    Parameters
    ----------
    num_puzzles : int
        Number of puzzles to generate.
    n_remove : int
        Number of cells to mask per puzzle.
    seed : int
        Base random seed for reproducibility.
    """

    def __init__(self, num_puzzles: int = 5000, n_remove: int = 40,
                 seed: int = 42):
        super().__init__()
        self.num_puzzles = num_puzzles
        self.n_remove = n_remove

        rng = np.random.RandomState(seed)

        self.puzzles = []   # list of (9,9) int arrays with 0 = masked
        self.solutions = [] # list of (9,9) int arrays, full solution

        for _ in range(num_puzzles):
            solution = generate_solved_board(rng)
            puzzle = remove_cells(solution, n_remove, rng)
            self.puzzles.append(puzzle)
            self.solutions.append(solution)

    # -- pre-computed positional index grids (shared across all samples) -----
    _ROW_IDS = torch.arange(9).unsqueeze(1).expand(9, 9).reshape(81)
    _COL_IDS = torch.arange(9).unsqueeze(0).expand(9, 9).reshape(81)

    def __len__(self) -> int:
        return self.num_puzzles

    def __getitem__(self, idx: int) -> dict:
        """
        Returns
        -------
        dict
            grid_tokens : LongTensor [81]
                Puzzle values; 0 = MASK, 1-9 = given digits.
            target : LongTensor [81]
                Solution digits 1-9 everywhere, but given (non-masked) cells
                are set to -100 so they can be ignored in cross-entropy loss.
            row_ids : LongTensor [81]
                Row index (0-8) for each flattened cell.
            col_ids : LongTensor [81]
                Column index (0-8) for each flattened cell.
            mask : BoolTensor [81]
                True for masked (unknown) cells.
            n_masked : int
                Number of masked cells in this puzzle.
        """
        puzzle = self.puzzles[idx].ravel()      # (81,)
        solution = self.solutions[idx].ravel()  # (81,)

        grid_tokens = torch.from_numpy(puzzle.copy()).long()
        mask = grid_tokens == 0  # True where cell is unknown

        # Target: solution digits everywhere, but ignore given cells in loss
        target = torch.from_numpy(solution.copy()).long()
        target[~mask] = -100

        return {
            "grid_tokens": grid_tokens,
            "target": target,
            "row_ids": self._ROW_IDS.clone(),
            "col_ids": self._COL_IDS.clone(),
            "mask": mask,
            "n_masked": int(mask.sum().item()),
        }


# ---------------------------------------------------------------------------
# Convenience loader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    difficulty: str = "medium",
    batch_size: int = 64,
    num_train: int = 5000,
    num_val: int = 500,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for a given difficulty level.

    Parameters
    ----------
    difficulty : {'easy', 'medium', 'hard'}
        Controls how many cells are removed.
            - easy   → 30 cells removed
            - medium → 45 cells removed
            - hard   → 55 cells removed
    batch_size : int
        Batch size for both loaders.
    num_train : int
        Number of training puzzles.
    num_val : int
        Number of validation puzzles.
    seed : int
        Base seed; validation uses ``seed + 1`` for independence.

    Returns
    -------
    (DataLoader, DataLoader)
        Train and validation data loaders.
    """
    difficulty_map = {"easy": 30, "medium": 45, "hard": 55}
    if difficulty not in difficulty_map:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Choose from {list(difficulty_map.keys())}."
        )
    n_remove = difficulty_map[difficulty]

    train_ds = SudokuDataset(num_puzzles=num_train, n_remove=n_remove,
                             seed=seed)
    val_ds = SudokuDataset(num_puzzles=num_val, n_remove=n_remove,
                           seed=seed + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate a small dataset and inspect one sample
    ds = SudokuDataset(num_puzzles=10, n_remove=40)
    sample = ds[0]

    print("Puzzle:")
    print(sample["grid_tokens"].reshape(9, 9))
    print()
    print("Target:")
    print(sample["target"].reshape(9, 9))
    print()
    print("Masked cells:", sample["n_masked"])
    print(f"Dataset size: {len(ds)}")

    # Validate every generated board
    for i in range(len(ds)):
        sol = ds.solutions[i]
        assert _validate_board(sol), f"Board {i} is invalid!"
    print("All boards valid ✓")

    print("OK")

"""Maze data: faithful to TRM's build_maze_dataset.py conventions.

Encoding (theirs, verified against their source): PAD=0, '#'=1 (wall),
' '=2 (empty), 'S'=3, 'G'=4, 'o'=5 (path). vocab=6. Input grid contains
walls/empty/S/G; label grid additionally marks the solution path with 'o'.
Train augmentation: 8 dihedral transforms (their builder, aug=True).

load_trm_maze() consumes their builder's .npy output directly (zero drift).
synthetic_maze() generates small DFS mazes + BFS shortest-path labels for
CPU smoke tests only — NOT a benchmark substitute.

Metrics (matching the ablation dashboards): cell accuracy, solved (exact
board), path precision/recall/F1 over 'o' cells, copy-cell baseline.
"""
import os
from collections import deque

import numpy as np
import torch

PAD, WALL, EMPTY, START, GOAL, PATH = 0, 1, 2, 3, 4, 5
VOCAB_SIZE = 6


def load_trm_maze(root: str, split: str):
    d = os.path.join(root, split)
    x = np.load(os.path.join(d, "all__inputs.npy"))
    y = np.load(os.path.join(d, "all__labels.npy"))
    assert x.min() >= 1 and x.max() <= 5 and y.max() <= 5
    return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


def synthetic_maze(n: int, hw: int = 9, seed: int = 0):
    """DFS perfect mazes on odd-sized grids; label = BFS shortest path."""
    assert hw % 2 == 1
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    while len(xs) < n:
        g = np.full((hw, hw), WALL, dtype=np.int64)
        # DFS carve on odd lattice
        start = (1, 1)
        g[start] = EMPTY
        stack = [start]
        while stack:
            r, c = stack[-1]
            nbrs = [(r + dr, c + dc) for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2))
                    if 0 < r + dr < hw and 0 < c + dc < hw and g[r + dr, c + dc] == WALL]
            if not nbrs:
                stack.pop(); continue
            nr, nc = nbrs[rng.integers(len(nbrs))]
            g[(r + nr) // 2, (c + nc) // 2] = EMPTY
            g[nr, nc] = EMPTY
            stack.append((nr, nc))
        cells = np.argwhere(g == EMPTY)
        s_idx, g_idx = rng.choice(len(cells), 2, replace=False)
        (sr, sc), (gr, gc) = cells[s_idx], cells[g_idx]
        if (sr, sc) == (gr, gc):
            continue
        # BFS shortest path
        prev = {}
        q = deque([(sr, sc)])
        seen = {(sr, sc)}
        while q:
            r, c = q.popleft()
            if (r, c) == (gr, gc):
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < hw and 0 <= nc < hw and g[nr, nc] != WALL \
                        and (nr, nc) not in seen:
                    seen.add((nr, nc)); prev[(nr, nc)] = (r, c)
                    q.append((nr, nc))
        if (gr, gc) not in prev and (gr, gc) != (sr, sc):
            continue
        x = g.copy(); x[sr, sc] = START; x[gr, gc] = GOAL
        y = x.copy()
        node = prev.get((gr, gc))
        while node and node != (sr, sc):
            y[node] = PATH
            node = prev.get(node)
        if not (y == PATH).any():
            continue
        xs.append(x.flatten()); ys.append(y.flatten())
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


def maze_metrics(pred: torch.Tensor, label: torch.Tensor, inp: torch.Tensor):
    """pred/label/inp: (B, N) token ids."""
    cell = (pred == label).float().mean().item()
    solved = (pred == label).all(-1).float().mean().item()
    tp = ((pred == PATH) & (label == PATH)).sum().item()
    fp = ((pred == PATH) & (label != PATH)).sum().item()
    fn = ((pred != PATH) & (label == PATH)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    copy_baseline = (inp == label).float().mean().item()
    return {"cell_accuracy": cell, "solved_accuracy": solved,
            "path_precision": prec, "path_recall": rec, "path_f1": f1,
            "copy_cell_baseline": copy_baseline}


def sudoku_metrics(pred: torch.Tensor, label: torch.Tensor, inp: torch.Tensor):
    cell = (pred == label).float().mean().item()
    board = (pred == label).all(-1).float().mean().item()
    copy_baseline = (inp == label).float().mean().item()
    return {"cell_accuracy": cell, "board_accuracy": board,
            "copy_cell_baseline": copy_baseline}

"""Reasoning benchmarks used to compare FPSA-R against looped-transformer baselines.

Three families, chosen because each one is known to separate *recurrent-depth*
models from fixed-depth ones, and each stresses a different axis:

``sudoku``        constraint propagation on a 9x9 grid -- iterative refinement
``maze``          shortest-path finding on a grid -- long-range planning
``state_track``   prefix products in the symmetric group S5 -- provably outside
                  the reach of a constant-depth transformer, and the cleanest
                  test of length generalisation at test time

Everything is generated on the fly from a seed, so a run is reproducible without
shipping data files.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# =============================================================================
# Sudoku
# =============================================================================

def _valid(board: np.ndarray, r: int, c: int, v: int) -> bool:
    if v in board[r, :] or v in board[:, c]:
        return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    return v not in board[br:br + 3, bc:bc + 3]


def _fill(board: np.ndarray, rng: np.random.RandomState) -> bool:
    empty = np.argwhere(board == 0)
    if len(empty) == 0:
        return True
    r, c = empty[0]
    cand = np.arange(1, 10)
    rng.shuffle(cand)
    for v in cand:
        if _valid(board, r, c, v):
            board[r, c] = v
            if _fill(board, rng):
                return True
            board[r, c] = 0
    return False


def _solved_board(rng: np.random.RandomState) -> np.ndarray:
    b = np.zeros((9, 9), dtype=np.int64)
    for i in range(3):
        b[3 * i:3 * i + 3, 3 * i:3 * i + 3] = rng.permutation(np.arange(1, 10)).reshape(3, 3)
    _fill(b, rng)
    return b


class SudokuDataset(Dataset):
    """Input: puzzle with ``n_blank`` cells masked (token 0). Target: solution.

    Vocabulary: 0 = blank, 1..9 = digits.
    """
    vocab_size = 10
    seq_len = 81
    grid = (9, 9)
    conv_type = "conv2d"
    causal = False
    name = "sudoku"

    def __init__(self, n_samples: int, n_blank: int = 30, seed: int = 0,
                 skip_generate: bool = False):
        self.n_blank = n_blank
        if skip_generate:
            return
        rng = np.random.RandomState(seed)
        self.x, self.y = [], []
        for _ in range(n_samples):
            sol = _solved_board(rng)
            puz = sol.copy()
            idx = rng.permutation(81)[:n_blank]
            puz.reshape(-1)[idx] = 0
            self.x.append(puz.reshape(-1))
            self.y.append(sol.reshape(-1))
        self.x = torch.tensor(np.array(self.x), dtype=torch.long)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)
        self.n_blank = n_blank

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        # loss mask: only score the blanked cells
        return self.x[i], self.y[i], (self.x[i] == 0)


# =============================================================================
# Maze: shortest path on a grid
# =============================================================================

class MazeDataset(Dataset):
    """Input tokens: 0=wall, 1=free, 2=start, 3=goal. Target: 0=off-path,
    1=on-shortest-path (start and goal included).

    Solvable instances only; walls are sampled until a path exists.
    """
    vocab_size = 4
    conv_type = "conv2d"
    causal = False
    name = "maze"

    def __init__(self, n_samples: int, size: int = 9, wall_frac: float = 0.30,
                 seed: int = 0, skip_generate: bool = False):
        self.size = size
        self.seq_len = size * size
        self.grid = (size, size)
        self.out_vocab = 2
        if skip_generate:
            return
        rng = np.random.RandomState(seed)
        xs, ys = [], []
        while len(xs) < n_samples:
            g = (rng.rand(size, size) > wall_frac).astype(np.int64)   # 1 free / 0 wall
            free = np.argwhere(g == 1)
            if len(free) < 4:
                continue
            si, gi = rng.choice(len(free), 2, replace=False)
            s, t = tuple(free[si]), tuple(free[gi])
            path = _bfs_path(g, s, t)
            if path is None or len(path) < 3:
                continue
            inp = g.copy()
            inp[s] = 2
            inp[t] = 3
            tgt = np.zeros((size, size), dtype=np.int64)
            for (r, c) in path:
                tgt[r, c] = 1
            xs.append(inp.reshape(-1))
            ys.append(tgt.reshape(-1))
        self.x = torch.tensor(np.array(xs), dtype=torch.long)
        self.y = torch.tensor(np.array(ys), dtype=torch.long)
        self.out_vocab = 2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], torch.ones_like(self.x[i], dtype=torch.bool)


def _bfs_path(grid: np.ndarray, s: Tuple[int, int], t: Tuple[int, int]):
    from collections import deque
    n, m = grid.shape
    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == t:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < m and grid[nr, nc] == 1 and (nr, nc) not in prev:
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    return None


# =============================================================================
# S5 state tracking
# =============================================================================

def _parity(p: Tuple[int, ...]) -> int:
    seen, par = [False] * len(p), 0
    for i in range(len(p)):
        if seen[i]:
            continue
        j, cyc = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            cyc += 1
        par += cyc - 1
    return par % 2


def _group_elements(group: str) -> List[Tuple[int, ...]]:
    """Permutation groups used for state tracking. A5 and S5 are non-solvable,
    so prefix products cannot be computed by any constant-depth circuit family
    in TC0 -- the canonical separation between fixed-depth transformers and
    recurrent-depth models."""
    from itertools import permutations
    n = {"s3": 3, "s4": 4, "a5": 5, "s5": 5}[group]
    els = sorted(permutations(range(n)))
    if group == "a5":
        els = [e for e in els if _parity(e) == 0]
    return els


class StateTrackingDataset(Dataset):
    """Prefix products in S5.

    Input: a length-L sequence of group elements. Target at position i: the
    product ``g_1 * ... * g_i``. Solving this requires a serial scan, so a
    fixed-depth transformer needs depth growing with L while a recurrent-depth
    model can in principle solve any L. Held-out longer lengths are the cleanest
    test of whether test-time iteration actually buys extrapolation.
    """
    conv_type = "none"
    causal = True
    name = "state_track"

    def __init__(self, n_samples: int, length: int = 16, n_gen: int = 5, seed: int = 0,
                 group: str = "a5"):
        self.group = group
        self.elements = _group_elements(group)
        self.index = {e: i for i, e in enumerate(self.elements)}
        self.vocab_size = len(self.elements)
        self.seq_len = length
        self.grid = None
        self.n = len(self.elements[0])
        rng = np.random.RandomState(seed)

        # restrict the inputs to a small generating set: makes the task a genuine
        # sequential scan rather than a lookup
        gens = [self.elements[i] for i in rng.choice(len(self.elements), n_gen, replace=False)]
        xs, ys = [], []
        for _ in range(n_samples):
            picks = rng.randint(0, n_gen, size=length)
            seq = [gens[p] for p in picks]
            cur = tuple(range(self.n))
            out = []
            for g in seq:
                cur = tuple(g[cur[i]] for i in range(self.n))   # compose
                out.append(self.index[cur])
            xs.append([self.index[g] for g in seq])
            ys.append(out)
        self.x = torch.tensor(np.array(xs), dtype=torch.long)
        self.y = torch.tensor(np.array(ys), dtype=torch.long)
        self.gens = gens

    def with_length(self, length: int, n_samples: int, seed: int):
        """Same generators, different sequence length (length generalisation)."""
        other = StateTrackingDataset.__new__(StateTrackingDataset)
        other.group = self.group
        other.elements, other.index = self.elements, self.index
        other.n = self.n
        other.vocab_size, other.grid = self.vocab_size, None
        other.seq_len = length
        other.gens = self.gens
        rng = np.random.RandomState(seed)
        xs, ys = [], []
        n_gen = len(self.gens)
        for _ in range(n_samples):
            picks = rng.randint(0, n_gen, size=length)
            seq = [self.gens[p] for p in picks]
            cur = tuple(range(self.n))
            out = []
            for g in seq:
                cur = tuple(g[cur[i]] for i in range(self.n))
                out.append(self.index[cur])
            xs.append([self.index[g] for g in seq])
            ys.append(out)
        other.x = torch.tensor(np.array(xs), dtype=torch.long)
        other.y = torch.tensor(np.array(ys), dtype=torch.long)
        return other

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], torch.ones_like(self.x[i], dtype=torch.bool)


# =============================================================================
# registry
# =============================================================================

_CACHE_DIR = os.environ.get(
    "FPSA_R_CACHE", os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "results", ".datacache"))


def _cached(key: str, build):
    """Sudoku generation costs ~2s per 100 boards; with dozens of runs in the
    grid that would dominate the wall clock. Cache the tensors on disk keyed by
    every generating parameter."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, key + ".pt")
    if os.path.exists(path):
        try:
            blob = torch.load(path, weights_only=False)
            obj = build(skip_generate=True)
            obj.x, obj.y = blob["x"], blob["y"]
            for k, v in blob.get("attrs", {}).items():
                setattr(obj, k, v)
            return obj
        except Exception:
            pass
    obj = build()
    torch.save({"x": obj.x, "y": obj.y,
                "attrs": {k: getattr(obj, k) for k in ("out_vocab", "vocab_size",
                                                       "seq_len", "grid", "n_blank")
                          if hasattr(obj, k)}}, path)
    return obj


@dataclass
class TaskSpec:
    name: str
    train: Dataset
    test: Dataset
    vocab_size: int
    out_vocab: int
    seq_len: int
    conv_type: str
    causal: bool
    extra_evals: Optional[Dict[str, Dataset]] = None


def make_task(name: str, n_train: int, n_test: int, seed: int = 0, **kw) -> TaskSpec:
    if name == "sudoku":
        nb = kw.get("n_blank", 30)
        mk = lambda n, b, sd: _cached(
            f"sudoku_n{n}_b{b}_s{sd}",
            lambda skip_generate=False: SudokuDataset(n, b, seed=sd,
                                                      skip_generate=skip_generate))
        tr = mk(n_train, nb, seed)
        te = mk(n_test, nb, seed + 9973)
        extra = {f"blank{b}": mk(max(64, n_test // 2), b, seed + 5000 + b)
                 for b in kw.get("extra_blanks", [])}
        return TaskSpec("sudoku", tr, te, 10, 10, 81, "conv2d", False, extra or None)

    if name == "maze":
        size = kw.get("size", 9)
        mk = lambda n, sd: _cached(
            f"maze_n{n}_sz{size}_s{sd}",
            lambda skip_generate=False: MazeDataset(n, size, seed=sd,
                                                    skip_generate=skip_generate))
        tr = mk(n_train, seed)
        te = mk(n_test, seed + 9973)
        # Larger grids than the model was trained on: does test-time iteration
        # buy generalisation to bigger planning problems?
        extra = {}
        for sz in kw.get("extra_sizes", []):
            mk2 = lambda n, sd, sz=sz: _cached(
                f"maze_n{n}_sz{sz}_s{sd}",
                lambda skip_generate=False: MazeDataset(n, sz, seed=sd,
                                                        skip_generate=skip_generate))
            extra[f"size{sz}"] = mk2(max(64, n_test // 2), seed + 7000 + sz)
        return TaskSpec("maze", tr, te, 4, 2, size * size, "conv2d", False, extra or None)

    if name == "state_track":
        L = kw.get("length", 16)
        g = kw.get("group", "a5")
        ng = kw.get("n_gen", 4)
        tr = StateTrackingDataset(n_train, L, n_gen=ng, seed=seed, group=g)
        te = tr.with_length(L, n_test, seed=seed + 9973)
        extra = {f"len{l}": tr.with_length(l, max(64, n_test // 2), seed=seed + 6000 + l)
                 for l in kw.get("extra_lengths", [])}
        return TaskSpec("state_track", tr, te, tr.vocab_size, tr.vocab_size, L,
                        "none", True, extra or None)

    raise KeyError(f"unknown task {name!r}")

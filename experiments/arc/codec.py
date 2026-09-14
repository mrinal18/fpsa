"""ARC canvas codec and reversible augmentations compatible with pinned TRM.

PAD=0, EOS=1, colors 0..9 -> 2..11. Black is NOT padding.
The eight transform IDs follow TRM dataset/common.py; grid decoding follows its
largest valid top-left rectangle rule. See docs/ARC-Refinement.md for provenance.
"""
import numpy as np

INVERSE = (0, 3, 2, 1, 4, 5, 6, 7)


def transform(grid, tid):
    a = np.asarray(grid)
    if a.ndim != 2 or tid not in range(8):
        raise ValueError('Expected a 2D grid and transform ID 0..7')
    return (a, np.rot90(a), np.rot90(a, 2), np.rot90(a, 3),
            np.fliplr(a), np.flipud(a), a.T, np.fliplr(np.rot90(a)))[tid].copy()


def parse_identifier(name):
    if '|||' not in name:
        return name, 0, np.arange(10)
    base, tid, colors = name.rsplit('|||', 2)
    if not tid.startswith('t'):
        raise ValueError('Invalid augmentation ID')
    perm = np.array([int(c) for c in colors])
    if sorted(perm.tolist()) != list(range(10)) or perm[0] != 0:
        raise ValueError('Invalid color permutation or moved black')
    if int(tid[1:]) not in range(8):
        raise ValueError('Invalid transform')
    return base, int(tid[1:]), perm


def inverse(grid, tid, permutation):
    return np.argsort(permutation)[transform(grid, INVERSE[tid])]


def encode(grid, size=30, offset=(0, 0)):
    a = np.asarray(grid, dtype=np.int64)
    if a.ndim != 2 or a.size == 0 or a.min() < 0 or a.max() > 9:
        raise ValueError('Invalid ARC grid')
    r, c = offset
    h, w = a.shape
    if min(r, c) < 0 or r+h > size or c+w > size:
        raise ValueError('Grid does not fit canvas')
    canvas = np.zeros((size, size), dtype=np.int64)
    canvas[r:r+h, c:c+w] = a + 2
    if r+h < size:
        canvas[r+h, c:c+w] = 1
    if c+w < size:
        canvas[r:r+h, c+w] = 1
    return canvas.flatten()


def decode(sequence, size=30):
    """Official-style largest top-left valid rectangle; None for invalid output."""
    a = np.asarray(sequence).reshape(size, size)
    width, best_area, best = size, 0, None
    for row in range(size):
        invalid = np.flatnonzero((a[row, :width] < 2) | (a[row, :width] > 11))
        if invalid.size:
            width = int(invalid[0])
        area = (row+1) * width
        if area > best_area:
            best_area = area
            best = a[:row+1, :width].copy() - 2
    return best


def grid_key(grid):
    a = np.asarray(grid)
    if a.ndim != 2 or not a.size or a.shape[0] > 30 or a.shape[1] > 30:
        raise ValueError('Invalid candidate shape')
    if not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() > 9:
        raise ValueError('Invalid candidate colors')
    return (tuple(a.shape), tuple(int(v) for v in a.flat))

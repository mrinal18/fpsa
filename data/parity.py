"""Prefix-parity toy task.

Input:  binary sequence b_1..b_N (tokens 0/1).
Target: t_i = XOR(b_1..b_i) at every position (dense supervision).
Position i requires aggregating all positions j <= i, so solving it exactly
exercises iterative information propagation in the loop.

Train/test split is by disjoint random seeds over uniform sequences; for
N >= 16 the collision probability between 50k train and 8k test sequences
is negligible but we dedupe anyway.
"""
import numpy as np
import torch


def make_parity(n_train: int, n_test: int, seq_len: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    total = n_train + n_test
    if seq_len <= 20:  # enumerate all 2^L, shuffle, slice -> guaranteed disjoint
        all_ints = rng.permutation(2 ** seq_len)[:total]
        assert total <= 2 ** seq_len
        seqs = ((all_ints[:, None] >> np.arange(seq_len)) & 1).astype(np.int64)
    else:
        seqs = rng.integers(0, 2, size=(int(total * 3), seq_len), dtype=np.int64)
        seqs = np.unique(seqs, axis=0)
        rng.shuffle(seqs)
        seqs = seqs[:total]
        assert len(seqs) == total, "not enough unique sequences"
    targets = np.bitwise_xor.accumulate(seqs, axis=1)
    xtr, ytr = seqs[:n_train], targets[:n_train]
    xte, yte = seqs[n_train:], targets[n_train:]
    to = lambda a: torch.from_numpy(a.copy())
    return (to(xtr), to(ytr)), (to(xte), to(yte))

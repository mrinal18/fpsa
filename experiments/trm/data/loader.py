"""Dataset loading and batching for the TRM experiments.

Semantics match the TRM reference: one "epoch" visits every GROUP once,
sampling one random augmented variant per group; the label ignore id (0/PAD)
is remapped to -100 for the loss.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

IGNORE_LABEL_ID = -100


@dataclass
class SplitData:
    inputs: np.ndarray          # (num_examples, seq_len)
    labels: np.ndarray
    group_indices: np.ndarray   # (num_groups + 1,)
    metadata: dict


def load_split(dataset_dir: str, split: str) -> SplitData:
    d = os.path.join(dataset_dir, split)
    with open(os.path.join(d, "dataset.json")) as f:
        metadata = json.load(f)
    return SplitData(
        inputs=np.load(os.path.join(d, "all__inputs.npy")),
        labels=np.load(os.path.join(d, "all__labels.npy")),
        group_indices=np.load(os.path.join(d, "all__group_indices.npy")),
        metadata=metadata,
    )


def _to_batch(inputs: np.ndarray, labels: np.ndarray, ignore_label_id: Optional[int],
              device: torch.device) -> Dict[str, torch.Tensor]:
    labels = labels.astype(np.int64)
    if ignore_label_id is not None:
        labels = np.where(labels == ignore_label_id, IGNORE_LABEL_ID, labels)
    return {
        "inputs": torch.from_numpy(inputs.astype(np.int64)).to(device),
        "labels": torch.from_numpy(labels).to(device),
    }


class TrainBatcher:
    """Streams shuffled train batches; each epoch samples one variant/group."""

    def __init__(self, data: SplitData, batch_size: int, seed: int = 0):
        self.data = data
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.num_groups = len(data.group_indices) - 1
        self.steps_per_epoch = max(1, self.num_groups // batch_size)
        self._queue: List[int] = []

    def _refill(self):
        g = self.data.group_indices
        starts, ends = g[:-1], g[1:]
        picks = starts + (self.rng.rand(self.num_groups) * (ends - starts)).astype(np.int64)
        order = self.rng.permutation(self.num_groups)
        # Extend (not replace): the leftover tail of the previous epoch is
        # kept, so every group is visited once per epoch, and the loop in
        # next_batch terminates even when num_groups < batch_size.
        self._queue.extend(picks[order])

    def next_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        while len(self._queue) < self.batch_size:
            self._refill()
        idx = np.array([self._queue.pop() for _ in range(self.batch_size)])
        return _to_batch(
            self.data.inputs[idx], self.data.labels[idx],
            self.data.metadata.get("ignore_label_id"), device,
        )


def iter_test_batches(data: SplitData, batch_size: int,
                      device: torch.device) -> Iterator[Tuple[Dict[str, torch.Tensor], int]]:
    """Yields (batch, n_real) — the final batch is padded up to batch_size by
    repeating the first example; n_real marks how many rows are genuine."""
    n = len(data.inputs)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        inputs = data.inputs[start:end]
        labels = data.labels[start:end]
        n_real = end - start
        if n_real < batch_size:
            pad = batch_size - n_real
            inputs = np.concatenate([inputs, np.repeat(inputs[:1], pad, axis=0)])
            labels = np.concatenate([labels, np.repeat(labels[:1], pad, axis=0)])
        yield _to_batch(inputs, labels, data.metadata.get("ignore_label_id"), device), n_real

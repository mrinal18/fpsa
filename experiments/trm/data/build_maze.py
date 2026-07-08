"""Build the Maze-Hard 30x30 dataset (TRM/HRM benchmark protocol).

Downloads sapientinc/maze-30x30-hard-1k from the Hugging Face Hub. Vocab:
0=PAD, then '#'(wall), ' '(free), 'S'(start), 'G'(goal), 'o'(path) -> 1..5.

Note: the default is NO augmentation — FPRM reports the 8x dihedral-augmented
build collapses to ~5% accuracy; pass --aug to reproduce that setting.

Usage:
    python build_maze.py --output-dir data/maze-30x30-hard-1k-noaug
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PuzzleDatasetMetadata, dihedral_transform, save_split

CHARSET = "# SGo"


def convert_subset(set_name: str, args) -> None:
    from huggingface_hub import hf_hub_download

    inputs, labels = [], []
    grid_size = None
    csv_path = hf_hub_download(args.source_repo, f"{set_name}.csv", repo_type="dataset")
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for _source, q, a, _rating in reader:
            if grid_size is None:
                n = int(len(q) ** 0.5)
                grid_size = (n, n)
            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size))

    if set_name == "train" and args.subsample_size is not None and args.subsample_size < len(inputs):
        idx = np.random.choice(len(inputs), size=args.subsample_size, replace=False)
        inputs = [inputs[i] for i in idx]
        labels = [labels[i] for i in idx]

    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1

    out_inputs, out_labels, group_indices = [], [], [0]
    n_aug = 8 if (set_name == "train" and args.aug) else 1
    for inp, out in zip(inputs, labels):
        for aug_idx in range(n_aug):
            out_inputs.append(char2id[dihedral_transform(inp, aug_idx).reshape(-1)])
            out_labels.append(char2id[dihedral_transform(out, aug_idx).reshape(-1)])
        group_indices.append(len(out_inputs))

    metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        vocab_size=len(CHARSET) + 1,
        seq_len=int(math.prod(grid_size)),
        total_groups=len(group_indices) - 1,
        sets=["all"],
    )
    save_split(args.output_dir, set_name, np.stack(out_inputs), np.stack(out_labels),
               np.array(group_indices, dtype=np.int64), metadata)
    print(f"{set_name}: {len(out_inputs)} examples in {len(group_indices) - 1} groups -> {args.output_dir}/{set_name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-repo", default="sapientinc/maze-30x30-hard-1k")
    p.add_argument("--output-dir", default="data/maze-30x30-hard-1k-noaug")
    p.add_argument("--subsample-size", type=int, default=None)
    p.add_argument("--aug", action="store_true", help="8x dihedral augmentation (known to hurt)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    convert_subset("train", args)
    convert_subset("test", args)


if __name__ == "__main__":
    main()

"""Build the Sudoku-Extreme dataset (TRM/HRM benchmark protocol).

Downloads sapientinc/sudoku-extreme from the Hugging Face Hub, subsamples the
training set, and applies validity-preserving augmentations. Matches the TRM
reference build exactly (vocab: 0=PAD, 1='0' blank, 2..10 = digits 1..9).

Usage (TRM paper protocol — 1000 puzzles x 1000 augmentations):
    python build_sudoku_extreme.py \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 --num-aug 1000
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PuzzleDatasetMetadata, save_split, shuffle_sudoku


def convert_subset(set_name: str, args) -> None:
    from huggingface_hub import hf_hub_download

    inputs, labels = [], []
    csv_path = hf_hub_download(args.source_repo, f"{set_name}.csv", repo_type="dataset")
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # header
        for _source, q, a, rating in reader:
            if args.min_difficulty is not None and int(rating) < args.min_difficulty:
                continue
            assert len(q) == 81 and len(a) == 81
            inputs.append(np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8).reshape(9, 9) - ord("0"))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord("0"))

    if set_name == "train" and args.subsample_size is not None and args.subsample_size < len(inputs):
        idx = np.random.choice(len(inputs), size=args.subsample_size, replace=False)
        inputs = [inputs[i] for i in idx]
        labels = [labels[i] for i in idx]

    num_aug = args.num_aug if set_name == "train" else 0
    out_inputs, out_labels, group_indices = [], [], [0]
    for orig_inp, orig_out in zip(inputs, labels):
        for aug_idx in range(1 + num_aug):
            inp, out = (orig_inp, orig_out) if aug_idx == 0 else shuffle_sudoku(orig_inp, orig_out)
            out_inputs.append(inp.reshape(-1))
            out_labels.append(out.reshape(-1))
        group_indices.append(len(out_inputs))

    inputs_arr = np.stack(out_inputs).astype(np.uint8) + 1   # shift: 0 becomes PAD-free blank id 1
    labels_arr = np.stack(out_labels).astype(np.uint8) + 1
    assert inputs_arr.min() >= 1 and inputs_arr.max() <= 10

    metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        vocab_size=11,   # PAD + '0'..'9'
        seq_len=81,
        total_groups=len(group_indices) - 1,
        sets=["all"],
    )
    save_split(args.output_dir, set_name, inputs_arr, labels_arr,
               np.array(group_indices, dtype=np.int64), metadata)
    print(f"{set_name}: {len(inputs_arr)} examples in {len(group_indices) - 1} groups -> {args.output_dir}/{set_name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-repo", default="sapientinc/sudoku-extreme")
    p.add_argument("--output-dir", default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--subsample-size", type=int, default=1000)
    p.add_argument("--num-aug", type=int, default=1000)
    p.add_argument("--min-difficulty", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    convert_subset("train", args)
    convert_subset("test", args)


if __name__ == "__main__":
    main()

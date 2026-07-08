#!/usr/bin/env bash
# Maze-Hard 30x30 benchmark: TRM baseline vs Implicit TRM.
#
# seq_len=900 at global batch 768 exceeds a single 40GB GPU for the TRM
# baseline; the configs use gradient accumulation (micro-batches with
# independent ACT carries, semantics-preserving). On an 80GB GPU you can
# raise micro_batch_size. The Implicit TRM fits larger micro-batches because
# the inner loop stores no activations.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0}"

pip install -q torch numpy pyyaml huggingface_hub

if [ ! -d data/maze-30x30-hard-1k-noaug ]; then
  python data/build_maze.py --output-dir data/maze-30x30-hard-1k-noaug
fi

for SEED in $SEEDS; do
  python train.py --config configs/maze_trm.yaml  --seed "$SEED" --run_name "maze-trm-s${SEED}"
  python train.py --config configs/maze_itrm.yaml --seed "$SEED" --run_name "maze-itrm-s${SEED}"
done

python summarize_results.py --results-dir results

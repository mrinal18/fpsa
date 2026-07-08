#!/usr/bin/env bash
# Sudoku-Extreme benchmark: TRM baseline vs Implicit TRM.
#
# Requirements: 1 GPU with >= 40GB (L40S / A100). Full protocol takes
# ~18-20h per run; pass PILOT=1 for the 10%-budget pilot (~2h per run).
#
#   PILOT=1 ./run_sudoku_benchmark.sh     # quick ranking
#   ./run_sudoku_benchmark.sh             # full benchmark
#   SEEDS="0 1 2" ./run_sudoku_benchmark.sh  # multi-seed
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0}"
PILOT="${PILOT:-0}"

pip install -q torch numpy pyyaml huggingface_hub

# 1. Dataset (downloads from HF on first run)
if [ ! -d data/sudoku-extreme-1k-aug-1000 ]; then
  python data/build_sudoku_extreme.py \
    --output-dir data/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 --num-aug 1000
fi

if [ "$PILOT" = "1" ]; then
  TRM_CFG=configs/sudoku_pilot_trm.yaml
  ITRM_CFG=configs/sudoku_pilot_itrm.yaml
  TAG=pilot
else
  TRM_CFG=configs/sudoku_trm.yaml
  ITRM_CFG=configs/sudoku_itrm.yaml
  TAG=full
fi

for SEED in $SEEDS; do
  # TRM baseline (reference protocol)
  python train.py --config "$TRM_CFG"  --seed "$SEED" --run_name "sudoku-${TAG}-trm-s${SEED}"
  # Implicit TRM (ours)
  python train.py --config "$ITRM_CFG" --seed "$SEED" --run_name "sudoku-${TAG}-itrm-s${SEED}"
done

python summarize_results.py --results-dir results

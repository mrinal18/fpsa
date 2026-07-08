#!/usr/bin/env bash
# Ablations for the Implicit TRM on Sudoku-Extreme (pilot budget each):
#   1. gradient estimator: neumann vs phantom vs truncated BPTT
#   2. inner-depth scaling: max_iter 16 -> 32 -> 64 at O(1) memory
#   3. contractivity: jacobian regularization on/off, spectral norm on/off
set -euo pipefail
cd "$(dirname "$0")"

pip install -q torch numpy pyyaml huggingface_hub

if [ ! -d data/sudoku-extreme-1k-aug-1000 ]; then
  python data/build_sudoku_extreme.py \
    --output-dir data/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 --num-aug 1000
fi

CFG=configs/sudoku_pilot_itrm.yaml

# 1. Gradient estimator
python train.py --config $CFG --grad_mode neumann --run_name abl-grad-neumann
python train.py --config $CFG --grad_mode phantom --run_name abl-grad-phantom
python train.py --config $CFG --grad_mode bptt --bptt_steps 6 --run_name abl-grad-bptt6

# 2. Inner-depth scaling (memory should stay flat; check peak_mem_gb in logs)
python train.py --config $CFG --inner_max_iter 32 --run_name abl-depth-32
python train.py --config $CFG --inner_max_iter 64 --run_name abl-depth-64

# 3. Contractivity mechanisms
python train.py --config $CFG --jacobian_reg_lambda 0.1 --run_name abl-jacreg-0.1
python train.py --config $CFG --spectral_norm true --run_name abl-specnorm

python summarize_results.py --results-dir results

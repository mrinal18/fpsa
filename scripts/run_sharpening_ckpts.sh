#!/usr/bin/env bash
# Two checkpoints for the attention-sharpening analysis: the heavily constrained
# recipe and the rho-targeted one, otherwise identical.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/ckpt
run() {
  arch=$1; shift
  out="results/ckpt/maze7__${arch}__s0.json"
  [ -f "${out%.json}.pt" ] && return
  python3 experiments/reasoning/train.py --arch "$arch" --seed 0 --steps 1000 \
    --threads 1 --task maze --maze_size 7 --bs 32 --n_train 8000 --n_test 256 \
    --max_iter 32 --max_iter_eval 32 --save_model --verbose \
    --out "$out" "$@" > "${out%.json}.log" 2>&1
}
run deq_block &
run deq_free_anderson --contraction_target 1.0 &
wait
echo CKPT_COMPLETE >> results/grid.log

#!/usr/bin/env bash
# The controlled comparison the T=8 grid points to.
#
# A loop with rho ~ 0.8 needs roughly 30 iterations for its residual to fall
# below tolerance. Trained at T=8 the forward is nowhere near its fixed point,
# so implicit differentiation is answering a question the forward pass never
# asked -- while truncated BPTT is exactly correct for the 8 steps that did run.
# This script re-runs the same architectures at T=32, where the equilibrium
# premise actually holds, and records what each one costs in memory to get
# there.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/converged

run() {
  arch=$1; seed=$2; shift 2
  out="results/converged/maze7__${arch}__s${seed}.json"
  [ -f "$out" ] && return
  python3 experiments/reasoning/train.py --arch "$arch" --seed "$seed" \
    --steps 1000 --threads 1 --task maze --maze_size 7 --extra_sizes 9 11 \
    --bs 32 --n_train 8000 --n_test 512 \
    --max_iter 32 --max_iter_eval 32 --n_backwards 4 --verbose \
    --out "$out" "$@" > "${out%.json}.log" 2>&1
}

for seed in 0 1 2; do
  for arch in fpsa_r deq_block fprm looped_bptt; do
    run "$arch" "$seed" &
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  done
done
wait
echo CONVERGED_COMPLETE >> results/grid.log

#!/usr/bin/env bash
# More seeds for the winning recipe, and whether the in-layer FPSA loop helps
# once the per-layer spectral caps are gone.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/contraction
run() {
  arch=$1; seed=$2; shift 2
  out="results/contraction/maze7__${arch}__s${seed}.json"
  [ -f "$out" ] && return
  python3 experiments/reasoning/train.py --arch "$arch" --seed "$seed" \
    --steps 1000 --threads 1 --task maze --maze_size 7 --extra_sizes 9 11 \
    --bs 32 --n_train 8000 --n_test 512 --max_iter 32 --max_iter_eval 32 \
    --contraction_target 1.0 --verbose \
    --out "$out" "$@" > "${out%.json}.log" 2>&1
}
for seed in 2 3; do
  run deq_free_anderson "$seed" &
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
done
for seed in 0 1; do
  run fpsa_free_anderson "$seed" &
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
done
wait
echo FOLLOWUP_COMPLETE >> results/grid.log

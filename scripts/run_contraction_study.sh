#!/usr/bin/env bash
# Can we keep the equilibrium without paying the capacity cost of hard
# per-layer spectral caps?
#
# Two independent levers, crossed:
#   solver    picard + Neumann/Anderson adjoint  vs  Anderson forward + GMRES adjoint
#   constraint  per-layer spectral caps (sigma <= 1)  vs  none, rho-target only
#
# The premise is that rho < 1 is a requirement of the *solvers*, not of implicit
# differentiation: the adjoint needs (I - J) invertible, and the forward needs a
# findable fixed point. Stronger solvers should let the model sit near rho = 1,
# where the spectral caps are no longer buying anything.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/contraction

run() {
  arch=$1; seed=$2; shift 2
  out="results/contraction/maze7__${arch}__s${seed}.json"
  [ -f "$out" ] && return
  python3 experiments/reasoning/train.py --arch "$arch" --seed "$seed" \
    --steps 1000 --threads 1 --task maze --maze_size 7 --extra_sizes 9 11 \
    --bs 32 --n_train 8000 --n_test 512 \
    --max_iter 32 --max_iter_eval 32 --verbose \
    --out "$out" "$@" > "${out%.json}.log" 2>&1
}

for seed in 0 1; do
  run deq_gmres        "$seed" &                                   # backward only
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  run deq_anderson_fwd "$seed" &                                   # both solvers
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  run deq_free_anderson "$seed" --contraction_target 1.0 &         # + no caps
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
done
wait
echo CONTRACTION_COMPLETE >> results/grid.log

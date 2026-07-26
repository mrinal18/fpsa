#!/usr/bin/env bash
# Everything: mechanism suite, comparison grid, matched-memory study.
# ~2.5h on 4 CPU cores. Re-running skips any result JSON that already exists.
set -u
cd "$(dirname "$0")/.."
mkdir -p results

MECH_ONLY="${MECH_ONLY:-m1_gradient_fidelity m2_activation_memory m5_solver_cost \
m6_token_convergence m7_rank_collapse m4_adjoint_solver m3_contraction_dynamics \
m1b_fidelity_vs_contraction}"

{
  echo "### mechanism suite"
  # shellcheck disable=SC2086
  python3 experiments/reasoning/mechanism.py --threads 1 --only $MECH_ONLY
  echo "### mechanism done"
} >> results/mechanism.log 2>&1 &
MECH=$!

{
  echo "### main grid: maze7"
  python3 experiments/reasoning/run_suite.py --stage main --tasks maze7 \
      --seeds 0 1 2 --steps 1000 --workers 3
  echo "### ablation grid: maze7"
  python3 experiments/reasoning/run_suite.py --stage ablation --ablation_task maze7 \
      --seeds 0 1 --steps 1000 --workers 3
} >> results/grid.log 2>&1
wait $MECH

# Matched-memory study: the practical payoff of the O(1) backward is that
# FPSA-R can afford 4x the fixed-point iterations while still storing less than
# FPRM's truncated BPTT does at 8.
mkdir -p results/matched_memory
for s in 0 1 2; do
  python3 experiments/reasoning/train.py --arch fpsa_r --seed "$s" --steps 1000 --threads 1 \
    --task maze --maze_size 7 --extra_sizes 9 11 --bs 32 --n_train 8000 --n_test 512 \
    --max_iter 32 --max_iter_eval 32 --verbose \
    --out "results/matched_memory/maze7__fpsa_r_T32__s$s.json" \
    > "results/matched_memory/maze7__fpsa_r_T32__s$s.log" 2>&1 &
done
wait
echo ALL_COMPLETE >> results/grid.log

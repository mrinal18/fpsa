#!/usr/bin/env bash
# Everything: comparison grid at the training budget, ablations, the
# converged-forward study, and the matched-memory study.
# Re-running skips any result JSON that already exists, so it resumes safely.
set -u
cd "$(dirname "$0")/.."
mkdir -p results

{
  echo "### main grid: maze7, T=8"
  python3 experiments/reasoning/run_suite.py --stage main --tasks maze7 \
      --seeds 0 1 2 --steps 1000 --workers 3
  echo "### ablation grid: maze7, T=8"
  python3 experiments/reasoning/run_suite.py --stage ablation --ablation_task maze7 \
      --seeds 0 1 --steps 1000 --workers 3
} >> results/grid.log 2>&1

# The forward budget where the equilibrium premise actually holds.
./scripts/run_converged_forward.sh

# Matched memory: FPSA-R at 32 iterations still stores less than truncated
# BPTT does at 8.
mkdir -p results/matched_memory
for s in 0 1 2; do
  out="results/matched_memory/maze7__fpsa_r_T32__s$s.json"
  [ -f "$out" ] && continue
  python3 experiments/reasoning/train.py --arch fpsa_r --seed "$s" --steps 1000 --threads 1 \
    --task maze --maze_size 7 --extra_sizes 9 11 --bs 32 --n_train 8000 --n_test 512 \
    --max_iter 32 --max_iter_eval 32 --verbose \
    --out "$out" > "${out%.json}.log" 2>&1 &
done
wait

python3 experiments/reasoning/mechanism.py --threads 2 \
    --only m1c_fidelity_vs_forward_budget >> results/mechanism.log 2>&1
echo ALL_COMPLETE >> results/grid.log

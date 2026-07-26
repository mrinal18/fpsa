#!/usr/bin/env bash
# Full comparison grid for FPSA-R. Runs on CPU; ~3h on 4 cores.
set -u
cd "$(dirname "$0")/.."
LOG=results/grid.log
mkdir -p results
{
  echo "=== main: maze7 (6 archs x 3 seeds) ==="
  python3 experiments/reasoning/run_suite.py --stage main --tasks maze7 \
      --seeds 0 1 2 --steps 1200 --workers 4
  echo "=== ablation: maze7 (7 variants x 3 seeds) ==="
  python3 experiments/reasoning/run_suite.py --stage ablation --ablation_task maze7 \
      --seeds 0 1 2 --steps 1200 --workers 4
  echo "=== main: maze9 (6 archs x 2 seeds) ==="
  python3 experiments/reasoning/run_suite.py --stage main --tasks maze9 \
      --seeds 0 1 --steps 1000 --workers 4
  echo GRID_COMPLETE
} >> "$LOG" 2>&1

# GATE 2 runbook — Sudoku-Extreme on A100 (Colab)

## 0. What this gate must produce
Exact-match accuracy on the FULL Sudoku-Extreme test split, trained on the
canonical 1k-train/1000-aug protocol, with convergence instrumentation.
Reference points (from TRM README, their reported numbers, not reproduced
by us): TRM-MLP ~87% +/- 2, TRM-Att variant lower; HRM ~55% (per TRM paper).

## 1. Data (ZERO protocol drift — use THEIR builder)
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git
cd TinyRecursiveModels && pip install -r requirements.txt  # or just: argdantic pydantic tqdm huggingface_hub numpy
python dataset/build_sudoku_dataset.py \
  --output-dir ../fpsa-bench/data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 --num-aug 1000
cd ../fpsa-bench
# NOTE: their builder is unseeded; record the build artifact (hash the npys)
# alongside results. Our loader consumes all__inputs.npy/all__labels.npy.

## 2. Sanity on the GPU box (minutes)
python3 tests/test_gate2.py                      # unit tests
python3 train/train_sudoku.py --config configs/gate2_smoke.yaml --synthetic 16
# expect cell acc ~0.9+ by step 400 (raw, not EMA)

## 3. Pilots — TWO ARMS (first real signal, ~2-5 h each on A100)
# Arm A: stabilized FPSA (guard v2 + jac-reg)
python3 train/train_sudoku.py --config configs/gate2_sudoku_pilot.yaml --seed 0
# Arm B: ACE-relaxed (anchored equilibrium; NO guard, NO jac-reg;
#        sigma audited every eval). Validated on parity: ~FPSA accuracy,
#        zero collapses, zero stability machinery. NOTE: slower warm-up
#        than FPSA (attention gain must be learned from near-zero) —
#        judge the pilot on the curve's slope after breakout, not on
#        early-step comparisons.
python3 train/train_sudoku.py --config configs/gate2_sudoku_ace_pilot.yaml --seed 0
# DECISION POINT: paste both pilot train_log.csv files back before
# launching full runs. Watch for ACE: sigma trajectory (expect ~0.8-1.2
# band, converging solves throughout); FPSA: guard count and sigma
# excursions.

## 4. Full runs (3 seeds/arm; ~1-2 days/seed on A100, est. UNVERIFIED;
##    pilot s/step is ground truth)
for s in 0 1 2; do
  python3 train/train_sudoku.py --config configs/gate2_sudoku.yaml --seed $s
  python3 train/train_sudoku.py --config configs/gate2_sudoku_ace.yaml --seed $s
done
# Post-training (optional, ACE): token-freezing compute analysis via the
# pattern in eval/ace_v4_calibrate.py (33% solve compute saved at 100%
# agreement on parity; Sudoku numbers TBD).
# Resumable: re-running the same command resumes from checkpoint.
# Colab disconnects: use --max_minutes to checkpoint before timeout.

## 5. Report back (paste into chat)
- runs/gate2_sudoku_seed*/train_log.csv  (full instrumentation)
- runs/gate2_sudoku_seed*/final.json     (full-test exact acc, raw + EMA)
- any guard activations / sigma excursions

## 6. Known knobs if things go wrong (in priority order, per protocol:
##    solver/norm first, hyperparameters second)
- residual not reaching ~tol: raise max_iter before anything else
- sigma rides >0.9 persistently: lower jac_rho_target to 0.8
- guard fires often: inspect residual curves FIRST (paste them)
- only then: lr, batch size
- value_mode: evolving is the validated alternate (half params)

## 7. Honest caveats
- FPSA configs are UNTUNED first passes; TRM's numbers come from a tuned
  recipe (50k epochs, wd=1.0, EMA, halting schedule). A gap on first run
  is expected and is information, not failure.
- No deep supervision yet (TRM/HRM both use it); planned enhancement 8.3.
- Compute estimates are extrapolations from CPU step timing; treat the
  pilot's measured s/step as ground truth.

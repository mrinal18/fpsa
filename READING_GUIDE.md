# WHERE TO LOOK — reading guide for the `ace-bench` branch

Start here, then follow the path for whatever you want to verify.

## If you read only three files
1. `docs/GOAL.md`      — what we're testing and the falsifiable criteria.
2. `models/trm_substrate.py` — the current main model (all 5 arms, 3 backward
   modes, deep supervision, freezing, jac-reg + guard, validity certificates).
3. `docs/RUNBOOK_TRM_SUBSTRATE.md` — the exact A100 command matrix + status.

## The code, by concern

### The models (`models/`)
- `trm_substrate.py`  ← **the current model.** TRM-shaped (z,y) deep-supervised
  loop. value_mode ∈ {fixed, fixed_ffn, evolving, blended, ace_relaxed};
  backward ∈ {neumann_k, bptt_k, phantom1}. Contains `_ImplicitAttach`
  (the single-pass Neumann adjoint) and the jac-reg/guard/certificate code.
- `ace.py`            ← ACEBlock: nonexpansive-by-construction map, learned
  anchor, ball projection, GroupSort FFN, certified vs relaxed attention.
- `fpsa_block.py`     ← original FPSA block + RoPE1D/RoPE2D (shared by all).
- `solver.py`        ← standalone fixed_point_solve + Neumann backward
  (used by the FPSA/ACE single-block models in `model.py`).
- `model.py`         ← FPSASeqModel / FPSADeepModel (single- & multi-block,
  correct residual convention + double_count toggle for the bug study).
- `trmfp.py`         ← TRM-FP: the coupled (z,y) blended-3 experiment
  (the one that confirmed the disagreement↔instability prediction; kept as
  a documented negative/analysis result).

### Data (`data/`) — all match the HRM/TRM protocol
- `sudoku.py`  ← loader for THEIR builder output (byte-equivalent aug verified)
                 + synthetic generator + metrics.
- `maze.py`    ← TRM-faithful maze encoding/loader (PAD/#/space/S/G/o = 0..5)
                 + synthetic DFS/BFS generator + path P/R/F1 metrics.
- `parity.py`  ← prefix-parity toy (Gate 1 / gradchecks).

### Training (`train/`)
- `train_bench.py`      ← **unified Sudoku+Maze harness, all arms** (EMA,
  dashboard metrics, jac-reg/guard, freezing). This is the one to run.
- `train_sudoku.py`     ← older single-block FPSA/ACE Sudoku harness (Gate 2).
- `train_ace_toy.py` / `train_trmfp_toy.py` / `train_toy.py` / `train_deep.py`
  ← parity trainers for ACE / TRM-FP / FPSA / multi-block studies.

### Validation & analysis (`eval/`)
- `trm_substrate_gradcheck.py` ← **proves the substrate's implicit path**:
  FD vs bptt_full vs neumann_k ladder (3e-13 @ k=40), phantom1≡neumann_1.
- `ace_validate.py` / `ace_v4_calibrate.py` ← ACE V1–V4 (gradcheck, σ audit,
  certificate calibration, token freezing).
- `gradcheck.py`   ← FPSA implicit-gradient validation (FD/BPTT/Neumann/dense).
- `plot_*.py`      ← figure generators for each experiment.

### Tests (`tests/`)
- `test_gate2_verify.py` ← adversarial: augmentation byte-equivalence vs THEIR
  code, real-builder→loader, RoPE2D translation invariance, EMA, resume
  determinism.  `test_gate2.py` ← unit gradient/shape checks.

### Docs (`docs/`)
- `GOAL.md`               ← pinned goal, hypotheses H1/H2, gate criteria.
- `ACE_PROPOSAL.md`       ← ACE architecture + theorems T1–T5 (proofs).
- `ACE_VALIDATION.md`     ← ACE V1–V4 results (incl. the certified-vs-relaxed
  finding and the σ_max-vs-spectral-radius correction).
- `ARCHITECTURE.md`       ← FPSA record + the bug registry (spectral-norm,
  residual double-count §8, memory leaks).
- `RUNBOOK_TRM_SUBSTRATE.md` ← current experiment matrix (Sudoku+Maze+H1).
- `RUNBOOK_GATE2.md`      ← original single-block Gate 2 runbook.

### `runs/`  ← CSV+JSON logs for every experiment referenced in docs
(checkpoints are gitignored; the numbers are all reproducible from configs).

## What is validated vs pending (be honest with yourself here)
VALIDATED (CPU): all gradient paths (gradcheck), ACE V1–V4, parity learning
for ace_relaxed and (post-port) fixed_ffn, freezing, certificates,
maze/sudoku plumbing + metrics.
PENDING (needs A100): every benchmark-scale number. No Sudoku/Maze accuracy
claim exists yet. Start with G-A in RUNBOOK_TRM_SUBSTRATE.md (parity H1
shakeout), STOP, review logs, then scale.

## Two open TODOs called out in the code/docs
- auto-fallback to bptt_k on any step with adjoint_tail > 1 (5-line change,
  makes neumann gradients valid-by-construction step-wise).
- fpsa-family arms: jac-reg/guard now ported, but only 1-seed/480-step
  parity evidence — the 3-seed shakeout is the arbiter.

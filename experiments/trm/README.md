# Implicit TRM: Fixed-Point Recursive Reasoning with Implicit Differentiation

This experiment applies the FPSA paper's implicit-differentiation machinery
(contractive fixed-point iteration + Neumann-series adjoint, Appendices B–D)
to **TRM** ([Jolicoeur-Martineau, 2025](https://arxiv.org/abs/2510.04871)),
benchmarked on **Sudoku-Extreme** and **Maze-Hard**.

## Motivation

| Model | Recursion | Gradients | Memory in inner depth K |
|---|---|---|---|
| HRM | fixed cycles | 1-step IFT approximation (unjustified) | O(1) |
| TRM | fixed cycles | backprop through last cycle (n+1 = 7 apps) | O(n) |
| FPRM | fixed-point + residual halting | truncated BPTT (6 steps, biased) | O(n_backwards) |
| **Implicit TRM (ours)** | fixed-point + residual halting | **exact implicit gradient (Neumann adjoint)** | **O(1)** |

TRM dropped HRM's fixed-point justification and pays O(n) memory to backprop
through the recursion. [FPRM](https://arxiv.org/abs/2606.18206) restored
fixed-point *halting* but still trains with biased truncated backprop. The
Implicit TRM closes the loop: the latent recursion is a genuine equilibrium
solve — `z* = f(z*)`, `f(z) = net(z, y + x)` — trained at the fixed point
with the implicit function theorem. The adjoint equation
`λ = J_fᵀ λ + ∂L/∂z*` is solved by truncated Neumann iteration, so training
memory is constant in the number of reasoning iterations: **inner depth
16 → 64+ costs no extra memory**, where TRM's graph grows linearly and OOMs.

## What makes the map contractive

A residual block `h ← h + f(h)` has an identity path in its Jacobian, so the
raw TRM block can never contract (verified in `tests/test_implicit_trm.py`).
The Implicit TRM block therefore uses:

1. **Learnable residual scaling** (from FPRM): `h ← α₁h + β₁·out` at each
   residual junction and `h ← α₂z + β₂·injection` at the loop input, with
   betas weight-tied so the identity path has gain `α₂·α₁^(2L) < 1`.
2. **Pre-norm RMS** blocks (FPSA Appendix B assumption; FPRM found the same).
3. **Damped iteration with patience-based step-size decay** (FPRM's FPOPT):
   `z ← (1−s)z + s·f(z)`, decaying `s` per sample when the residual stalls.
4. Optional **spectral norm** on all linear maps and optional **Jacobian
   regularization** (finite-difference `‖J v‖²` penalty) — ablation flags.

Gradient correctness under damping: the adjoint runs on the damped map `f_s`.
Since `(I − J_{f_s}) = S(I − J_f)`, the composed phantom-step/adjoint solve
recovers exactly `gᵀ(I − J_f)⁻¹ ∂f/∂θ` — the true implicit gradient — while
the Neumann series converges whenever the damped forward iteration does.
This identity is verified numerically in
`tests/test_implicit_trm.py::test_damped_map_gradient_identity`.

Everything else follows the TRM protocol: deep supervision with ACT
(Q-learning halting, up to 16 supervision steps), EMA 0.999, stablemax
cross-entropy, global batch 768, 1k puzzles × 1000 augmentations.

Known divergence from the TRM reference: the puzzle embedding is a plain
learned prefix parameter (its own optimizer group at `puzzle_emb_lr`) instead
of `CastedSparseEmbedding` + sign-SGD — identical expressivity for
Sudoku/Maze, which have a single puzzle identifier.

## Layout

```
models/layers.py        transformer primitives (TRM reference ports)
models/trm.py           TRM baseline + ACT wrapper + shared embedding base
models/implicit_trm.py  the Implicit TRM (equilibrium solve + implicit grads)
losses.py               stablemax CE + ACT loss head
data/                   dataset builders (HF download) + loader
train.py                training harness (single GPU, optional grad accum)
configs/                YAML configs (full protocol, pilots, CPU smoke)
run_*.sh                turnkey benchmark scripts
summarize_results.py    results table from run logs
```

The generic fixed-point solver and Neumann adjoint live in `src/implicit.py`
at the repo root (shared with the FPSA spatial experiments); correctness tests
are in `tests/` (exact dense-IFT comparisons, contraction spectral radii,
end-to-end training).

## Running

**CPU smoke test** (no downloads, ~3 min):

```bash
python data/build_synthetic_sudoku.py --output-dir data/sudoku-synthetic-small
python train.py --config configs/smoke_itrm.yaml
python train.py --config configs/smoke_trm.yaml
```

**Sudoku-Extreme** (1× L40S/A100-40GB; ~2h pilot, ~20h full):

```bash
PILOT=1 ./run_sudoku_benchmark.sh   # 10% budget, ranks variants
./run_sudoku_benchmark.sh           # full protocol
SEEDS="0 1 2" ./run_sudoku_benchmark.sh
```

**Maze-Hard 30×30** (80GB GPU recommended; configs use gradient accumulation):

```bash
./run_maze_benchmark.sh
```

**Ablations** (gradient estimator, inner-depth scaling, contractivity):

```bash
./run_ablations.sh
```

Logs stream to `results/<run>/log.jsonl` (JSON lines: train metrics, eval
exact-accuracy, inner iterations, peak GPU memory). Summarize with:

```bash
python summarize_results.py --results-dir results
```

## Reference numbers to compare against

| Model | Sudoku-Extreme | Maze-Hard | Source |
|---|---|---|---|
| TRM-Att (7M) | ~75% | 85.3% | TRM paper/repo |
| TRM-MLP (5M) | ~87% | — | TRM paper/repo |
| FPRM | > TRM at lower cost | > TRM | FPRM paper (2606.18206) |

Targets: reproduce the TRM baseline within ±2%; Implicit TRM ≥ baseline at
matched compute with flat memory vs inner depth and adaptive per-puzzle
iteration counts (visible in `eval/inner_iters_per_step`).

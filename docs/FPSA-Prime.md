# FPSA-Prime v0

FPSA-Prime is a clean implementation of **attention-level recurrence without a
recurrent Transformer block**. The token encoder and output decoder execute
once. The only state driven to equilibrium is a residual attention scratchpad.

This package is intentionally separate from `src/fpsa_r`. The existing package
remains the block-DEQ baseline; `src/fpsa_prime` tests the stronger causal claim
that recursive attention itself can supply reasoning depth.

## 1. Fixed-point equation

Let `A` be the immutable anchor memory produced by the one-time encoder, and let
`R` be the recurrent residual scratchpad:

```text
S(R) = RMSNorm(A + R)
Q(R), K(R) = per-head normalized projections of S(R)
V_e = W_Ve A                 # frozen evidence values
V_s(R) = W_Vs S(R)           # writable scratch values
Phi(R; A) = W_O Attention(Q(R), K(R), concat(V_e, V_s(R)))
R* = Phi(R*; A)
```

Evidence heads preserve the original FPSA asymmetry: Q and K evolve while their
values remain tied to the input memory. Scratch heads add a controlled dynamic
value channel so the recurrent attention operator can construct intermediate
content without recurring an MLP or convolution.

The readout runs once:

```text
H = A_encoded + g R*
H = H + SwiGLU(RMSNorm(H))    # optional, one-time
logits = Head(RMSNorm(H))
```

The parameter-matched hero uses a one-time **input MLP only**. The old two-MLP
configuration remains available as an explicit capacity control rather than the
default.

## 2. Forward semantics are separate from gradient semantics

The original v0 field `grad_mode` overloaded two different decisions. In
particular, its BPTT arm trained on a finite damped Picard trajectory but
silently switched to an Anderson equilibrium during evaluation. The corrected
configuration separates:

```text
forward_mode  = equilibrium | fixed_unroll
backward_mode = implicit | bptt | one_step
```

Valid combinations are:

- `equilibrium + implicit`: solve `R = Phi(R)` and use the implicit adjoint;
- `equilibrium + one_step`: same numerical fixed point, one-step phantom gradient;
- `fixed_unroll + bptt`: exactly T damped attention updates in both train and eval.

The architectural fixed-point equation is always `R = Phi(R; A)`. Numerical
damping is applied only by the solver:

```text
R_{t+1} = R_t + eta (Phi(R_t; A) - R_t)
```

Changing `eta` changes the path and convergence speed, not the fixed point. The
implementation measures the true residual `||Phi(R) - R||`, not the damped step
difference.

The default equilibrium forward method is per-sample Anderson acceleration.
Each sample gets its own mixing coefficients, so an example's solution does not
depend on which other examples share its batch. Damped Picard iteration is also
available as a reference solver.

## 3. Robust implicit backward

After the no-grad forward solver returns `R*`, the backward pass solves

```text
(I - J_Phi(R*)^T) lambda = dL/dR*
```

with restarted GMRES and sends `lambda` through one differentiable evaluation
of `Phi`. The numerical forward value remains exactly the solver output rather
than silently replacing it with one additional map evaluation.

The original batched GMRES built a complete restart-width basis before checking
convergence. In float32 it could reach a good solution at a short Krylov depth,
continue expanding an ill-conditioned basis, and return a much worse result.
The corrected solver now uses:

- incremental Givens rotations and a residual estimate after every Arnoldi column;
- per-sample early stopping and per-sample happy breakdown;
- two-pass modified Gram-Schmidt reorthogonalisation;
- float64 scalar reductions and small Hessenberg solves;
- true linear-system residual verification at every restart;
- independent Krylov state for every batch item while retaining one batched VJP
  per global Arnoldi column.

This gives activation memory that is constant in the number of equilibrium
iterations. The gradient is an equilibrium gradient only when both the forward
fixed-point solve and the adjoint solve meet their tolerances and `I - J_Phi` is
nonsingular. Both checks remain fail-fast by default through
`require_convergence=True` and `require_backward_convergence=True`.

## 4. Soft local stability control

Hard spectral caps on Q/K/O can remove useful attention capacity. FPSA-Prime
instead exposes a one-sided local penalty on the estimated largest singular
value of the recurrent Jacobian:

```text
L_stability = relu(sigma_max(dPhi/dR at R*) - target)^2
```

The right singular-vector probe is estimated with power iteration on `J^T J`.
Probe updates are detached, and the final finite-difference JVP remains
differentiable with respect to recurrent attention parameters. The default
training launcher uses target `0.95`; the model configuration keeps the weight
explicit so ablations can disable or sweep it.

## 5. Structural reasoning support

`src/fpsa_prime/structures.py` provides:

- Sudoku row, column, box, and overlapping row-box/column-box relations;
- directional Maze relations (`up`, `down`, `left`, `right`);
- local-head masks with a configurable number of unrestricted global heads;
- padding helpers for global hypothesis or constraint slots.

RoPE is applied to Q and K on every recurrent attention evaluation. Additive
structural masks and learned relation biases are both supported.

## 6. Attractor and verifier support

`src/fpsa_prime/losses.py` includes:

- a differentiable Sudoku constraint energy;
- an exact discrete Sudoku violation counter for particle selection;
- a positive/negative fixed-point margin loss for attractor shaping.

`forward_particles` runs independent initial residual states and selects the
lowest-energy **converged** candidate. Unconverged particles are excluded from
selection, and inference fails when no particle converges. A learned verifier
head is opt-in because it is not meaningful until trained against a
task-specific correctness target.

## 7. Presets

| Preset | One-time MLP | Recurrent values | Forward | Backward |
|---|---|---|---|---|
| `fpsa_prime` | encoder | frozen evidence + dynamic scratch | equilibrium | implicit |
| `fpsa_fixed_v` | encoder | frozen evidence only | equilibrium | implicit |
| `fpsa_dynamic_v` | encoder | dynamic scratch only | equilibrium | implicit |
| `fpsa_prime_decoder_mlp` | decoder | dual bank | equilibrium | implicit |
| `fpsa_prime_full` | encoder + decoder | dual bank | equilibrium | implicit |
| `fpsa_prime_bptt` | encoder | dual bank | fixed unroll | BPTT |
| `fpsa_prime_one_step` | encoder | dual bank | equilibrium | one-step |

At Maze-7 dimensions (`d=128`, eight heads, seven relation types), the matched
hero has 134,601 parameters versus 135,558 for the existing block-DEQ control.
The explicit two-MLP capacity control has 202,185 parameters.

## 8. Correctness tests

Run:

```bash
python -m pytest tests/test_fpsa_prime_*.py -q
```

The current **32 tests** include:

1. pure attention-only recurrence and all value-bank presets;
2. frozen evidence values and dynamic scratch values;
3. damping-invariant fixed points and best-residual solver return values;
4. long-restart GMRES against direct batched solves;
5. larger GMRES budgets preserving already-converged solutions;
6. long batched and per-sample GMRES agreement;
7. strict forward and backward convergence defaults;
8. implicit gradients against an analytic linear solution and deep unrolling;
9. identical fixed-unroll train/eval semantics for the BPTT control;
10. parameter matching and explicit capacity controls;
11. local Jacobian spectral-penalty accuracy on a known linear map;
12. constant saved-activation size as equilibrium depth grows;
13. Sudoku/Maze relations, masks, global slots, particle selection, and finite training.

## 9. Starter experiments

A single FPSA-Prime run:

```bash
python -m experiments.fpsa_prime.train \
  --task maze \
  --maze_size 7 \
  --extra_sizes 9 11 \
  --device cuda \
  --steps 1200 \
  --max_iter 16 \
  --result_json results/fpsa_prime/maze7_s0.json
```

`max_iter_eval` defaults to `max_iter`, so the primary result does not silently
change its forward budget. Set it explicitly only for a labeled test-time
compute-scaling experiment.

## 10. Controlled comparison harness

Use `experiments/fpsa_prime/controlled_compare.py` for both model families. It
holds the data, optimizer, LR schedule, loss, batching, seeds, evaluation, and
JSON schema fixed.

FPSA-Prime:

```bash
python -m experiments.fpsa_prime.controlled_compare \
  --family prime \
  --arch fpsa_prime \
  --task maze \
  --maze_size 7 \
  --extra_sizes 9 11 \
  --device cuda \
  --seed 0 \
  --output results/controlled/prime_maze7_s0.json
```

Block-DEQ control:

```bash
python -m experiments.fpsa_prime.controlled_compare \
  --family block \
  --arch deq_block \
  --task maze \
  --maze_size 7 \
  --extra_sizes 9 11 \
  --device cuda \
  --seed 0 \
  --output results/controlled/deq_maze7_s0.json
```

The harness records parameters, exact match, token accuracy, residual, forward
iterations/function evaluations, backward iterations/residual, elapsed time,
and larger-grid extrapolation. No benchmark-superiority claim is made before
multi-seed runs from this common path are complete.

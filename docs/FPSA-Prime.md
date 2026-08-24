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

## 2. Architecture versus numerical solver

The architectural equation is always `R = Phi(R; A)`. Numerical damping is
applied only by the solver:

```text
R_{t+1} = R_t + eta (Phi(R_t; A) - R_t)
```

Changing `eta` therefore changes the path and convergence speed, but not the
fixed point. The implementation measures the true residual
`||Phi(R) - R||`, not the damped step difference.

The default forward method is per-sample Anderson acceleration. Each sample gets
its own mixing coefficients, so an example's solution does not depend on which
other examples share its batch. Damped Picard iteration is also available as a
reference solver.

## 3. Implicit backward

After the no-grad forward solver returns `R*`, the backward pass solves

```text
(I - J_Phi(R*)^T) lambda = dL/dR*
```

with restarted GMRES and sends `lambda` through one differentiable evaluation
of `Phi`. The numerical forward value remains exactly the solver output rather
than silently replacing it with one additional map evaluation.

This gives activation memory that is constant in the number of equilibrium
iterations. The gradient is an equilibrium gradient only when both the forward
fixed-point solve and the adjoint solve meet their tolerances and
`I - J_Phi` is nonsingular. Both checks are fail-fast by default through
`require_convergence=True` and `require_backward_convergence=True`. The starter
trainer exposes explicit diagnostic-only escape hatches:
`--allow_nonconvergence` and `--allow_inexact_backward`.

## 4. Structural reasoning support

`src/fpsa_prime/structures.py` provides:

- Sudoku row, column, box, and overlapping row-box/column-box relations;
- directional Maze relations (`up`, `down`, `left`, `right`);
- local-head masks with a configurable number of unrestricted global heads;
- padding helpers for global hypothesis or constraint slots.

RoPE is applied to Q and K on every recurrent attention evaluation. Additive
structural masks and learned relation biases are both supported.

## 5. Attractor and verifier support

`src/fpsa_prime/losses.py` includes:

- a differentiable Sudoku constraint energy;
- an exact discrete Sudoku violation counter for particle selection;
- a positive/negative fixed-point margin loss for attractor shaping.

`forward_particles` runs independent initial residual states and selects the
lowest-energy **converged** candidate. Unconverged particles are excluded from
selection, and inference fails when no particle converges. A learned verifier
head is opt-in because it is not meaningful until trained against a
task-specific correctness target.

## 6. Presets

| Preset | Recurrent values | Gradient |
|---|---|---|
| `fpsa_prime` | frozen evidence + dynamic scratch | implicit |
| `fpsa_fixed_v` | frozen evidence only | implicit |
| `fpsa_dynamic_v` | dynamic scratch only | implicit |
| `fpsa_prime_bptt` | dual bank | full unroll |
| `fpsa_prime_one_step` | dual bank | one-step phantom gradient |

## 7. Correctness tests

Run:

```bash
python -m pytest tests/test_fpsa_prime_*.py -q
```

The 24 tests cover:

1. pure attention-only recurrence and all value-bank presets;
2. fixed evidence values and dynamic scratch values;
3. damping-invariant fixed points and best-residual solver return values;
4. per-sample Anderson and per-sample truncated GMRES independence;
5. strict forward and backward convergence defaults;
6. implicit gradients against an analytic linear solution and deep unrolling;
7. exact numerical fixed-point forward values and the one-step gradient ablation;
8. constant saved-activation size as equilibrium depth grows;
9. Sudoku/Maze relations, masks, global slots, and differentiable relation bias;
10. verifier behavior, particle-energy requirements, and converged-only selection;
11. deterministic evaluation, invalid-configuration rejection, and finite training.

## 8. Starter experiment

From the repository root:

```bash
python -m experiments.fpsa_prime.train \
  --task sudoku \
  --device auto \
  --steps 2000 \
  --max_iter 24 \
  --max_iter_eval 64
```

Maze:

```bash
python -m experiments.fpsa_prime.train \
  --task maze \
  --maze_size 9 \
  --device auto
```

The launcher reports exact match, token accuracy, mean architectural residual,
mean solver iterations, and converged-sample fraction. Forward and implicit
backward convergence are fail-fast by default. The diagnostic flags
`--allow_nonconvergence` and `--allow_inexact_backward` must be supplied
explicitly to bypass those checks.

## 9. What v0 does not claim

This implementation does not yet claim benchmark superiority. The decisive
experiment must compare `src/fpsa_prime` against `src/fpsa_r` and other recurrent
baselines with matched:

- parameter count;
- attention and FFN FLOPs;
- optimizer steps and tokens seen;
- forward function evaluations;
- wall-clock latency and peak memory;
- solver tolerance and non-convergence policy.

The next implementation stage should add a full attractor-shaping curriculum,
Maze verification energy, adaptive particle allocation, ARC object/hypothesis
slots, and benchmark launchers with multi-seed reporting.

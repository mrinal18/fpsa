# GOAL (pinned by Mrinal)

## The research question
Can a transformer "think for a variable amount of time" by running one
weight-tied block to a fixed point z* = f(z*, x) — and does solving that
fixed point PROPERLY (implicit differentiation, a real convergence
certificate) buy anything over unrolling the loop a fixed number of steps?

Everything in this repo is machinery in service of answering that honestly.

## The two hypotheses under test
H1 (implicit vs unrolled): at matched compute, training through the
    equilibrium (Neumann implicit gradients) is at least as accurate as
    unrolled/truncated BPTT while providing O(1)-memory backward and
    better test-time iteration scaling / input-adaptive depth.
    Honest prior: FPRM's 94.2% Sudoku with truncated BPTT is evidence that
    accuracy alone may not separate; the differentiators are memory,
    certificates, and adaptivity — those must be measured, not assumed.
H2 (certification/anchoring): a map contractive BY CONSTRUCTION (ACE)
    matches a regularizer-stabilized expressive map (FPSA) without any
    stability machinery, while making the residual a rigorous distance
    bound and enabling per-token freezing with a perturbation bound.
    Parity evidence: supported (ACE-relaxed ≈ stabilized FPSA, zero
    collapses). Certified variant blocked on a tighter attention bound.

## Falsifiable success criteria per gate
- Gate 2 (Sudoku-Extreme, exact HRM/TRM protocol): within striking
  distance of FPRM (94.2%) / TRM (~87%); ACE arm stable with zero
  guard/jac-reg; convergence logged every step.
- Gate 3 (matched-budget baselines): neumann vs bptt vs phantom1 on the
  SAME architecture decides H1; unrolled fixed-K baseline (a) is the
  control — NOTE: the prototype's Colab ablation accidentally trained
  exactly this control while claiming the method; the bench must never
  repeat that (backward mode is stamped in every config; all 10
  committed configs are `backward: neumann`).
- Adaptivity evidence: iterations-to-tol vs difficulty correlation
  (Sudoku givens), freezing compute savings at fixed accuracy,
  certificate calibration (bound validity + tightness).

## Non-goals / discipline
No claimed number without a committed config + command. No advancing past
a STOP gate without review. No relabeling accidents as methods (see
docs/ARCHITECTURE.md §8 and the residual-convention guard). Comparisons
against FPRM/TRM only under their exact data protocol.

## Standing next action
Run the two Gate 2 pilots on A100 (configs/gate2_sudoku_pilot.yaml,
configs/gate2_sudoku_ace_pilot.yaml), paste train_log.csv back for
review before committing the full 3-seed budget.

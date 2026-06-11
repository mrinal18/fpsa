# FPSA — architecture & engineering record

Companion to the diagram in `docs/fpsa_architecture.svg`. Everything in this
document is either implemented and validated in this repo or explicitly
marked OPEN / PROPOSED.

## 1. The map

One weight-tied block f, iterated to (approximate) equilibrium:

    z*  =  f(z*, x),     f(z, x) = damp( x + W_O · Attn(Q(z), K(z)) · V  [+ FFN] )

- Q, K from LN(z) (pre-norm), spectral-normalized projections, RoPE, learned
  per-head temperature tau.
- V anchored to the input: V = W_V · LN(x), computed once per solve
  (value modes: fixed | evolving | blended | fixed_ffn; see §5).
- Input injection: "+ x" every iteration (DEQ-style anchoring).
- fixed_ffn adds a shared in-loop FFN (LN -> GELU) after the attention update.
- Damping: z_{k+1} = (1-a) z_k + a f_raw(z_k), a = 0.5. Same fixed point;
  Jacobian (1-a)I + aJ tames rotation/oscillation.
- Non-causal (bidirectional) attention: the equilibrium is a JOINT
  constraint-satisfaction over all positions, not left-to-right generation.

## 2. Solver (forward)

Damped Picard with per-token relative residual r_i = ||dz_i||/||z_i||;
stop when max_i r_i < tol or k = K. Per-iteration residual curves
(mean/max), iterations, converged-fraction are ALWAYS logged (SolveStats).
A run that doesn't log convergence is not a valid run.

Engineering invariants:
- spectral_norm power iteration is FROZEN within a solve
  (parametrize.cached()) so f does not mutate between iterations.
- Parametrization caches are populated WITH grad enabled before the
  no-grad solve (else implicit backward silently returns zero gradient
  for spectral-normed weights — verified by exact dense adjoint).

## 3. Backward

Modes (all validated against finite differences / full BPTT in float64):
- neumann (default): phantom gradient, lam <- J^T lam + g, truncated
  (I - J^T)^{-1} g via VJPs on a SEPARATE graph from the hooked output
  (same-graph hook recurses -> OOM). Error vs BPTT decays geometrically
  at the contraction rate; 2.9e-5 at k>=10 on the gradcheck instance.
- phantom1: HRM-style 1-step gradient. Measured bias ~15% rel-err,
  cos 0.989. Cheap, biased; kept as ablation axis.
- bptt: full unroll. Reference, O(K) memory.

## 4. Stability stack (in causal order of discovery)

1. Jacobian regularization: differentiable sigma_max(J at z*) via
   power-iteration VJPs (create_graph on last step);
   penalty w * relu(sigma - rho)^2, rho=0.85, w=1.0.
   Without it, training drives f expansive by step ~200 and the Neumann
   series diverges (Gate-1 diagnostic run 1).
2. Solver budget: K must reach tol at trained sigma (K=24 for sigma~0.7-0.85).
   K=12 left exit residuals ~0.2 -> noisy gradients, oscillation
   (diagnostic run 2).
3. Divergence guard v2: when exit residual > 0.15, the step applies ONLY
   (jac penalty + differentiable exit-residual penalty). v1 (penalty-only)
   deadlocks when the sigma estimate under-reads (2 power iters is a lower
   bound) — observed on evolving and blended seed 1. v2 uses 4 power iters
   + the residual escape; it rescued evolving from 60% -> 100%.
4. Memory: stats-held graph tensors (jac_sigma_est, exit_residual_t) form
   a reference cycle with the backward-hook closure -> ~20-30 MB/step leak.
   Cleared after opt.step() + periodic gc.collect().

## 5. Validated results (prefix parity L=16, disjoint train/test)

Gate 1 (fixed_ffn, no guard, 3 seeds): token 99.99 +/- 0.01,
seq 99.91 +/- 0.05. Trained maps sigma ~0.54-0.66.

Value-mode ablation (guard era; 1-2 seeds; param counts differ by design):
fixed_ffn 33.6K: ~100% (s0,s1) | evolving 16.9K (guard v2): ~100% (s0,s1)
blended 25.2K: 100% (s0), 76.9% (s1, v1 deadlock) | fixed 16.9K: 54.6% (fails)
SN-bug repro (prototype behavior): fatal — 57.8% vs 99.99% control.

Length-gen probe (trained L=16): anytime property holds in-distribution
(81% @k=8 -> 100% @k=24, monotone); OOD lengths mostly fail
(L=20: 73-80%, L=24: 52-60%); extra iterations don't rescue OOD.

## 6. Open issues

- Plateau: trained equilibria stall at residual 3e-4..1e-2 > tol for 2/3
  seeds; accuracy is 100% and budget-insensitive at the plateau, but the
  tol-based early exit never fires -> the "certificate" is currently
  nominal, not operational.
- Collapse-and-recover episodes between evals (2/3 ungated seeds);
  guard v2 mitigates, root-cause sigma excursions remain.
- Length OOD failure: single-length training learns position-specific
  routing, not a length-invariant algorithm.
- Cross-mode ablation confounded by guard version; rerun grid under v2.

## 7. Comparison (HRM / TRM / FPSA) — annotated honestly

Iterated map: HRM two coupled RNN modules | TRM tiny 2-layer net over (z, y)
| FPSA one attention block, Q/K from LN(z), V anchored to x.
Solve: HRM fixed unroll | TRM fixed recursion | FPSA damped Picard to
measured tolerance (CAVEAT: tol currently above trained plateau).
Backward: HRM 1-step (~15% bias, measured) | TRM full BPTT (unbiased, O(K)
mem) | FPSA Neumann implicit (unbiased at equilibrium, O(1) mem).
Halting: HRM learned Q-head | TRM fixed | FPSA residual certificate
(operational only after plateau fix, §8.1).
Stability: HRM/TRM none needed | FPSA SN + damping + jac-reg + guard
(the tax we pay for the equilibrium).
Deep supervision: HRM/TRM yes (large practical win) | FPSA not yet (§8.3).

## 8. PROPOSED enhancements (not implemented)

8.1 Residual shaping: promote the differentiable exit-residual penalty
    from guard-escape to always-on small weight -> push equilibria below
    tol, restore early exit, make the certificate real.
8.2 Per-token freezing: tokens with r_i < tol stop updating and become
    static context (true token-level adaptive depth + compute savings).
8.3 Equilibrium deep supervision: supervise intermediate solver
    checkpoints (k=8,16,exit) with phantom gradients, Neumann at exit;
    imports HRM/TRM's main practical advantage and trains the anytime
    property directly.
8.4 Anderson acceleration: tolerate sigma closer to 1 (expressivity vs
    contraction tension measured in the fixed-mode failure).
8.5 Answer-coupled equilibrium: joint fixed point (z*, y*) with a slower
    damped answer track — TRM's z/y split in equilibrium form.
8.6 Path independence: randomize z0 in {x, 0, noise} during training
    (Anil et al. 2022) — targets test-time iteration scaling.
8.7 Mixed-length / mixed-difficulty curriculum: prerequisite for any OOD
    extrapolation claim (length probe failure).
8.8 Input-conditioned damping a(x) per token: easy tokens converge fast,
    hard tokens take careful steps.

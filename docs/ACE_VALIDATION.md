# ACE validation report (V1-V4) — parity testbed, CPU, seed 0

## Models
- ACE-v0 (certified): nonexpansive-by-construction T (derived attention
  bound L <= sqrt(N) R ||V||2 / c with exact SVD norms; GroupSort FFN;
  convex residuals; ball projection) + learned per-token anchor,
  beta in [0.2, 0.8] => a-priori rho = 0.8.
- ACE-relaxed (ablation): identical EXCEPT learnable attention temperature
  (certified bound not enforced). No a-priori guarantee.
Both: NO Jacobian regularization, NO divergence guard, NO damping knob.

## Results
V1 gradcheck (float64): PASS. FD vs BPTT exact under hybrid criterion
  (initial "failure" was FD roundoff on ~1e-10 q/k gradients — itself
  diagnostic of the certified temperature crushing routing gradients).
  Neumann vs BPTT: 6.1e-6 rel err at k=5, cos 1.000000.
V2 constraint audit: PASS for v0. Worst sigma(J at z*) = 0.155 untrained
  (bound 0.8); never exceeded 0.29 across all of training. Zero collapses,
  zero interventions, every solve converged.
V3 parity:
  - v0 (certified): FAIL on task — 52.9% (chance) after 3000 steps.
    Mechanism: conservative constant => near-uniform attention + ~1e-7
    q/k gradients => routing frozen. The theory held; the bound is too
    loose to leave usable cross-token communication.
  - relaxed: 99.74% tok / 98.70% seq, ZERO stability machinery, zero
    collapses — while sigma_max(J) repeatedly exceeded 1 (max 1.19).
    Reconciliation: sigma_max is the largest SINGULAR value; convergence
    needs spectral RADIUS < 1; non-normal J permits ||J||>1 with rho(J)<1.
    Anchored map rode the boundary where ungated FPSA collapsed
    permanently and gated FPSA collapsed transiently (2/3 seeds).
Certificate calibration (T2):
  - v0 a-priori bound (rho=0.8): 0/38 violations over 32 inputs x 38
    iterations, median tightness 4.7x. THEOREM EMPIRICALLY VALID + tight.
  - relaxed: lam_hat = 1.15 > 1 => certificate vacuous (as theory says:
    no contraction guarantee, no bound).
V4 token freezing (relaxed model): PASS.
  tol_f = 1e-3: 100.0% prediction agreement with the full solve at 67%
  of the token-iteration budget (33% compute saved, zero accuracy cost);
  3e-4: 100% @ 73%; 3e-3 too aggressive (96% agreement, -3.4pp).
  Note: the freezing perturbation THEOREM applies only under certified
  rho < 1; on the relaxed model this is an empirical result.

## Verdict
The anchored-equilibrium framework is implemented correctly (V1), its
theorems hold exactly where their assumptions hold (V2, calibration), and
per-token freezing delivers real adaptive compute (V4). The single failure
is the v0 attention bound's looseness (V3-certified), cleanly isolated by
the relaxed ablation, which matches stabilized FPSA's accuracy with zero
stability machinery.

## Next-iteration options
1. Tighter certified attention bound (Kim et al. 2021 L2-attention
   constant, implemented from the paper) => recover a-priori guarantees
   without the expressivity cliff. The principled path.
2. ACE-relaxed as a practical variant for Gate 2: anchor as structural
   stabilizer, a-posteriori sigma monitoring, optionally reinstating the
   sigma penalty only as telemetry-triggered insurance.
3. Hybrid: certified FFN/residual/projection + monitored attention —
   T2/T5 hold conditionally on per-solve measured contraction.

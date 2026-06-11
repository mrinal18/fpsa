# ACE: Anchored Contractive Equilibrium transformer — design proposal

Status: PROPOSED (no code, no results). Successor candidate to FPSA,
motivated by the measured failure modes in docs/ARCHITECTURE.md §4-6.

## 0. Scope of claims

Provable: existence/uniqueness of equilibrium, geometric convergence with
known rate, a-posteriori error certificate, global validity of implicit
gradients with truncation bounds, anytime monotonicity, bounded-error
token freezing. NOT provable: benchmark superiority (empirical, gated).
Novelty: synthesis to be checked against literature before any claim.

## 1. Definition

Iteration:  z_{k+1} = B(x) x + (I - B(x)) T(z_k; x)
- B(x) = diag(beta_i(x) I_d), beta_i = beta_min + (beta_max-beta_min)*
  sigmoid(MLP(x_i, mean(x))), with 0 < beta_min <= beta_i <= beta_max < 1.
  beta depends on x ONLY (never z) — required by all proofs below.
- T nonexpansive by construction (||T(z)-T(z')|| <= ||z-z'||):
  a) L2-distance multi-head attention (Kim/Papamakarios/Mnih 2021: standard
     dot-product attention is provably NOT Lipschitz; L2 attention is, with
     a computable constant c(N, W)). Divide attention output by c at
     runtime => 1-Lipschitz. Alternative: LipsFormer scaled cosine attention.
     RoPE: orthogonal rotations, exactly 1-Lipschitz.
  b) Convex residuals: y = (1-gamma) z + gamma h(z), gamma in [0,1]
     (learned, x-conditioned allowed). Convex combinations of nonexpansive
     maps are nonexpansive.
  c) FFN: spectral-normalized linears + GroupSort. Norm-constrained
     GroupSort nets are universal approximators of Lipschitz functions
     (Anil, Lucas, Grosse 2019) — the expressivity theorem.
  d) In-loop normalization: ball projection P_{||z||<=R} (firmly
     nonexpansive). NO LayerNorm inside the loop (unbounded Jacobian).
- Unconstrained encoder (tokens -> x) before the loop; unconstrained
  LN + head after the loop. Constraints apply only to the iterated map.

## 2. Theorems

T1 (Banach). ||F(z)-F(z')|| = ||(I-B)(T(z)-T(z'))||
   <= max_i (1-beta_i) ||z-z'|| <= rho ||z-z'||, rho := 1-beta_min < 1.
   Unique z*; ||z_k - z*|| <= rho^k ||z_0 - z*||.

T2 (a-posteriori certificate). ||z_k - z*|| <= ||z_{k+1}-z_k|| / (1-rho).
   Proof: telescope z* - z_k = sum_{m>=0} (z_{k+m+1} - z_{k+m}), each term
   <= rho^m ||z_{k+1}-z_k||. The observed one-step residual is a rigorous
   distance bound with KNOWN constant — operational stopping rule.

T3 (certified gradients). ||dF/dz|| <= rho everywhere (T differentiable
   a.e.; GroupSort piecewise-linear). (I-J) globally invertible,
   ||(I-J)^{-1}|| <= 1/(1-rho); k-term Neumann truncation error
   <= rho^{k+1} ||g|| / (1-rho). Combined gradient-error bound =
   truncation term + inexact-solve term (Lipschitz-of-VJP x certificate).

T4 (anytime). Distance contracts monotonically each iteration; with
   L_h-Lipschitz head, output error <= L_h rho^k ||z_0 - z*||.

T5 (freezing perturbation). Freeze coordinate set S at values within
   delta of z*_S. The reduced map G(z_{S^c}) = F(frozen, z_{S^c})_{S^c}
   is rho-Lipschitz; its fixed point z~ satisfies
   ||z~ - z*_{S^c}|| <= rho delta / (1-rho).
   Proof: ||z~ - z*|| <= rho (delta + ||z~ - z*||) and solve.

Interpretation of beta: token-level iterations-to-tol ~
log(1/tol) / log(1/(1-beta_i)) — the network predicts per-token thinking
time from the input, before solving.

## 3. What this provably fixes (vs measured FPSA failures)

- sigma > 1 excursions / collapses / guard machinery: impossible by T1.
- Nominal certificate (plateau above tol): T2 makes residual a true bound;
  geometric decay to ANY tol guaranteed.
- Gradient invalidity during excursions + phantom-1 bias: T3 bounds.
- Anytime property only empirical: T4.
- Per-token adaptive depth unrealized: T5 + freezing implementation.

## 4. Honest costs / risks

- Expressivity: 1-Lipschitz-per-component maps are restrictive; known
  underfitting risk in Lipschitz-constrained nets. Mitigations:
  unconstrained encoder/head, GroupSort universality (for Lipschitz
  targets), unbounded iteration depth. MUST BE MEASURED (parity gate).
- L2-attention constant depends on N (sequence length) — tension at
  ARC scale (900 tokens). Mitigations: per-length calibration of c,
  cosine-attention variants with tighter constants. Open engineering.
- Theorems assume exact constraint enforcement; power-iteration spectral
  norm is approximate during training. Spec: exact normalization
  (SVD on d x d) at eval; PI-steps budget studied in validation gate.
- rho = 1-beta_min near 1 (expressive regime) slows convergence as
  1/log(1/rho); Anderson/Halpern acceleration compatible (KM/Halpern
  theory applies; Halpern gives O(1/k) residual for beta_k -> 0).
- Empirical superiority over TRM/HRM: NOT claimed; Gates 2-3 decide.

## 5. Validation gate (precedes any benchmark)

V1 gradcheck: FD/BPTT/Neumann agreement (reuse harness).
V2 constraint audit: numerically estimate sup_z ||J|| via power iteration
   at many (x, z) — predict <= 1-beta_min ALWAYS, all training stages.
V3 parity, 3 seeds, ZERO stability machinery (no jac-reg, no guard):
   predict zero collapses; residual slope <= log(1-beta_min);
   certificate calibration plot (true distance vs T2 bound).
V4 token freezing on: accuracy unchanged within T5 bound; compute saved
   reported per token.
Falsification of any prediction => implementation bug or spec error;
that separability is the point.

## 6. Implementation diff vs fpsa-bench

models/ace_block.py (L2/cosine attention + convex residual + GroupSort FFN
+ ball projection + beta-MLP); solver unchanged except: remove jac_reg,
remove guard, add per-token freezing path + certificate logging
(certified_distance = residual/(1-rho)). Training script: remove guard
branches. Instrumentation: keep ALL of it — it now tests theorems.

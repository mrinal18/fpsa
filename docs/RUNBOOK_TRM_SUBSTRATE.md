# TRM-substrate program: the 4-point plan -> executable experiment matrix

Model: models/trm_substrate.py (TRMSubstrate). TRM lineage kept: (z,y)
two-stream recursion, deep supervision (per-segment loss, detached y),
shared recurrent computation. Replacements: fixed-point inner solve with
residual halting (vs fixed n); certificate freezing (vs ACT head).
v0 deviations: separate z/y cores (TRM shares one net); no EMA yet.

## Validation status (CPU, this session — all committed)
- float64 gradcheck: FD == bptt_full exact (T=1); neumann_k -> bptt_full
  geometrically, 3.2e-13 at k=40; phantom1 == neumann_1 bitwise.
- 15-combination smoke (5 value modes x 3 backward modes): pass.
- ace_relaxed arm (ported validated ACEBlock): stable + learning on parity
  (75% tok @ 300 steps, residuals ~1e-4 throughout, zero machinery).
- Per-token certificate freezing: 100% agreement at 24-36% compute saved
  (eps in [3e-3, 1e-2]); dose-response knee at ~3e-2.
- KNOWN + EXPECTED: fpsa-family arms (fixed/fixed_ffn/evolving/blended)
  destabilize ~step 250 with zero machinery (res 0.87) — the documented
  FPSA failure mode. TODO before their A100 runs: port jac-reg + guard v2
  from models/solver.py into _solve_z, OR run them only as bptt_k arms
  (H1 controls do not require convergent solves).

## Arms (plan point 2)
A1 fixed      : frozen [x;y_t] values (safe topology)   — needs stabilizer
A2 fixed_ffn  : A1 + in-loop FFN (TRM-closest)          — needs stabilizer
A3 evolving   : live values                 RISK ARM    — needs stabilizer
A4 blended    : live gated values           RISK ARM    — tests inversion
                prediction: its unrolled-regime win should NOT survive
                enforced convergence
A5 ace_relaxed: ported ACEBlock — stable with zero machinery (shipped)

## Per-token certified adaptive computation (plan point 3)
freeze_eps > 0 enables certificate freezing (res_i/(1-lam_i) < eps).
Labeling: a-priori certified ONLY under ACE-certified (pending tighter
attention bound); ace_relaxed and others are A-POSTERIORI CALIBRATED.
Report: accuracy-vs-eps curve + token-iteration savings + (Sudoku)
freeze-maps vs givens.

## H1 controlled study (plan point 4)
Same arm, same data, same recipe, same no-grad forward solve; swap ONLY:
  backward in {bptt_k, neumann_k, phantom1} at equal k
  (k backward passes through f_z each; bptt_k pays k extra fwd
   re-materializations — report measured s/step alongside).
Primary comparison on A5 (converged solves -> implicit well-posed) and A2
(after stabilizer port). Metrics: accuracy, s/step, peak memory (the O(1)
vs O(k) claim), sensitivity to k in {1, 4, 8, 16}.
Interpretation pinned in docs/GOAL.md: tie = positive (memory at parity).

## Sequence (A100)
G-A parity shakeout: A5 x {neumann_k, bptt_k, phantom1}, k=6, 3k steps,
    3 seeds. STOP: paste logs.
G-B stabilizer port -> A2 joins; blended-inversion test (A4 under enforced
    convergence vs the unrolled Colab numbers).
G-C Sudoku-Extreme via data/sudoku.py protocol + gate2 configs adapted to
    TRMSubstrate (add EMA + their recipe). Bar: TRM ~87 / FPRM 94.2.
G-D freezing curves + freeze-maps on the G-C winner.

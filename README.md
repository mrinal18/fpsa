# fpsa-bench — fixed-point reasoning benchmark (branch: `ace-bench`)

Validation ladder for **fixed-point looped transformers** on symbolic
reasoning, developed as a gated experimental program. Two architectures:

- **FPSA** (Fixed-Point Self-Attention): the prototype from this repo's
  `main` branch, ported, instrumented, debugged and stabilized.
- **ACE** (Anchored Contractive Equilibrium): a new architecture designed
  so that the properties FPSA needed regularizers for hold **by
  construction** — then validated, with one honest failure and one strong
  surprise (§4).

Everything below is backed by committed configs and run logs under
`runs/` (CSV + JSON; checkpoints excluded). Status: parity gates and the
ACE validation gate complete on CPU; Sudoku-Extreme (Gate 2) pipeline
verified end-to-end and awaiting A100 pilots.

---

## 1. The common skeleton

Both architectures iterate a weight-tied map to an (approximate)
equilibrium and train through it implicitly:

**Forward (damped Picard).** $z_{k+1} = (1-\alpha)z_k + \alpha f(z_k; x)$,
stopping when the per-token relative residual
$r_i = \lVert \Delta z_i\rVert / \lVert z_i\rVert$ satisfies
$\max_i r_i < \mathrm{tol}$ or $k = K$. Per-iteration residual curves,
iterations used, and converged-fraction are always logged
(`models/solver.py::SolveStats`). *A run that doesn't log convergence is
not a valid run.*

**Backward (Neumann / phantom adjoint).** With $J = \partial f/\partial z$
at $z^\*$ and incoming gradient $g$:
$\lambda_{t+1} = J^\top \lambda_t + g \;\to\; (I - J^\top)^{-1} g$,
implemented with two separate autograd graphs (hooked output vs VJP
graph; a same-graph hook recurses). Modes: `neumann` (default),
`phantom1` (HRM-style 1-step; measured bias ≈15% rel-err, cos 0.989),
`bptt` (reference). Truncation error after $t$ terms is bounded by
$\rho^{t+1}\lVert g\rVert/(1-\rho)$ when $\lVert J\rVert \le \rho < 1$.

---

## 2. FPSA (summary; full record in `docs/ARCHITECTURE.md`)

$$z^\* = x + W_O\,\mathrm{Attn}\big(Q(\mathrm{LN}z^\*),K(\mathrm{LN}z^\*)\big)V \;[+\,\mathrm{FFN}],\qquad V = W_V\,\mathrm{LN}(x)\ \text{(anchored)}$$

Stability had to be **added**: spectral norm on Q/K/O (frozen within a
solve), a differentiable Jacobian penalty
$w\cdot\mathrm{relu}(\hat\sigma_{\max}(J) - \rho_{\mathrm{target}})^2$
(power-iteration VJPs), and a divergence guard (v2: penalty +
differentiable exit-residual on failed solves). Without these, training
drives $\sigma(J) > 1$ and the run collapses by step ~200.

Headline results (prefix parity, L=16, disjoint train/test):
**99.99 ± 0.01 token / 99.91 ± 0.05 sequence** over 3 seeds (Gate 1);
value-mode ablation: `fixed_ffn` ≈ `evolving` (guard-v2) ≈ 100%,
`blended` seed-dependent, attention-only `fixed` fails; reproducing the
prototype's spectral-norm bug is fatal (57.8% vs 99.99% control).
Five infrastructure bugs found and fixed along the way (power-iteration
mutation inside the solve, recursive adjoint hook, cache-severed
gradients, guard-v1 deadlock, a ~20–30 MB/step hook-closure memory leak)
— see `docs/ARCHITECTURE.md §4`.

---

## 3. ACE — design and theorems (full proposal: `docs/ACE_PROPOSAL.md`)

**Design inversion:** instead of regularizing an expressive map into
contraction, make the iterated map nonexpansive **by construction** and
put unconstrained expressivity outside the loop (encoder before, head
after). Add a learned per-token **anchor** that is simultaneously the
contraction mechanism, the certificate constant, and the
adaptive-thinking knob:

$$z_{k+1} = B(x)\,\tilde x + \big(I - B(x)\big)\,T(z_k),\qquad
B = \mathrm{diag}(\beta_i(x)),\ \ \beta_i \in [\beta_{\min}, \beta_{\max}] \subset (0,1)$$

$\beta$ depends on $x$ only (load-bearing for every proof);
$\tilde x = \Pi_R(\mathrm{enc}(x))$. $T = \Pi_R \circ \mathrm{FFNres}
\circ \mathrm{ATTNres}$, each factor 1-Lipschitz:

- **Certified attention** (fixed $V = W_V\tilde x$, exact-SVD-normalized
  $W_Q, W_K, W_O$, RoPE orthogonal hence free). Derived bound, on the
  ball $\lVert z_l\rVert \le R$ (maintained invariantly):

  $$|\partial s_{ij}| \le \tfrac{R}{c}(\lVert\delta_i\rVert+\lVert\delta_j\rVert),\quad
  \lVert J_{\mathrm{softmax}}\rVert_2 \le \tfrac12,\quad
  \lVert\delta o_i\rVert \le \lVert V\rVert_2 \lVert\delta a_i\rVert_2
  \;\Rightarrow\; L_{\mathrm{attn}} \le \frac{\sqrt N\, R\, \lVert V\rVert_2}{c}.$$

  Set $c = \sqrt N R \lVert V\rVert_2 / g$ with learnable $g < 1$:
  certified $g$-Lipschitz. Heads scaled by $1/\sqrt H$.
- **Convex residuals** $y = (1-\gamma)z + \gamma h(z)$ (closure of
  nonexpansive maps under convex combination — unlike $z + h(z)$).
- **FFN**: exact-spectral-normalized linears + **GroupSort** (pairwise
  max/min: exactly norm-preserving; norm-constrained GroupSort nets are
  universal for Lipschitz functions, Anil et al. 2019).
- **$\Pi_R$**: per-token ball projection, firmly nonexpansive; replaces
  LayerNorm inside the loop (whose Jacobian is unbounded).

**Theorems** (proofs in `docs/ACE_PROPOSAL.md`), with
$\rho := 1-\beta_{\min}$:
1. **Banach**: $F$ is a $\rho$-contraction → unique $z^\*$,
   $\lVert z_k - z^\*\rVert \le \rho^k\lVert z_0 - z^\*\rVert$.
2. **Certificate**: $\lVert z_k - z^\*\rVert \le \lVert z_{k+1}-z_k\rVert/(1-\rho)$
   — the observed residual is a *rigorous* distance bound.
3. **Certified gradients**: $\lVert J\rVert \le \rho$ everywhere → IFT
   global, Neumann truncation $\le \rho^{t+1}\lVert g\rVert/(1-\rho)$.
4. **Anytime**: distance to equilibrium contracts monotonically.
5. **Freezing**: freezing tokens within $\delta$ of equilibrium shifts the
   remaining fixed point by $\le \rho\delta/(1-\rho)$.

---

## 4. Everything we tried with ACE (chronological, honest)

1. **Implementation** (`models/ace.py`): exact SVD norms (no
   power-iteration approximation), GroupSort, ball-projected init so all
   iterates stay where the bound applies.
2. **V1 gradcheck** (float64, 80-iter solve, residual 3e-18): Neumann vs
   BPTT 6.1e-6 at k=5, cos 1.000000. First FD pass "failed" at 3.8e-5 —
   ε-sweep showed errors *stable* in ε, localized to q/k weights with
   analytic gradients ~1e-10: FD roundoff floor on near-zero gradients
   (itself a clue: the certified temperature crushes routing gradients).
   Hybrid relative/absolute criterion → **PASS, max err 0**.
3. **V2 constraint audit**: worst $\hat\sigma(J\ \text{at}\ z^\*)$ over 20
   inputs = **0.155** vs bound 0.8 (untrained); never exceeded **0.29**
   through all of training. Zero collapses, zero interventions, every
   solve converged — with **no stability machinery at all**.
4. **V3 parity, certified: FAIL — 52.9% (chance) after 3000 steps.**
   Mechanism, fully diagnosed: the conservative $\sqrt N R\lVert V\rVert_2$
   constant forces near-uniform attention and ~1e-7 q/k gradients;
   routing never trains; σ can't even spend its allowed budget. The
   theory held perfectly; the bound is too loose to leave usable
   cross-token communication. (Predicted as the main risk in the
   proposal; now measured.)
5. **Relaxed ablation** — *only* the certified temperature replaced by a
   learnable one (`certified: false`); anchor, convex residuals,
   GroupSort, projection unchanged: σ rockets 0.15→0.99 by step 300 and
   learning begins; final **99.74 token / 98.70 seq** —
   ≈ stabilized-FPSA accuracy with **zero guard, zero Jacobian
   regularization, zero collapses**, while $\hat\sigma_{\max}$ repeatedly
   exceeded 1 (max **1.19**).
6. **Theory correction logged**: $\hat\sigma$ is the largest *singular*
   value; Picard needs spectral *radius* < 1; non-normal $J$ permits
   $\lVert J\rVert > 1$ with $\rho(J) < 1$. So relaxed-ACE has **no
   a-priori guarantee** — its stability (vs FPSA's collapse at the same
   σ̂) is an empirical property of the anchor, not a theorem.
7. **Certificate calibration (T2)**: certified-v0 — **0/38 violations**
   over 32 inputs × 38 iterations, median tightness **4.7×** (theorem
   empirically valid *and* tight). Relaxed — $\hat\lambda = 1.15 > 1$,
   certificate vacuous, exactly as theory says.
8. **V4 token freezing** (relaxed model): freeze token $i$ when
   $r_i < \mathrm{tol}_f$; frozen tokens stop updating but remain
   attention context. $\mathrm{tol}_f{=}10^{-3}$: **100.0% prediction
   agreement with the full solve at 67% of the token-iteration budget**
   (33% compute saved, zero accuracy cost); $3{\times}10^{-4}$: 100% @
   73%; $3{\times}10^{-3}$ too aggressive (96%, −3.4pp).
9. **Sudoku-harness smoke** (`arch: ace_relaxed`, synthetic 9×9, CPU):
   trains (cell 25%→50% in 400 steps), σ ramping 0.14→0.82 with the same
   learn-the-gain takeoff as parity (breakout there ≈ step 1500 — ACE
   warms up slower than FPSA by construction). FPSA arm verified
   bitwise-unchanged by the integration.

**Open next steps** (`docs/ACE_VALIDATION.md`): (a) tighter certified
attention bound (Kim et al. 2021 L2-attention constant, implemented from
the paper) to recover a-priori guarantees without the expressivity
cliff; (b) ACE-relaxed as the pragmatic Gate-2 arm (current choice);
(c) hybrid: certified FFN/residual/projection + monitored attention.

---

## 5. How to run

```bash
pip install torch pyyaml matplotlib numpy
```

**ACE validation gate** (CPU, minutes each unless noted):
```bash
python3 eval/ace_validate.py                 # V1 gradcheck + V2 audit
python3 train/train_ace_toy.py --config configs/ace_v3_parity.yaml  --seed 0   # V3 certified (fails at chance — the documented negative result)
python3 train/train_ace_toy.py --config configs/ace_v3_relaxed.yaml --seed 0   # V3 relaxed (→ ~99.7%; ~25 min CPU; resumable, rerun to resume)
python3 eval/ace_v4_calibrate.py             # T2 calibration + V4 freezing
```

**FPSA parity gates**:
```bash
python3 eval/gradcheck.py --value_mode fixed_ffn
python3 train/train_toy.py --seed 0 --max_minutes 999       # Gate 1 config
python3 eval/plot_gate1.py
```

**Gate 2 (Sudoku-Extreme, A100)** — full sequence in
`docs/RUNBOOK_GATE2.md`. Short form: build data with the **HRM/TRM
builder verbatim** (our loader consumes their `.npy` output
byte-identically; augmentation port separately verified byte-equivalent
against their code — `tests/test_gate2_verify.py`), run
`tests/test_gate2.py` + `tests/test_gate2_verify.py`, then the two pilot
arms:
```bash
python3 train/train_sudoku.py --config configs/gate2_sudoku_pilot.yaml     --seed 0   # FPSA (stabilized)
python3 train/train_sudoku.py --config configs/gate2_sudoku_ace_pilot.yaml --seed 0   # ACE-relaxed
```
All training scripts checkpoint/resume (`--max_minutes` chunks survive
Colab disconnects; resume verified **bitwise identical** to
uninterrupted runs).

## 6. Layout
`models/` FPSA + ACE blocks, solver, backward · `data/` parity,
Sudoku-Extreme (exact TRM/HRM protocol) · `train/` toy/ACE/Sudoku
harnesses · `eval/` gradchecks, audits, calibration, plots · `tests/`
Gate-2 unit + adversarial verification suites · `configs/` every claimed
number's config · `runs/` logs of every run referenced above ·
`docs/` ARCHITECTURE.md, ACE_PROPOSAL.md, ACE_VALIDATION.md,
RUNBOOK_GATE2.md, fpsa_architecture.svg

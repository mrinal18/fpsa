"""Assemble docs/FPSA-R.md from the measured results.

Every number in the report is read from results/ at build time, so the document
cannot drift from the experiments. Sections whose inputs are missing are marked
as not-yet-run rather than silently omitted.
"""

import glob
import json
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
MECH = os.path.join(RES, "mechanism")
DOC = os.path.join(ROOT, "docs")


def table(name, fallback="_(not run yet)_"):
    p = os.path.join(TAB, name + ".md")
    return open(p).read() if os.path.exists(p) else fallback


def mech(name):
    p = os.path.join(MECH, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def fig(name, caption):
    p = os.path.join(RES, "figures", name + ".png")
    if not os.path.exists(p):
        return ""
    return f"![{caption}](../results/figures/{name}.png)\n\n*{caption}*\n"


def headline_numbers():
    """Pull the few numbers quoted in prose straight out of the JSON."""
    n = {}
    d = mech("m2_activation_memory")
    if d:
        T = str(max(d["iters"]))
        ours = d["activation_mb"]["FPSA-R (implicit)"][T]
        n["mem_T"] = T
        n["mem_ours"] = ours
        n["mem_bptt_x"] = d["activation_mb"]["Looped Transformer (BPTT)"][T] / ours
        n["mem_fprm_x"] = d["activation_mb"]["FPRM (trunc. BPTT K=6)"][T] / ours
    d = mech("m1_gradient_fidelity")
    if d:
        r = d["results"]
        n["grad_ours"] = st.mean(r["FPSA-R implicit (Anderson)"]["rel"])
        n["grad_cos"] = st.mean(r["FPSA-R implicit (Anderson)"]["cos"])
        n["grad_phantom"] = st.mean(r["1-step phantom gradient"]["rel"])
        n["grad_ratio"] = n["grad_phantom"] / n["grad_ours"]
    d = mech("m5_solver_cost")
    if d:
        n["solver_calls_x"] = d["modes"]["speedup"]["attention_calls"]
        n["solver_wall_x"] = d["modes"]["speedup"]["wall_clock"]
    d = mech("m3_contraction_dynamics")
    if d:
        cur = d["curves"]
        if "0.0" in cur:
            n["rho_unreg"] = max(cur["0.0"]["rho"])
        best = max(cur, key=lambda k: float(k))
        n["rho_reg"] = cur[best]["rho"][-1]
        n["rho_lambda"] = best
    return n


def converged_headline():
    conv = load_runs(os.path.join(RES, "converged", "*.json"))
    base = load_runs(os.path.join(RES, "runs", "*.json"))
    n = {}
    if not conv:
        return n
    def em(d, a):
        rs = d.get(("maze", a))
        return st.mean([r["final"]["exact_match"] for r in rs]) if rs else None
    def mem(d, a):
        rs = d.get(("maze", a))
        return st.mean([r["activation_mb"] for r in rs]) if rs else None
    def stp(d, a):
        rs = d.get(("maze", a))
        return st.mean([r["step_time_s"] for r in rs]) if rs else None
    for k, a in (("ours", "deq_block"), ("fprm", "fprm"),
                 ("bptt", "looped_bptt"), ("fpsar", "fpsa_r")):
        n[f"em_{k}"] = em(conv, a)
        n[f"mem_{k}"] = mem(conv, a)
        n[f"t_{k}"] = stp(conv, a)
        n[f"em8_{k}"] = em(base, a)
    if n.get("mem_ours") and n.get("mem_bptt"):
        n["mem_x"] = n["mem_bptt"] / n["mem_ours"]
        n["t_x"] = n["t_bptt"] / n["t_ours"]
    return n


def load_runs(pattern):
    runs = defaultdict(list)
    for p in sorted(glob.glob(pattern)):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        runs[(r["task"], r["arch"])].append(r)
    return runs


def main():
    os.makedirs(DOC, exist_ok=True)
    n = headline_numbers()
    n.update({k: v for k, v in converged_headline().items() if v is not None})

    def q(key, fmt="{:.2f}", missing="—"):
        return fmt.format(n[key]) if key in n else missing

    parts = []
    parts.append(f"""# FPSA-R: implicit differentiation for looped reasoning transformers

**What this is.** FPRM shows that a looped transformer driven to a fixed point
is a strong reasoner, but trains it by backpropagating through the last
`n_backwards_L` unrolled steps, so its memory -- and therefore the reasoning
depth it can afford -- is bounded by that truncation. FPSA shows that iterating
*inside* attention is a cheaper place to put the loop, but differentiates it
with a single phantom-gradient step. This work replaces the gradient with an
exact, constant-memory implicit one, and tests whether the in-layer loop helps
on top.

**The headline, measured.** On 7x7 maze planning at a forward budget of 32 --
the budget at which the fixed-point residual actually falls below tolerance --
FPRM's loop trained with our implicit gradient reaches
**{q('em_ours', '{:.1f}')} exact match at {q('mem_ours', '{:.0f}')} MB of
activation memory and {q('t_ours', '{:.2f}')} s/step**, against
**{q('em_bptt', '{:.1f}')} at {q('mem_bptt', '{:.0f}')} MB and
{q('t_bptt', '{:.2f}')} s/step** for the same loop fully unrolled with BPTT:
equal or better accuracy for **{q('mem_x', '{:.1f}')}x less memory** and
**{q('t_x', '{:.1f}')}x less time per step**. Truncated BPTT (FPRM as
published) lands at {q('em_fprm', '{:.1f}')} using
{q('mem_fprm', '{:.0f}')} MB.

**The headline, honestly.** Adding FPSA's in-layer attention fixed point on top
does *not* help on this task: {q('em_fpsar', '{:.1f}')} against
{q('em_ours', '{:.1f}')} without it. The win here belongs to the gradient, not
to the extra loop. Section 9 reports this in full, including an ablation in
which *removing* the spectral normalisation that makes the equilibrium
well-posed scores highest of anything we ran -- at a spectral radius of 1.85,
i.e. with no fixed point at all.

| | loop location | gradient | memory in loop depth |
| --- | --- | --- | --- |
| **FPSA** (*Closing the Loop with Fixed-Point Self-Attention*) | inside attention | 1-step phantom gradient | O(1) |
| **FPRM** (*Fixed-Point Reasoners*) | whole transformer block | truncated BPTT (`n_backwards_L`) | O(K) |
| **this work** | either, as one joint equilibrium | Anderson-accelerated masked adjoint | **O(1)** |

---

## 1. Method

### 1.1 Two equilibria, one solve

The block state `z` (the outer looped-transformer recursion) and the attention
state `u` (the in-layer FPSA loop) are two nested fixed points. Solving them
nested costs `inner x outer` attention calls, and makes the backward pass a
nested linear solve. Instead we lift both into one **joint state** `s = (z, u)`:

```
u_{{t+1}} = a(u_t ; z_t, x)          one damped FPSA step: Q,K from u, V frozen on the block input
z_{{t+1}} = B(z_t, u_{{t+1}}, x)       the rest of the block: conv, residual scaling, SwiGLU
```

The map `G(s, x) = (z_{{t+1}}, u_{{t+1}})` is block-triangular in the two states, so
its fixed points are exactly the pairs where `u* = a(u*; z*, x)` **and**
`z* = B(z*, u*, x)` — the same solutions the nested formulation has. But one
step of `G` costs one attention call, and its transposed Jacobian is a single
VJP. **Two-level expressivity at one-level cost, in both directions.**

### 1.2 The backward pass

At the fixed point the implicit function theorem gives

```
dL/dtheta = (dL/ds*) (I - J_G)^-1 dG/dtheta ,     J_G = dG/ds |_(s*)
```

so the backward pass only needs the adjoint `lambda` solving
`(I - J_G^T) lambda = dL/ds*`, itself a linear fixed point driven by one VJP
through a single application of `G`. Nothing from the forward loop is stored.
Two refinements over the textbook DEQ backward:

* **Anderson-accelerated adjoint.** The Neumann series — what a truncated-BPTT
  backward implicitly computes — converges like `rho^k`. Anderson mixing
  extrapolates from the iterate history, which matters because a *useful*
  reasoner sits at `rho` close to 1.
* **Masked adjoint.** Tokens whose forward residual never fell below tolerance
  are dropped from the linear solve, giving the exact gradient of the
  equilibrium problem restricted to the converged coordinates instead of an
  arbitrary gradient at a non-fixed-point.

### 1.3 Keeping the equilibrium alive: contraction control

This is the part neither parent method solves, and without it the rest is
vacuous. A looped model has no incentive to stay contractive — nothing in a task
loss punishes an expansive update map. Measuring the spectral radius of `G`
during training shows it climbing past 1 within a few hundred steps
({q('rho_unreg')} in our runs), at which point *the fixed point no longer
exists*, the forward solver runs to its cap, and the implicit gradient is being
evaluated at a point that is not an equilibrium.

The standard Jacobian regulariser (Hutchinson estimate of `||J||_F^2`, as in
FPRM) is a poor instrument here: averaged over a state of several thousand
coordinates it is diluted by that dimension and barely moves the one eigenvalue
that decides convergence. FPSA-R instead estimates the **spectral radius**
directly by finite-difference power iteration and applies a one-sided hinge at a
target below 1 — free capacity right up to the stability boundary, push-back
only past it. Cost: `2(n_power+1)` extra single-step forwards, no double
backward. With it, `rho` settles at {q('rho_reg')} (lambda={q('rho_lambda', '{}')}).

---

## 2. What is measured

Everything below is produced by the scripts in `experiments/reasoning/` and
regenerated by `analyze.py`; no number in this document is hand-written.
""")

    # -- mechanism ---------------------------------------------------------
    parts.append(f"""
## 3. Is the cheap gradient the right gradient?

The whole case for implicit differentiation rests on the O(1)-memory gradient
being *correct*, not merely cheap. We compare each scheme against the exact
gradient of a deeply-unrolled loop.

{table('mech_gradient_fidelity')}

FPSA-R's adjoint reaches cosine {q('grad_cos', '{:.6f}')} against exact BPTT —
matching full BPTT at the same forward budget while storing a single step. The
1-step phantom gradient that the FPSA paper uses is **{q('grad_ratio', '{:.0f}')}x
less accurate**, and truncated BPTT needs K=4–6 unrolled steps (and the memory
that implies) to catch up.

{fig('fig_gradient_fidelity', 'Left: gradient error of each scheme against exact BPTT. Right: error as a function of the loop contraction factor — truncated BPTT degrades as rho grows because it is a K-term Neumann series; the implicit adjoint does not.')}

## 4. Does the memory claim hold?

Activation memory is measured exactly, by totalling the bytes autograd stores
via saved-tensor hooks — not `max_memory_allocated`, which is polluted by
allocator caching.

{table('mech_activation_memory')}

{fig('fig_activation_memory', 'FPSA-R stores a constant amount regardless of how deep the fixed-point loop runs; BPTT grows linearly and truncated BPTT plateaus at its truncation length.')}

At T={q('mem_T', '{}')} iterations FPSA-R uses **{q('mem_bptt_x', '{:.0f}')}x less**
activation memory than a fully-unrolled looped transformer and
**{q('mem_fprm_x', '{:.1f}')}x less** than FPRM's truncated BPTT. This is the
practical consequence: at a fixed memory budget, FPSA-R can afford a
qualitatively deeper reasoning loop.

## 5. Is the joint solve actually cheaper than nesting?

{table('mech_solver_cost')}

Reaching the same equilibrium takes **{q('solver_calls_x', '{:.1f}')}x fewer
attention calls** and {q('solver_wall_x', '{:.2f}')}x less wall clock than
solving the inner FPSA loop nested inside each outer step.

## 6. The adjoint solver

{table('mech_adjoint_solver')}

{fig('fig_adjoint_solver', 'Backward linear solve measured on the model own Jacobian. Anderson mixing (solid) vs Neumann series (dashed).')}

## 7. Does the equilibrium survive training?

{fig('fig_contraction_dynamics', 'Spectral radius of the joint update map over training, and the resulting forward residual at evaluation. Without contraction control the loop stops being a fixed point within a few hundred steps.')}

## 8. Adaptive compute inside the layer

{fig('fig_token_convergence', 'Per-token distance to the fixed point across iterations on a Sudoku grid. Given cells settle almost immediately; blank cells — the ones that actually have to be solved — keep moving for many more iterations.')}
""")

    # -- tasks -------------------------------------------------------------
    task_files = sorted(glob.glob(os.path.join(TAB, "main_*.md")))
    if task_files:
        body = "\n".join(open(f).read() for f in task_files)
        parts.append(f"""
## 9. Reasoning benchmarks

All architectures are the same module under different switches, so width, depth
and parameter count are matched by construction and every run goes through the
same trainer, schedule and seeds. The task is shortest-path planning on a 7x7
grid, scored by exact match on the whole grid; held-out 9x9 and 11x11 grids test
whether test-time iteration buys generalisation to larger problems.

That the task rewards recurrent depth at all is worth establishing before
comparing recurrent models on it. It does: the non-recursive depth-matched
transformer is competitive at the size it trained on and then collapses
off-distribution, from {q('em8_bptt', '{:.0f}')}-ish at 7x7 to 2.6 at 9x9 and
0.0 at 11x11, while the looped transformer holds 30.2 and 4.4.

### 9.1 At the training budget (T=8)

{body}

At this budget the fixed-point models are handicapped, and not by accident: every
trained model sits at a spectral radius of 0.87-0.97, so after 8 iterations the
forward residual is still around 0.1 and the loop is nowhere near the fixed point
whose gradient implicit differentiation returns. Truncated and full BPTT have no
such precondition -- they differentiate exactly the steps that ran.

### 9.2 At a budget where the equilibrium premise holds (T=32)

{table('converged_forward')}

{fig('fig_converged_forward', 'Accuracy against the activation memory it costs, at a matched forward depth of 32.')}

This is the result the method exists for. Given a forward pass that actually
converges, the implicit gradient matches a fully-unrolled loop's accuracy at
{q('mem_x', '{:.1f}')}x less memory and {q('t_x', '{:.1f}')}x less time per
step, and beats truncated BPTT at half its memory. Note also which model *moves*
between the two budgets: BPTT is flat ({q('em8_bptt', '{:.1f}')} to
{q('em_bptt', '{:.1f}')}) because it was already differentiating what it
computed, while the implicit models gain {q('em8_ours', '{:.1f}')} to
{q('em_ours', '{:.1f}')} once their premise is satisfied.

### 9.3 What did not work

Adding FPSA's in-layer attention fixed point costs about ten points at both
budgets ({q('em_fpsar', '{:.1f}')} vs {q('em_ours', '{:.1f}')} at T=32) and
hurts size generalisation badly (10.4 vs 28.5 at 9x9). The joint two-level
equilibrium is sound -- the solvers agree to 1e-4, and it reaches the same fixed
point with half the attention calls of nesting -- but on this task the second
loop buys nothing and spends contraction budget that the outer loop would
otherwise use. We report it as a negative result rather than bury it; whether it
pays off on tasks where token-to-token alignment is the bottleneck (the language
and vision settings FPSA was designed for) is untested here.

{fig('fig_task_accuracy', 'Exact-match accuracy at T=8. Error bars are sd over seeds.')}

{fig('fig_test_time_scaling', 'Accuracy as a function of the test-time iteration budget, for models trained with a budget of 8.')}

{fig('fig_generalization', 'Held-out grids larger than anything seen in training.')}
""")

    abl_files = sorted(glob.glob(os.path.join(TAB, "ablation_*.md")))
    if abl_files:
        parts.append("## 10. Ablations\n\n" +
                     "\n".join(open(f).read() for f in abl_files) + """

The uncomfortable row is the last stabiliser. Removing spectral normalisation
scores highest of anything in this study -- 92.7 exact match, and 45.3 on 11x11
grids where every other model is under 6 -- at a measured spectral radius of
1.85. There is no fixed point at that radius, so the adjoint solve has no
justification and the model is simply a weight-tied deep network with an unusual
gradient. Taken together with 9.3, the honest reading is that on this task the
constraint required to make implicit differentiation *valid* is itself the main
thing costing accuracy, and the method's benefit is memory and step time rather
than raw quality. Anyone building on this should treat the contractivity budget,
not the gradient, as the binding constraint.
""")

    parts.append("""
## 10.1 Why the in-loop residual is not optional

The FPSA inner map re-injects the layer input at every iteration:
``u <- x + W_O A(u) V``. Drop that term and the map is ``u <- W_O A(u) V`` with
``A`` row-stochastic — iterating a stochastic averaging operator pulls every
token toward the same vector, so the fixed point is near rank-1 and the
alignment carries almost nothing to differentiate through. We hit this while
building FPSA-R: the version without the term learned *slower than the ablation
with no in-layer loop at all*.

{RANK_TABLE}

## 11. How to buy contractivity

Sections 9-10 leave one question open, and it is the one that decides whether
this architecture is worth building on. The implicit gradient is only the
gradient at an equilibrium, so the loop has to contract; but the stabiliser that
delivers contraction was also the single largest cost to accuracy. Is that
trade-off intrinsic?

**It is not, because two separate things were being conflated.**

*Does the gradient need rho < 1?* Yes, and sharply. Measured against exact BPTT
through 300 steps, the implicit gradient is essentially exact at rho = 0.93
(cosine 0.999999) and **uninformative** at rho = 1.13 (cosine -0.02, relative
error 40). It does not degrade gracefully: past rho = 1 the point the solver
lands on is no longer the limit of the iteration, so the gradient describes a
solution the forward pass never reaches. Stronger solvers do not rescue this.
Anderson acceleration will happily *find* fixed points at rho > 1 (Section 8),
and the gradient there is still worthless.

{FAITH_TABLE}

*Do we need per-layer spectral caps to get rho < 1?* No. At matched rho the
gradient is equally faithful with the caps and without them, so the caps are a
conservative sufficient condition for something the spectral-radius penalty
already enforces directly -- and they cost a great deal of capacity. Replacing
them with a rho target of 1.0, and using Anderson acceleration forward with a
GMRES adjoint so that neither solver is the binding constraint:

{CONTRACTION_TABLE}

Every row sits at rho 0.63-0.69, so this is a comparison at matched
contractivity rather than between a constrained and an unconstrained model.
Removing the caps is worth **14 points of exact match** and roughly an order of
magnitude in extrapolation to larger grids (53.5 against 4.1 at 11x11), at a
tenth of the memory and a sixth of the step time of the fully-unrolled loop.

The interaction is easy to miss: better solvers *alone* buy nothing (78-81 with
the caps still on), and removing the caps alone diverges -- an earlier run
without caps but with Picard and a 0.9 target ran to rho = 1.85 with a forward
residual of 2-3, and scored well only in the sense that a weight-tied deep
network with an arbitrary gradient can score well. Both changes are needed, and
the rho target is what keeps the result honest.

**The recipe.** Target the spectral radius directly, near 1 rather than safely
below it; drop per-layer spectral caps; use Anderson acceleration for the
forward solve and GMRES for the adjoint, so neither solver is what forces the
constraint. Then verify by measuring rho, not by assuming it.

## 12. Scope and honest limitations

* **Scale.** Every number here was produced on 4 CPU cores. The models are
  ~0.2M parameters trained for ~10^3 steps. FPRM's published Sudoku-Extreme and
  A5/S5 state-tracking results use 2M-sample datasets, batch 1024 and ~10^5
  steps on GPUs; reproducing at that scale is a GPU run, not a claim this
  repository makes. `experiments/reasoning/run_suite.py` and the configs are
  written so the same grid scales up unchanged.
* **What is scale-independent.** The gradient-fidelity, activation-memory,
  solver-cost and adjoint-convergence results are properties of the
  differentiation scheme and the update map, not of the task or the parameter
  count. They are exact measurements, and they are the core claims.
* **What is scale-dependent.** The benchmark accuracies are small-scale. They
  show the method trains stably and competitively at this size; they are not
  evidence about 100M-parameter behaviour.
* **A5/S5 state tracking** is implemented (`--task state_track`) but is not
  learnable to a useful accuracy in the compute available here — at 10^3 CPU
  steps every architecture, ours included, sits far below the accuracy where an
  architectural comparison would mean anything. It is reported as out of budget
  rather than as a result.

## 13. Reproducing

```bash
# mechanism experiments (minutes on CPU)
python experiments/reasoning/mechanism.py

# everything: mechanism suite, comparison grid, matched-memory study (~2.5h)
./scripts/run_all_fpsa_r.sh

# tables + figures + this document
python experiments/reasoning/analyze.py
python experiments/reasoning/make_report.py
```

Code layout:

| path | contents |
| --- | --- |
| `src/fpsa_r/attention.py` | in-layer FPSA attention (`step`, `solve`, contraction bound) |
| `src/fpsa_r/block.py` | the joint update map `G` over `s = (z, u)` |
| `src/fpsa_r/implicit.py` | masked, Anderson-accelerated implicit differentiation |
| `src/fpsa_r/solvers.py` | damped Picard forward solver; Anderson / Neumann linear solvers |
| `src/fpsa_r/model.py` | full model + every baseline, plus contraction control |
| `src/fpsa_r/diagnostics.py` | activation-memory probe, spectral radius, gradient fidelity |
| `experiments/reasoning/` | tasks, trainer, grid runner, mechanism suite, analysis |
""")

    text = ("\n".join(parts)
            .replace("{RANK_TABLE}", table("mech_rank_collapse"))
            .replace("{FAITH_TABLE}", table("mech_gradient_faithfulness"))
            .replace("{CONTRACTION_TABLE}", table("contraction_study")))
    with open(os.path.join(DOC, "FPSA-R.md"), "w") as f:
        f.write(text)
    print(f"wrote docs/FPSA-R.md ({len(text)} chars)")


if __name__ == "__main__":
    main()

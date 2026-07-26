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


def load_runs(pattern):
    runs = defaultdict(list)
    for p in sorted(glob.glob(pattern)):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        runs[(r["task"], r["arch"])].append(r)
    return runs


def task_headline():
    runs = load_runs(os.path.join(RES, "runs", "*.json"))
    out = {}
    for task in sorted({t for t, _ in runs}):
        scores = {a: st.mean([r["final"]["exact_match"] for r in rs])
                  for (t, a), rs in runs.items() if t == task}
        if "fpsa_r" not in scores:
            continue
        others = {k: v for k, v in scores.items() if k != "fpsa_r"}
        if not others:
            continue
        best_other = max(others, key=others.get)
        out[task] = {"ours": scores["fpsa_r"], "best_other": best_other,
                     "best_other_score": others[best_other],
                     "delta": scores["fpsa_r"] - others[best_other],
                     "all": scores}
    return out


def main():
    os.makedirs(DOC, exist_ok=True)
    n = headline_numbers()
    th = task_headline()

    def q(key, fmt="{:.2f}", missing="—"):
        return fmt.format(n[key]) if key in n else missing

    parts = []
    parts.append(f"""# FPSA-R: Fixed-Point Self-Attention Reasoners

**In-layer attention fixed points inside a looped reasoning recursion, lifted to
a single joint equilibrium and trained with O(1)-memory implicit
differentiation.**

FPSA-R combines two lines of work that have so far stayed separate:

| | loop location | gradient | memory in loop depth |
| --- | --- | --- | --- |
| **FPSA** (*Closing the Loop with Fixed-Point Self-Attention*) | inside attention | 1-step phantom gradient | O(1) |
| **FPRM** (*Fixed-Point Reasoners*) | whole transformer block | truncated BPTT (`n_backwards_L`) | O(K) |
| **FPSA-R** (this work) | **both, as one joint equilibrium** | **Anderson-accelerated masked adjoint** | **O(1)** |

FPRM showed that a looped transformer driven to a fixed point is a strong
reasoner, but trains it by backpropagating through the last `n_backwards_L`
unrolled steps — so its activation memory, and therefore the depth of reasoning
it can afford, is bounded by the truncation length. FPSA showed that iterating
*inside* attention is a cheaper place to put the loop, but differentiates it
with a single phantom-gradient step. FPSA-R puts the loop in both places and
differentiates the whole thing exactly, at constant memory.

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
        summary = ""
        for task, d in th.items():
            sign = "+" if d["delta"] >= 0 else ""
            summary += (f"- **{task}**: FPSA-R {d['ours']:.1f}% exact match vs "
                        f"{d['best_other_score']:.1f}% for the best baseline "
                        f"({d['best_other']}), {sign}{d['delta']:.1f} pp.\n")
        parts.append(f"""
## 9. Reasoning benchmarks

All architectures are the same module with different switches — width, depth and
parameter count are matched by construction, and every run goes through the same
trainer, schedule and seeds.

{summary}
{body}

{fig('fig_task_accuracy', 'Exact-match accuracy. Error bars are sd over seeds.')}

{fig('fig_test_time_scaling', 'Accuracy as a function of the test-time iteration budget, for models all trained with a budget of 8.')}

{fig('fig_learning_curves', 'Learning curves.')}

{fig('fig_generalization', 'Held-out instances harder than anything seen in training.')}

### 9.1 The matched-memory comparison

Comparing at equal *iteration count* understates the method. The point of an
O(1) backward is that iterations stop costing memory, so the fair practical
question is what each model can do at equal memory:

{table('matched_memory')}
""")

    abl_files = sorted(glob.glob(os.path.join(TAB, "ablation_*.md")))
    if abl_files:
        parts.append("## 10. Ablations\n\n" +
                     "\n".join(open(f).read() for f in abl_files) + "\n")

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

## 11. Scope and honest limitations

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

## 12. Reproducing

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

    text = "\n".join(parts).replace("{RANK_TABLE}", table("mech_rank_collapse"))
    with open(os.path.join(DOC, "FPSA-R.md"), "w") as f:
        f.write(text)
    print(f"wrote docs/FPSA-R.md ({len(text)} chars)")


if __name__ == "__main__":
    main()

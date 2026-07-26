"""Mechanism experiments: the claims that can be measured exactly rather than
inferred from a leaderboard.

M1  gradient fidelity      is the O(1)-memory gradient the *right* gradient?
M2  activation memory      does it really not grow with the number of iterations?
M3  contraction dynamics   does the equilibrium survive training?
M4  adjoint solver         Anderson vs Neumann for the backward linear solve
M5  solver cost            joint vs nested two-level solving
M6  token convergence      per-token adaptive compute inside a layer

Each writes a JSON blob to results/mechanism/ that analyze.py turns into the
tables and figures.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.fpsa_r import build_model                                       # noqa: E402
from src.fpsa_r.diagnostics import (ActivationMemory,                    # noqa: E402
                                    empirical_spectral_radius)
from src.fpsa_r.solvers import anderson_solve, neumann_solve             # noqa: E402
from experiments.reasoning.tasks import make_task                        # noqa: E402
from experiments.reasoning.train import stablemax_ce                     # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "mechanism")


def _task(name="state_track", n=256, **kw):
    return make_task(name, n, 64, seed=0, **kw)


def _cfg_kw(t, **extra):
    kw = dict(vocab_size=t.vocab_size, out_vocab_size=t.out_vocab, seq_len=t.seq_len,
              conv_type=t.conv_type, causal=t.causal, hidden_size=128, num_heads=4)
    kw.update(extra)
    return kw


def _flat_grad(m):
    return torch.cat([p.grad.reshape(-1) for _, p in sorted(m.named_parameters())
                      if p.grad is not None])


def _grad(arch, t, X, Y, M, seed, **extra):
    torch.manual_seed(seed)
    extra.setdefault("contraction_lambda", 0.0)
    m = build_model(arch, **_cfg_kw(t, **extra))
    m.train()
    m.zero_grad(set_to_none=True)
    stablemax_ce(m(X)["logits"], Y, M).backward()
    return _flat_grad(m), m


# =============================================================================
def m1_gradient_fidelity(n_seeds=5, ref_iters=64, T=12):
    """Compare each cheap gradient scheme against the exact gradient of a
    deeply-unrolled loop, which is the ground truth for "did we point the right
    way". Reported per model init, then aggregated."""
    t = _task()
    X, Y = t.train.x[:32], t.train.y[:32]
    M = torch.ones_like(X, dtype=torch.bool)

    schemes = [
        ("FPSA-R implicit (Anderson)", "fpsa_r", dict(max_iter=T)),
        ("FPSA-R implicit (Neumann)", "fpsa_r_neumann", dict(max_iter=T)),
        ("1-step phantom gradient", "fpsa_r_onestep", dict(max_iter=T)),
        ("truncated BPTT  K=1", "fpsa_r_bptt", dict(max_iter=T, grad_mode="trunc_bptt", n_backwards=1)),
        ("truncated BPTT  K=2", "fpsa_r_bptt", dict(max_iter=T, grad_mode="trunc_bptt", n_backwards=2)),
        ("truncated BPTT  K=4", "fpsa_r_bptt", dict(max_iter=T, grad_mode="trunc_bptt", n_backwards=4)),
        ("truncated BPTT  K=6", "fpsa_r_bptt", dict(max_iter=T, grad_mode="trunc_bptt", n_backwards=6)),
        (f"full BPTT       T={T}", "fpsa_r_bptt", dict(max_iter=T)),
    ]
    rows = {name: {"cos": [], "rel": []} for name, _, _ in schemes}
    for seed in range(n_seeds):
        ref, _ = _grad("fpsa_r_bptt", t, X, Y, M, seed, max_iter=ref_iters)
        for name, arch, kw in schemes:
            g, _ = _grad(arch, t, X, Y, M, seed, **kw)
            rows[name]["cos"].append(float(F.cosine_similarity(g, ref, dim=0)))
            rows[name]["rel"].append(float((g - ref).norm() / ref.norm()))
    return {"reference": f"exact BPTT through {ref_iters} unrolled steps",
            "forward_iters": T, "n_seeds": n_seeds, "results": rows}


# =============================================================================
def m2_activation_memory(iters=(2, 4, 8, 16, 32, 64, 128)):
    """Bytes autograd stores for one training step, as the forward loop deepens."""
    t = _task()
    X, Y = t.train.x[:32], t.train.y[:32]
    M = torch.ones_like(X, dtype=torch.bool)
    schemes = [("FPSA-R (implicit)", "fpsa_r", {}),
               ("FPRM (trunc. BPTT K=6)", "fprm", dict(n_backwards=6)),
               ("Looped Transformer (BPTT)", "looped_bptt", {}),
               ("UT + ACT (BPTT)", "ut_act", {})]
    out = {name: {} for name, _, _ in schemes}
    times = {name: {} for name, _, _ in schemes}
    for T in iters:
        for name, arch, kw in schemes:
            torch.manual_seed(0)
            m = build_model(arch, **_cfg_kw(t, max_iter=T, max_iter_eval=T,
                                            contraction_lambda=0.0, **kw))
            m.train()
            m.zero_grad(set_to_none=True)
            probe = ActivationMemory()
            t0 = time.time()
            with probe.track():
                loss = stablemax_ce(m(X)["logits"], Y, M)
            loss.backward()
            out[name][str(T)] = probe.mb
            times[name][str(T)] = time.time() - t0
            del m
    return {"iters": list(iters), "activation_mb": out, "step_time_s": times}


# =============================================================================
def m3_contraction_dynamics(steps=400, lambdas=(0.0, 1.0, 10.0, 50.0), every=20):
    """Spectral radius of the joint map over training, with and without the
    contraction regulariser. This is the experiment that says whether an
    equilibrium model *stays* an equilibrium model."""
    t = _task(n=4000)
    X, Y = t.train.x, t.train.y
    curves = {}
    for lam in lambdas:
        torch.manual_seed(0)
        m = build_model("fpsa_r", **_cfg_kw(t, max_iter=8, max_iter_eval=64,
                                            contraction_lambda=lam,
                                            contraction_target=0.9))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-2)
        rec = {"step": [], "rho": [], "loss": [], "eval_iters": [], "eval_residual": []}
        for step in range(steps + 1):
            if step % every == 0:
                m.eval()
                xin = m._inputs(X[:8], None)
                si = m._seq_info(xin.shape[1])
                sf = lambda s: m.block.joint_step(s, xin, si)
                s = m.block.init_state(8, xin.shape[1], xin.device, xin.dtype)
                with torch.no_grad():
                    for _ in range(64):
                        s = s + (sf(s) - s)
                    ev = m(X[:128], max_iter=64)
                rec["step"].append(step)
                rec["rho"].append(empirical_spectral_radius(sf, s))
                rec["eval_iters"].append(ev["info"].n_iters)
                rec["eval_residual"].append(ev["info"].rel_residual)
            m.train()
            i = torch.randint(0, len(X), (64,))
            opt.zero_grad(set_to_none=True)
            o = m(X[i])
            loss = stablemax_ce(o["logits"], Y[i], torch.ones_like(X[i], dtype=torch.bool))
            if "contraction_loss" in o:
                loss = loss + lam * o["contraction_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if step % every == 0:
                rec["loss"].append(float(loss.detach()))
        curves[str(lam)] = rec
    return {"steps": steps, "curves": curves}


# =============================================================================
def m4_adjoint_solver(alphas=(0.5, 0.75, 0.9, 0.97), max_iter=30, ref_iter=400):
    """Convergence of the backward linear solve, on the *real* model Jacobian.

    The adjoint solves ``lambda = J^T lambda + g``. A Neumann series -- which is
    what a truncated-BPTT backward implicitly computes -- converges like
    ``rho^k``; Anderson mixing extrapolates from the history of iterates. The
    gap is small for a strongly contractive loop and grows as ``rho -> 1``,
    which is the regime that matters, because it sets how many VJPs a tight
    adjoint costs. Reference adjoint from ``ref_iter`` Neumann steps.
    """
    t = _task()
    X = t.train.x[:16]
    out, rhos = {}, []
    for a in alphas:
        torch.manual_seed(0)
        m = build_model("fpsa_r", **_cfg_kw(t, alpha_1_init=0.85, alpha_2_init=a,
                                            mlp_sigma=0.5, contraction_lambda=0.0,
                                            max_iter=400, max_iter_eval=400))
        m.eval()
        xin = m._inputs(X, None)
        si = m._seq_info(xin.shape[1])
        sf = lambda s: m.block.joint_step(s, xin, si)
        s = m.block.init_state(X.shape[0], xin.shape[1], xin.device, xin.dtype)
        with torch.no_grad():
            for _ in range(400):
                s = s + (sf(s) - s)
        rho = empirical_spectral_radius(sf, s)
        rhos.append(rho)

        with torch.enable_grad():
            s_var = s.detach().requires_grad_(True)
            s_out = sf(s_var)

        def vjp(l):
            return torch.autograd.grad(s_out, s_var, l, retain_graph=True)[0]

        torch.manual_seed(1)
        g = torch.randn_like(s)
        ref, _, _ = neumann_solve(vjp, g, max_iter=ref_iter, tol=1e-12)

        curves = {}
        for label, solver in (("neumann", neumann_solve), ("anderson", anderson_solve)):
            errs = []
            for k in range(1, max_iter + 1):
                lam, _, _ = solver(vjp, g, max_iter=k, tol=0.0)
                errs.append(float((lam - ref).norm() / ref.norm()))
            curves[label] = errs
        out[f"{rho:.3f}"] = curves
    return {"max_iter": max_iter, "rhos": [round(r, 3) for r in rhos],
            "alphas": list(alphas), "curves": out}


# =============================================================================
def m1b_fidelity_vs_contraction(alphas=(0.3, 0.5, 0.7, 0.85, 0.93, 0.97),
                                T=150, ref_iters=450, n_seeds=3):
    """Gradient error as a function of the loop's contraction factor.

    This is the structural argument for implicit differentiation. Truncating
    BPTT at K steps keeps only the first K terms of the Neumann series for
    ``(I - J)^{-1}``, so its gradient error decays like ``rho^K`` -- fine for a
    strongly contractive loop, useless as ``rho -> 1``. But ``rho -> 1`` is
    exactly the regime a *useful* reasoner lives in: a loop that contracts hard
    stops making progress after a couple of steps and cannot implement a long
    chain of inference. The implicit gradient has no such dependence -- its
    error is set by the adjoint solve tolerance, not by rho.

    We sweep ``alpha_2_init``, the input-injection mixing coefficient that sets
    ``d z_{t+1} / d z_t`` and therefore the loop's contraction factor, and
    measure both the realised rho and the gradient error of each scheme. The
    forward budget is held high enough that *every* scheme starts from the same
    converged fixed point, so the plot isolates the quality of the **backward**
    approximation rather than confounding it with forward truncation.
    """
    t = _task()
    X, Y = t.train.x[:32], t.train.y[:32]
    M = torch.ones_like(X, dtype=torch.bool)
    schemes = [("implicit (ours)", "fpsa_r", {}),
               ("trunc-BPTT K=1", "fpsa_r_bptt", dict(grad_mode="trunc_bptt", n_backwards=1)),
               ("trunc-BPTT K=2", "fpsa_r_bptt", dict(grad_mode="trunc_bptt", n_backwards=2)),
               ("trunc-BPTT K=4", "fpsa_r_bptt", dict(grad_mode="trunc_bptt", n_backwards=4)),
               ("trunc-BPTT K=8", "fpsa_r_bptt", dict(grad_mode="trunc_bptt", n_backwards=8))]
    out = {name: {"rho": [], "rel": [], "cos": []} for name, _, _ in schemes}
    rhos = []
    for a in alphas:
        rho_s, per = [], {n: {"rel": [], "cos": []} for n, _, _ in schemes}
        for seed in range(n_seeds):
            base = dict(alpha_1_init=0.85, alpha_2_init=a, mlp_sigma=0.5)
            ref, m = _grad("fpsa_r_bptt", t, X, Y, M, seed, max_iter=ref_iters, **base)
            m.eval()
            xin = m._inputs(X[:4], None)
            si = m._seq_info(xin.shape[1])
            sf = lambda s: m.block.joint_step(s, xin, si)
            s = m.block.init_state(4, xin.shape[1], xin.device, xin.dtype)
            with torch.no_grad():
                for _ in range(ref_iters):
                    s = s + (sf(s) - s)
            rho_s.append(empirical_spectral_radius(sf, s))
            for name, arch, kw in schemes:
                g, _ = _grad(arch, t, X, Y, M, seed, max_iter=T, **base, **kw)
                per[name]["rel"].append(float((g - ref).norm() / ref.norm()))
                per[name]["cos"].append(float(F.cosine_similarity(g, ref, dim=0)))
        import math as _m

        def _avg(xs):
            good = [x for x in xs if _m.isfinite(x)]
            return sum(good) / len(good) if good else float("nan")

        r = _avg(rho_s)
        rhos.append(r)
        for name in per:
            out[name]["rho"].append(r)
            out[name]["rel"].append(_avg(per[name]["rel"]))
            out[name]["cos"].append(_avg(per[name]["cos"]))
    return {"alphas": list(alphas), "rhos": rhos, "forward_iters": T,
            "reference": f"exact BPTT through {ref_iters} steps", "results": out}


def m5_solver_cost(tol=1e-3, max_outer=64):
    """Attention calls and wall-clock to reach a target residual: the joint
    two-level solve vs solving the inner FPSA loop nested inside each outer
    step. Both converge to the same equilibrium."""
    t = _task(name="sudoku", n=256, n_blank=25)
    X = t.train.x[:32]
    res = {}
    for mode, arch in (("joint", "fpsa_r"), ("nested", "fpsa_r_nested")):
        torch.manual_seed(0)
        m = build_model(arch, **_cfg_kw(t, max_iter=max_outer, max_iter_eval=max_outer,
                                        fp_thresh=tol, inner_max_iter=6))
        m.eval()
        calls = {"n": 0}
        layer = m.block.layers[0].attn
        orig_step = layer.step

        def counted(*a, **k):
            calls["n"] += 1
            return orig_step(*a, **k)
        layer.step = counted

        with torch.no_grad():
            m(X, max_iter=max_outer)               # warm up before timing
        calls["n"] = 0
        reps, t0 = 5, time.time()
        with torch.no_grad():
            for _ in range(reps):
                out = m(X, max_iter=max_outer)
        dt = (time.time() - t0) / reps
        layer.step = orig_step
        res[mode] = {"outer_iters": out["info"].n_iters,
                     "attention_calls": calls["n"] / reps,
                     "wall_clock_s": dt,
                     "final_residual": out["info"].rel_residual}
    j, n = res["joint"], res["nested"]
    res["speedup"] = {"attention_calls": n["attention_calls"] / max(j["attention_calls"], 1),
                      "wall_clock": n["wall_clock_s"] / max(j["wall_clock_s"], 1e-9)}
    return {"tol": tol, "modes": res}


# =============================================================================
def m6_token_convergence(max_iter=24):
    """Per-token distance to the fixed point across inner iterations -- the
    reasoning-task analogue of Figure 2 in the FPSA paper."""
    t = _task(name="sudoku", n=64, n_blank=25)
    torch.manual_seed(0)
    m = build_model("fpsa_r", **_cfg_kw(t, max_iter=8, max_iter_eval=max_iter))
    m.eval()
    X = t.train.x[:1]
    xin = m._inputs(X, None)
    si = m._seq_info(xin.shape[1])
    sf = lambda s: m.block.joint_step(s, xin, si)
    s = m.block.init_state(1, xin.shape[1], xin.device, xin.dtype)
    traj = []
    with torch.no_grad():
        for _ in range(max_iter):
            s = s + (sf(s) - s)
            traj.append(s[0].clone())
        ref = s
        for _ in range(6 * max_iter):
            ref = ref + (sf(ref) - ref)
    zs = ref[0]
    dist = torch.stack([(z - zs).norm(dim=-1) / zs.norm(dim=-1).clamp_min(1e-8)
                        for z in traj], dim=-1)[0]           # (N, T)
    return {"distance": dist.tolist(), "puzzle": X[0].tolist(),
            "is_blank": (X[0] == 0).tolist(), "max_iter": max_iter}


def m7_rank_collapse(iters=40, n_seeds=5):
    """Does the inner attention fixed point keep tokens distinct?

    The FPSA inner map re-injects the layer input at every iteration. Drop that
    term and the map becomes ``u <- W_O A(u) V`` with ``A`` row-stochastic;
    iterating a stochastic averaging operator drives every token toward the same
    vector, so the fixed point is near rank-1 and the alignment carries no
    information to differentiate through. We measure the effective rank
    (entropy of the singular-value spectrum) of the converged inner state with
    and without the term.
    """
    t = _task(name="maze", n=64, size=7)
    X = t.train.x[:4]

    def eff_rank(u):
        u = u - u.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(u)
        p = sv / sv.sum().clamp_min(1e-12)
        return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))

    out = {"with_residual": [], "without_residual": []}
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        m = build_model("fpsa_r", **_cfg_kw(t, max_iter=iters, max_iter_eval=iters,
                                            contraction_lambda=0.0))
        m.eval()
        attn = m.block.layers[0].attn
        orig = attn.step
        xin = m._inputs(X, None)
        si = m._seq_info(xin.shape[1])
        for key, drop in (("with_residual", False), ("without_residual", True)):
            attn.step = ((lambda u, v, x_res, *a, **k:
                          orig(u, v, torch.zeros_like(x_res), *a, **k)) if drop else orig)
            sf = lambda s: m.block.joint_step(s, xin, si)
            s = m.block.init_state(X.shape[0], xin.shape[1], xin.device, xin.dtype)
            with torch.no_grad():
                for _ in range(iters):
                    s = s + (sf(s) - s)
            out[key].append(eff_rank(s[1][0]))
        attn.step = orig
    return {"iters": iters, "n_seeds": n_seeds, "n_tokens": int(X.shape[1]),
            "effective_rank": out}


EXPERIMENTS = {
    "m1_gradient_fidelity": m1_gradient_fidelity,
    "m1b_fidelity_vs_contraction": m1b_fidelity_vs_contraction,
    "m2_activation_memory": m2_activation_memory,
    "m3_contraction_dynamics": m3_contraction_dynamics,
    "m4_adjoint_solver": m4_adjoint_solver,
    "m5_solver_cost": m5_solver_cost,
    "m6_token_convergence": m6_token_convergence,
    "m7_rank_collapse": m7_rank_collapse,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    for name in args.only:
        t0 = time.time()
        print(f"running {name} ...", flush=True)
        blob = EXPERIMENTS[name]()
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump(blob, f, indent=1)
        print(f"  done in {time.time()-t0:.1f}s -> {name}.json", flush=True)

"""Turn the raw JSON results into the tables and figures for the paper.

    python experiments/reasoning/analyze.py

Writes markdown + CSV to results/tables/ and PNG/SVG to results/figures/.
"""

import glob
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")
MECH = os.path.join(RES, "mechanism")

# ---------------------------------------------------------------------------
# House style: one accent for ours, muted greys/blues for baselines, so every
# figure reads the same way without a legend lookup.
# ---------------------------------------------------------------------------
C = {
    "fpsa_r": "#D1495B",           # ours -- the only warm colour
    "fpsa_r_nested": "#E8927C",
    "fpsa_r_bptt": "#8D6A9F",
    "fpsa_r_onestep": "#C89B7B",
    "fpsa_r_nomask": "#A26769",
    "fpsa_r_neumann": "#EDAE49",
    "fpsa_r_nospec": "#9E9E9E",
    "deq_block": "#2E86AB",
    "fprm": "#00798C",
    "looped_bptt": "#5C6B73",
    "ut_act": "#8FA6B2",
    "transformer": "#B0B0B0",
}
LABEL = {
    "fpsa_r": "FPSA-R (ours)",
    "fpsa_r_nested": "FPSA-R, nested solver",
    "fpsa_r_bptt": "FPSA-R, BPTT",
    "fpsa_r_onestep": "FPSA-R, 1-step phantom",
    "fpsa_r_nomask": "FPSA-R, unmasked adjoint",
    "fpsa_r_neumann": "FPSA-R, Neumann adjoint",
    "fpsa_r_nospec": "FPSA-R, no spectral norm",
    "deq_block": "DEQ block (no in-layer FPSA)",
    "fprm": "FPRM (trunc. BPTT)",
    "looped_bptt": "Looped Transformer (BPTT)",
    "ut_act": "Universal Transformer + ACT",
    "transformer": "Transformer (non-recursive)",
}
ORDER = ["fpsa_r", "deq_block", "fprm", "looped_bptt", "ut_act", "transformer"]
ABL_ORDER = ["fpsa_r", "fpsa_r_nested", "fpsa_r_bptt", "fpsa_r_onestep",
             "fpsa_r_nomask", "fpsa_r_neumann", "fpsa_r_nospec"]


def style(ax, title=None, xlabel=None, ylabel=None, legend=True, grid="y"):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BBBBBB")
    ax.tick_params(colors="#555555", labelsize=9)
    if grid:
        ax.grid(axis=grid, color="#E8E8E8", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11, color="#222222", pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color="#444444")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color="#444444")
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8.5, labelcolor="#333333")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figure -> {name}.png/.svg")


def write_table(name, header, rows, caption=""):
    os.makedirs(TAB, exist_ok=True)
    md = []
    if caption:
        md.append(f"**{caption}**\n")
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        md.append("| " + " | ".join(str(c) for c in r) + " |")
    text = "\n".join(md) + "\n"
    with open(os.path.join(TAB, name + ".md"), "w") as f:
        f.write(text)
    with open(os.path.join(TAB, name + ".csv"), "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(c).replace("**", "") for c in r) + "\n")
    print(f"  table   -> {name}.md/.csv")
    return text


def mean_sd(xs):
    if not xs:
        return float("nan"), 0.0
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def fmt(m, s, prec=2):
    return f"{m:.{prec}f} ± {s:.{prec}f}" if s else f"{m:.{prec}f}"


# ===========================================================================
# task results
# ===========================================================================
def load_runs(pattern):
    runs = defaultdict(list)
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                r = json.load(f)
        except Exception:
            continue
        runs[(r["task"], r["arch"])].append(r)
    return runs


def task_tables(runs, tag, order):
    tasks = sorted({t for t, _ in runs})
    out = []
    for task in tasks:
        rows = []
        for arch in order:
            rs = runs.get((task, arch))
            if not rs:
                continue
            em, em_s = mean_sd([r["final"]["exact_match"] for r in rs])
            tk, tk_s = mean_sd([r["final"]["token_acc"] for r in rs])
            mem, _ = mean_sd([r["activation_mb"] for r in rs])
            stp, _ = mean_sd([r["step_time_s"] for r in rs])
            it, _ = mean_sd([r["final"]["mean_iters"] for r in rs])
            rho = [r["spectral_radius"] for r in rs if r.get("spectral_radius")]
            rho_m, _ = mean_sd(rho)
            best = arch == max(
                (a for a in order if runs.get((task, a))),
                key=lambda a: st.mean([r["final"]["exact_match"] for r in runs[(task, a)]]))
            name = f"**{LABEL[arch]}**" if best else LABEL[arch]
            rows.append([name, rs[0]["params"], fmt(tk, tk_s), fmt(em, em_s),
                         f"{mem:.1f}", f"{stp:.3f}", f"{it:.1f}",
                         f"{rho_m:.2f}" if rho else "-", len(rs)])
        if rows:
            out.append(write_table(
                f"{tag}_{task}",
                ["Model", "Params", "Token acc (%)", "Exact match (%)",
                 "Act. mem (MB)", "s / step", "Eval iters", "rho", "seeds"],
                rows, caption=f"{task}: mean ± sd over seeds. "
                              f"Activation memory is bytes autograd stores for one "
                              f"training step; rho is the measured spectral radius "
                              f"of the update map at the solution."))
    return out


def fig_task_bars(runs, name="fig_task_accuracy"):
    tasks = sorted({t for t, _ in runs})
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.6), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        archs = [a for a in ORDER if runs.get((task, a))]
        ms, ss = [], []
        for a in archs:
            m, s = mean_sd([r["final"]["exact_match"] for r in runs[(task, a)]])
            ms.append(m)
            ss.append(s)
        y = range(len(archs))
        ax.barh(list(y), ms, xerr=ss, color=[C[a] for a in archs], height=0.62,
                error_kw=dict(ecolor="#888888", lw=1, capsize=2.5), zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels([LABEL[a] for a in archs], fontsize=8.5)
        ax.invert_yaxis()
        for i, (m, s) in enumerate(zip(ms, ss)):
            ax.text(m + max(ms) * 0.015 + s, i, f"{m:.1f}", va="center",
                    fontsize=8.5, color="#333333")
        style(ax, title=task, xlabel="exact match (%)", legend=False, grid="x")
        ax.set_xlim(0, max(ms) * 1.22 if max(ms) > 0 else 1)
    fig.suptitle("Exact-match accuracy on reasoning tasks", fontsize=12.5,
                 x=0.02, ha="left", y=1.04, color="#111111")
    save(fig, name)


def fig_test_time_scaling(runs, name="fig_test_time_scaling"):
    tasks = sorted({t for t, _ in runs})
    tasks = [t for t in tasks if any(runs[(t, a)][0].get("scaling") for a in ORDER
                                     if runs.get((t, a)))]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.0 * len(tasks), 3.6), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for a in ORDER:
            rs = runs.get((task, a))
            if not rs or not rs[0].get("scaling"):
                continue
            iters = sorted(int(k) for k in rs[0]["scaling"])
            ys, es = [], []
            for T in iters:
                m, s = mean_sd([r["scaling"][str(T)]["exact_match"] for r in rs
                                if str(T) in r.get("scaling", {})])
                ys.append(m)
                es.append(s)
            lw = 2.4 if a == "fpsa_r" else 1.5
            ax.plot(iters, ys, marker="o", ms=4, lw=lw, color=C[a], label=LABEL[a],
                    zorder=4 if a == "fpsa_r" else 3)
            ax.fill_between(iters, [y - e for y, e in zip(ys, es)],
                            [y + e for y, e in zip(ys, es)], color=C[a], alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)}"))
        style(ax, title=task, xlabel="test-time iterations (train budget = 8)",
              ylabel="exact match (%)", legend=(task == tasks[0]))
    fig.suptitle("Accuracy vs. test-time compute", fontsize=12.5, x=0.02,
                 ha="left", y=1.04, color="#111111")
    save(fig, name)


def fig_learning_curves(runs, name="fig_learning_curves"):
    tasks = sorted({t for t, _ in runs})
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.0 * len(tasks), 3.5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for a in ORDER:
            rs = runs.get((task, a))
            if not rs:
                continue
            steps = [h["step"] for h in rs[0]["history"]]
            n = min(len(r["history"]) for r in rs)
            ys = [st.mean([r["history"][i]["exact_match"] for r in rs]) for i in range(n)]
            ax.plot(steps[:n], ys, lw=2.4 if a == "fpsa_r" else 1.4, color=C[a],
                    label=LABEL[a], zorder=4 if a == "fpsa_r" else 3)
        style(ax, title=task, xlabel="training step", ylabel="exact match (%)",
              legend=(task == tasks[0]))
    fig.suptitle("Learning curves", fontsize=12.5, x=0.02, ha="left", y=1.04,
                 color="#111111")
    save(fig, name)


def fig_generalization(runs, name="fig_generalization"):
    """Held-out harder instances (bigger mazes / more blanks / longer sequences)."""
    panels = []
    for (task, arch), rs in runs.items():
        if rs[0].get("extra"):
            panels.append(task)
    panels = sorted(set(panels))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 3.5), squeeze=False)
    for ax, task in zip(axes[0], panels):
        keys = None
        for a in ORDER:
            rs = runs.get((task, a))
            if not rs or not rs[0].get("extra"):
                continue
            keys = keys or list(rs[0]["extra"])
            ys = [st.mean([r["extra"][k]["exact_match"] for r in rs if k in r["extra"]])
                  for k in keys]
            ax.plot(range(len(keys)), ys, marker="o", ms=4,
                    lw=2.4 if a == "fpsa_r" else 1.4, color=C[a], label=LABEL[a])
        if keys:
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, fontsize=8.5)
        style(ax, title=task, xlabel="held-out difficulty", ylabel="exact match (%)",
              legend=(task == panels[0]))
    fig.suptitle("Generalisation to harder instances than trained on",
                 fontsize=12.5, x=0.02, ha="left", y=1.04, color="#111111")
    save(fig, name)


# ===========================================================================
# mechanism figures
# ===========================================================================
def mech(name):
    p = os.path.join(MECH, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def fig_memory(name="fig_activation_memory"):
    d = mech("m2_activation_memory")
    if not d:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7))
    colors = {"FPSA-R (implicit)": C["fpsa_r"], "FPRM (trunc. BPTT K=6)": C["fprm"],
              "Looped Transformer (BPTT)": C["looped_bptt"], "UT + ACT (BPTT)": C["ut_act"]}
    for k, series in d["activation_mb"].items():
        xs = d["iters"]
        ys = [series[str(T)] for T in xs]
        ax.plot(xs, ys, marker="o", ms=4, lw=2.4 if "FPSA-R" in k else 1.5,
                color=colors.get(k, "#888"), label=k)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)}"))
    style(ax, title="Activation memory is flat in loop depth",
          xlabel="forward fixed-point iterations $T$",
          ylabel="stored activations (MB)")

    base = d["activation_mb"]["FPSA-R (implicit)"]
    for k, series in d["activation_mb"].items():
        if "FPSA-R" in k:
            continue
        xs = d["iters"]
        ys = [series[str(T)] / base[str(T)] for T in xs]
        ax2.plot(xs, ys, marker="o", ms=4, lw=1.8, color=colors.get(k, "#888"), label=k)
    ax2.axhline(1.0, color=C["fpsa_r"], lw=2.4, label="FPSA-R (ours)")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)}"))
    style(ax2, title="Memory relative to FPSA-R",
          xlabel="forward fixed-point iterations $T$", ylabel="x FPSA-R memory")
    save(fig, name)


def table_memory():
    d = mech("m2_activation_memory")
    if not d:
        return
    keys = list(d["activation_mb"])
    rows = []
    for T in d["iters"]:
        r = [T] + [f"{d['activation_mb'][k][str(T)]:.1f}" for k in keys]
        best = d["activation_mb"]["FPSA-R (implicit)"][str(T)]
        worst = d["activation_mb"]["Looped Transformer (BPTT)"][str(T)]
        r.append(f"**{worst/best:.1f}x**")
        rows.append(r)
    write_table("mech_activation_memory", ["T"] + keys + ["FPSA-R saving vs BPTT"], rows,
                caption="Activation memory (MB) stored for one training step, "
                        "measured with autograd saved-tensor hooks (seq=16, batch=32, d=128).")


def fig_gradient_fidelity(name="fig_gradient_fidelity"):
    d1, d2 = mech("m1_gradient_fidelity"), mech("m1b_fidelity_vs_contraction")
    if not d1:
        return
    ncol = 2 if d2 else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.4 * ncol, 3.8), squeeze=False)
    ax = axes[0][0]
    names = list(d1["results"])
    vals = [st.mean(d1["results"][n]["rel"]) for n in names]
    cols = [C["fpsa_r"] if "FPSA-R" in n else
            (C["fprm"] if "truncated" in n else C["looped_bptt"]) for n in names]
    y = range(len(names))
    ax.barh(list(y), vals, color=cols, height=0.62, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    hi = max(vals)
    for i, v in enumerate(vals):
        inside = v > hi * 0.5
        ax.text(v * (0.92 if inside else 1.12), i, f"{v:.4f}", va="center",
                ha="right" if inside else "left", fontsize=8.3,
                color="white" if inside else "#333333")
    style(ax, title=f"Gradient error vs exact BPTT (T={d1['forward_iters']})",
          xlabel="relative error  $\\|g-g^*\\|/\\|g^*\\|$", legend=False, grid="x")

    if d2:
        ax2 = axes[0][1]
        kshades = {"trunc-BPTT K=1": "#B9C6CE", "trunc-BPTT K=2": "#8FA6B2",
                   "trunc-BPTT K=4": "#5C6B73", "trunc-BPTT K=8": "#2E86AB"}
        import math as _m
        for k, v in d2["results"].items():
            ours = "implicit" in k
            pts = [(x, y) for x, y in zip(v["rho"], v["rel"])
                   if _m.isfinite(x) and _m.isfinite(y)]
            v = {"rho": [p[0] for p in pts], "rel": [p[1] for p in pts]}
            ax2.plot(v["rho"], v["rel"], marker="o", ms=4,
                     lw=2.8 if ours else 1.5, ls="-" if ours else "--",
                     color=C["fpsa_r"] if ours else kshades.get(k, "#999999"),
                     label=k, zorder=5 if ours else 3)
        ax2.set_yscale("log")
        style(ax2, title="Gradient error vs contraction factor",
              xlabel=r"measured spectral radius $\rho$",
              ylabel="relative gradient error")
    save(fig, name)


def table_gradient_fidelity():
    d = mech("m1_gradient_fidelity")
    if not d:
        return
    rows = []
    for n, v in d["results"].items():
        cm, cs = mean_sd(v["cos"])
        rm, rs_ = mean_sd(v["rel"])
        star = "**" if "FPSA-R implicit (Anderson)" == n else ""
        rows.append([f"{star}{n}{star}", f"{cm:.6f} ± {cs:.6f}", f"{rm:.4f} ± {rs_:.4f}"])
    write_table("mech_gradient_fidelity",
                ["Gradient scheme", "cosine vs exact", "relative error"], rows,
                caption=f"Fidelity against {d['reference']}, forward budget "
                        f"T={d['forward_iters']}, {d['n_seeds']} random inits.")


def fig_contraction(name="fig_contraction_dynamics"):
    d = mech("m3_contraction_dynamics")
    if not d:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7))
    shades = ["#B0B0B0", "#8FA6B2", "#2E86AB", C["fpsa_r"]]
    for (lam, rec), col in zip(sorted(d["curves"].items(), key=lambda kv: float(kv[0])), shades):
        ax.plot(rec["step"], rec["rho"], lw=2.4 if float(lam) >= 50 else 1.6,
                color=col, label=f"$\\lambda={lam}$")
        ax2.plot(rec["step"], rec["eval_residual"], lw=2.4 if float(lam) >= 50 else 1.6,
                 color=col, label=f"$\\lambda={lam}$")
    ax.axhline(1.0, color="#D1495B", ls=":", lw=1.4)
    ax.text(0.02, 1.03, "loss of contraction: no fixed point exists",
            transform=ax.get_yaxis_transform(), fontsize=8, color="#D1495B")
    style(ax, title="Spectral radius of the loop during training",
          xlabel="training step", ylabel=r"$\rho(\partial G/\partial s)$")
    ax2.set_yscale("log")
    style(ax2, title="Forward residual at evaluation",
          xlabel="training step", ylabel="relative residual after 64 iters")
    save(fig, name)


def fig_adjoint(name="fig_adjoint_solver"):
    d = mech("m4_adjoint_solver")
    if not d:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.8))
    shades = ["#F0B8C0", "#DE7183", "#D1495B", "#8E2233", "#5E0F1C"]
    keys = list(d["curves"])
    for i, k in enumerate(keys):
        c = d["curves"][k]
        col = shades[i % len(shades)]
        ax.plot(range(1, len(c["neumann"]) + 1), c["neumann"], lw=1.3, ls="--", color=col)
        ax.plot(range(1, len(c["anderson"]) + 1), c["anderson"], lw=2.2, color=col,
                label=f"$\\rho$={k}")
    ax.set_yscale("log")
    ax.plot([], [], color="#666666", ls="--", lw=1.3, label="Neumann (dashed)")
    ax.plot([], [], color="#666666", lw=2.2, label="Anderson (solid)")
    style(ax, title="Backward linear solve on the model Jacobian",
          xlabel="VJP evaluations", ylabel="relative error of the adjoint")

    # VJPs needed to reach a 1e-4 adjoint
    def first_below(seq, tol=1e-4):
        for i, v in enumerate(seq, 1):
            if v < tol:
                return i
        return None
    rr, na, nn = [], [], []
    for k in keys:
        c = d["curves"][k]
        a, n = first_below(c["anderson"]), first_below(c["neumann"])
        if a and n:
            rr.append(float(k))
            na.append(a)
            nn.append(n)
    w = 0.36
    xs = range(len(rr))
    ax2.bar([x - w / 2 for x in xs], nn, w, color="#8FA6B2", label="Neumann", zorder=3)
    ax2.bar([x + w / 2 for x in xs], na, w, color=C["fpsa_r"], label="Anderson (ours)", zorder=3)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([f"{r:.2f}" for r in rr])
    for x, (a, n) in enumerate(zip(na, nn)):
        ax2.text(x, max(a, n) + 0.4, f"{n/a:.2f}x", ha="center", fontsize=8.5,
                 color="#333333")
    style(ax2, title=r"VJPs to reach a $10^{-4}$ adjoint",
          xlabel=r"spectral radius $\rho$", ylabel="VJP evaluations")
    save(fig, name)

    if rr:
        write_table("mech_adjoint_solver",
                    ["rho", "Neumann VJPs", "Anderson VJPs", "speedup"],
                    [[f"{r:.3f}", n, a, f"**{n/a:.2f}x**"] for r, a, n in zip(rr, na, nn)],
                    caption="VJP evaluations needed for a relative adjoint error below "
                            "1e-4, measured on the trained model's own Jacobian.")


def fig_token_convergence(name="fig_token_convergence"):
    d = mech("m6_token_convergence")
    if not d:
        return
    import numpy as np
    dist = np.array(d["distance"])
    blank = np.array(d["is_blank"])
    order = np.argsort(~blank)          # blanks (the cells that need solving) first
    dist = dist[order]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(np.log10(dist + 1e-8), aspect="auto", cmap="magma_r",
                   interpolation="nearest")
    n_blank = int(blank.sum())
    ax.axhline(n_blank - 0.5, color="#2E86AB", lw=1.4)
    ax.text(0.5, n_blank - 1.5, "blank cells (must be solved)", color="#2E86AB",
            fontsize=8.5, va="bottom")
    ax.text(0.5, n_blank + 1.5, "given cells", color="#2E86AB", fontsize=8.5, va="top")
    cb = fig.colorbar(im, ax=ax, fraction=0.035)
    cb.set_label(r"$\log_{10}$ relative distance to $z^\star$", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    style(ax, title="Per-token convergence inside a Sudoku grid",
          xlabel="fixed-point iteration", ylabel="grid cell (sorted)",
          legend=False, grid=None)
    save(fig, name)


def table_solver_cost():
    d = mech("m5_solver_cost")
    if not d:
        return
    m = d["modes"]
    rows = [["Nested (inner loop inside each outer step)", m["nested"]["outer_iters"],
             f"{m['nested']['attention_calls']:.0f}", f"{m['nested']['wall_clock_s']*1000:.0f}",
             f"{m['nested']['final_residual']:.1e}"],
            ["**Joint (ours)**", m["joint"]["outer_iters"],
             f"**{m['joint']['attention_calls']:.0f}**",
             f"**{m['joint']['wall_clock_s']*1000:.0f}**",
             f"{m['joint']['final_residual']:.1e}"]]
    write_table("mech_solver_cost",
                ["Two-level solver", "outer iters", "attention calls", "ms / forward",
                 "final residual"], rows,
                caption=f"Cost of reaching residual < {d['tol']} on Sudoku (batch 32). "
                        f"Both solvers reach the same equilibrium; the joint "
                        f"formulation gets there with "
                        f"{d['modes']['speedup']['attention_calls']:.1f}x fewer attention "
                        f"calls and {d['modes']['speedup']['wall_clock']:.2f}x less wall clock.")


def table_rank_collapse():
    d = mech("m7_rank_collapse")
    if not d:
        return
    a, b = d["effective_rank"]["with_residual"], d["effective_rank"]["without_residual"]
    am, asd = mean_sd(a)
    bm, bsd = mean_sd(b)
    write_table("mech_rank_collapse",
                ["Inner FPSA map", "effective rank of the fixed point", "of max"],
                [[f"**u <- x + W_O A(u) V**  (FPSA-R)", f"**{am:.1f} ± {asd:.1f}**", d["n_tokens"]],
                 ["u <- W_O A(u) V  (no in-loop residual)", f"{bm:.1f} ± {bsd:.1f}", d["n_tokens"]]],
                caption=f"Effective rank (entropy of the singular-value spectrum) of the "
                        f"converged inner attention state, {d['n_seeds']} random inits, "
                        f"{d['iters']} iterations. Without the input re-injection the "
                        f"row-stochastic attention operator averages tokens together and "
                        f"the fixed point loses {100*(1-bm/am):.0f}% of its effective rank.")


def fig_architecture(name="fig_architecture"):
    """Schematic of where the loop sits in each family of model."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    panels = [
        ("Looped Transformer / FPRM", ["Attn", "MLP"], "loop the whole block",
         C["looped_bptt"], True, True),
        ("FPSA (in-layer only)", ["Attn", "MLP"], "loop attention, one pass through the block",
         C["deq_block"], True, False),
        ("FPSA-R (ours)", ["Attn", "MLP"], "joint equilibrium over both",
         C["fpsa_r"], True, True),
    ]
    for ax, (title, boxes, sub, col, inner, outer) in zip(axes, panels):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        xs = [2.2, 6.0]
        for x, b in zip(xs, boxes):
            ax.add_patch(plt.Rectangle((x, 2.4), 2.4, 1.5, facecolor="white",
                                       edgecolor="#555555", lw=1.4, zorder=3))
            ax.text(x + 1.2, 3.15, b, ha="center", va="center", fontsize=10.5,
                    color="#222222", zorder=4)
        ax.annotate("", xy=(6.0, 3.15), xytext=(4.6, 3.15),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3))
        ax.annotate("", xy=(2.2, 3.15), xytext=(0.7, 3.15),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3))
        ax.annotate("", xy=(9.5, 3.15), xytext=(8.4, 3.15),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3))
        if inner and title != "Looped Transformer / FPRM":
            ax.annotate("", xy=(2.4, 2.3), xytext=(4.4, 2.3),
                        arrowprops=dict(arrowstyle="->", color=col, lw=2.0,
                                        connectionstyle="arc3,rad=0.55"))
            ax.text(3.4, 1.15, "in-layer FPSA\n(Q,K from $u$; $V$ frozen)", ha="center",
                    fontsize=8, color=col)
        if outer:
            ax.annotate("", xy=(1.6, 4.3), xytext=(8.6, 4.3),
                        arrowprops=dict(arrowstyle="->", color=col, lw=2.0,
                                        connectionstyle="arc3,rad=0.35"))
            ax.text(5.1, 5.35, "outer recursion", ha="center", fontsize=8, color=col)
        ax.set_title(title, fontsize=11, color="#111111", loc="left")
        ax.text(0.0, 0.2, sub, fontsize=8.5, color="#666666", transform=ax.transAxes)
    save(fig, name)


# ===========================================================================
def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    print("mechanism results")
    table_gradient_fidelity()
    table_memory()
    table_solver_cost()
    table_rank_collapse()
    fig_gradient_fidelity()
    fig_memory()
    fig_contraction()
    fig_adjoint()
    fig_token_convergence()
    fig_architecture()

    print("task results")
    runs = load_runs(os.path.join(RES, "runs", "*.json"))
    if runs:
        task_tables(runs, "main", ORDER)
        fig_task_bars(runs)
        fig_test_time_scaling(runs)
        fig_learning_curves(runs)
        fig_generalization(runs)
    abl = load_runs(os.path.join(RES, "ablation", "*.json"))
    if abl:
        task_tables(abl, "ablation", ABL_ORDER)
    print("done")


if __name__ == "__main__":
    main()

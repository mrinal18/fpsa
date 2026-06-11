import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
 "fixed_ffn s0": ("runs/abl_fixed_ffn_seed0", "tab:blue", "-"),
 "fixed_ffn s1": ("runs/abl_fixed_ffn_seed1", "tab:blue", "--"),
 "blended s0": ("runs/abl_blended_seed0", "tab:orange", "-"),
 "blended s1": ("runs/abl_blended_seed1", "tab:orange", "--"),
 "evolving(gv2) s0": ("runs/abl_evolving_gv2_seed0", "tab:green", "-"),
 "evolving(gv2) s1": ("runs/abl_evolving_gv2_seed1", "tab:green", "--"),
 "fixed s0": ("runs/abl_fixed_seed0", "tab:red", "-"),
 "evolving(gv1) s0": ("runs/abl_evolving_seed0", "tab:purple", "-"),
 "fixed_ffn SN-bug s0": ("runs/abl_ffn_bugsn_seed0", "k", "-"),
}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for label, (run, c, ls) in RUNS.items():
    rows = list(csv.DictReader(open(os.path.join(run, "train_log.csv"))))
    ax.plot([int(r["step"]) for r in rows], [float(r["test_tok_acc"]) for r in rows],
            color=c, ls=ls, label=label, lw=1.5)
ax.set_xlabel("train step"); ax.set_ylabel("test token acc"); ax.set_ylim(0.45, 1.02)
ax.set_title("Value-mode ablation + SN-bug repro (prefix parity L=16)")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

ax = axes[1]
lg = json.load(open("runs/length_gen.json"))
colors = {"fixed_ffn": "tab:blue", "blended": "tab:orange", "evolving": "tab:green"}
marks = {16: "o", 20: "s", 24: "^"}
for name in colors:
    for L in [16, 20, 24]:
        ks, accs = [], []
        for k in [8, 16, 24, 48]:
            ks.append(k); accs.append(lg[f"{name}|L{L}|k{k}"][0])
        ax.plot(ks, accs, color=colors[name], marker=marks[L], ms=5,
                ls={16: "-", 20: "--", 24: ":"}[L],
                label=f"{name} L={L}" if True else None, lw=1.3)
ax.set_xlabel("inference iteration budget k"); ax.set_ylabel("test token acc")
ax.set_title("Anytime compute & length generalization (trained on L=16)")
ax.legend(fontsize=6.5, ncol=3); ax.grid(alpha=0.3); ax.set_ylim(0.45, 1.03)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/ablation_summary.png", dpi=130)
print("saved")

for label, (run, _, _) in RUNS.items():
    f = os.path.join(run, "final.json")
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"{label:<22} tok={d['final_test_tok_acc']:.4f} seq={d['final_test_seq_acc']:.4f} steps={d['steps_run']} params={d['params']}")

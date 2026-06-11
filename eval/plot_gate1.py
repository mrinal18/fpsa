import json, os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.model import FPSASeqModel
from data.parity import make_parity
from utils.logging_utils import set_seed

SEEDS = [0, 1, 2]
RUNS = {s: f"runs/gate1_parity_L16_k24_seed{s}" for s in SEEDS}
cfg = json.load(open(os.path.join(RUNS[0], "config.json")))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# --- (a) per-iteration residual curves across training (seed 0) ---
ax = axes[0, 0]
snaps = json.load(open(os.path.join(RUNS[0], "residual_snapshots.json")))
steps_avail = sorted(int(k) for k in snaps)
pick = [steps_avail[0]] + [s for s in [300, 800, 1200] if s in steps_avail] + [steps_avail[-1]]
for s in dict.fromkeys(pick):
    c = snaps[str(s)]
    ax.semilogy(range(1, len(c) + 1), c, marker="o", ms=3, label=f"train step {s}")
ax.axhline(cfg["tol"], color="k", ls="--", lw=1, label=f"tol={cfg['tol']}")
ax.set_xlabel("FPI iteration k"); ax.set_ylabel("mean rel residual ||Δz||/||z||")
ax.set_title("(a) Forward convergence across training (seed 0, eval batch)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# --- (b) accuracy curves, 3 seeds ---
ax = axes[0, 1]
for s in SEEDS:
    rows = list(csv.DictReader(open(os.path.join(RUNS[s], "train_log.csv"))))
    st = [int(r["step"]) for r in rows]
    ax.plot(st, [float(r["test_tok_acc"]) for r in rows], label=f"seed {s} token")
    ax.plot(st, [float(r["test_seq_acc"]) for r in rows], ls=":", alpha=0.7,
            label=f"seed {s} seq")
ax.set_xlabel("train step"); ax.set_ylabel("test accuracy"); ax.set_ylim(0.4, 1.02)
ax.set_title("(b) Held-out accuracy (3 seeds)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# --- (c) sigma_max(J) estimate over training ---
ax = axes[1, 0]
for s in SEEDS:
    rows = list(csv.DictReader(open(os.path.join(RUNS[s], "train_log.csv"))))
    ax.plot([int(r["step"]) for r in rows], [float(r["jac_sigma"]) for r in rows],
            marker=".", ms=4, label=f"seed {s}")
ax.axhline(1.0, color="r", ls="-", lw=1, label="divergence (σ=1)")
ax.axhline(cfg["jac_rho_target"], color="g", ls="--", lw=1,
           label=f"penalty target ρ={cfg['jac_rho_target']}")
ax.set_xlabel("train step"); ax.set_ylabel("σ̂_max(J) at z*")
ax.set_title("(c) Contraction factor during training"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# --- (d) trained models: extended-budget convergence ---
ax = axes[1, 1]
_, (xte, yte) = make_parity(cfg["n_train"], cfg["n_test"], cfg["seq_len"],
                            seed=cfg["data_seed"])
accs = []
for s in SEEDS:
    set_seed(s)
    m = FPSASeqModel(2, 2, d_model=cfg["d_model"], num_heads=cfg["num_heads"],
                     value_mode=cfg["value_mode"], damping=cfg["damping"],
                     max_iter=48, tol=0.0, backward=cfg["backward"],
                     neumann_steps=cfg["neumann_steps"],
                     use_spectral_norm=cfg["spectral_norm"], use_rope=True,
                     max_seq_len=cfg["seq_len"], ffn_mult=cfg["ffn_mult"])
    m.load_state_dict(torch.load(os.path.join(RUNS[s], "model.pt")))
    m.eval()
    with torch.no_grad():
        logits, stt = m(xte[:512], max_iter=48)
        tok = (logits.argmax(-1) == yte[:512]).float().mean().item()
    ax.semilogy(range(1, len(stt.rel_residual_mean) + 1), stt.rel_residual_mean,
                label=f"seed {s} (tok acc@48it {tok:.4f})")
    accs.append(tok)
ax.axhline(cfg["tol"], color="k", ls="--", lw=1)
ax.set_xlabel("FPI iteration k"); ax.set_ylabel("mean rel residual")
ax.set_title("(d) Trained models, 48-iteration budget"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
os.makedirs("/mnt/user-data/outputs", exist_ok=True)
out = "/mnt/user-data/outputs/gate1_convergence.png"
plt.savefig(out, dpi=130)
print("saved", out)

# --- empirical contraction slope of trained maps ---
for s in SEEDS:
    snapsf = json.load(open(os.path.join(RUNS[s], "final.json")))
    print(f"seed {s}: final_tok={snapsf['final_test_tok_acc']:.4f} "
          f"final_seq={snapsf['final_test_seq_acc']:.4f} steps={snapsf['steps_run']}")
toks = [json.load(open(os.path.join(RUNS[s], 'final.json')))['final_test_tok_acc'] for s in SEEDS]
seqs = [json.load(open(os.path.join(RUNS[s], 'final.json')))['final_test_seq_acc'] for s in SEEDS]
print(f"token acc: {np.mean(toks)*100:.2f} ± {np.std(toks)*100:.2f}")
print(f"seq   acc: {np.mean(seqs)*100:.2f} ± {np.std(seqs)*100:.2f}")

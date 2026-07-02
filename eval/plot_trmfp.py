import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.trmfp import TRMFPModel
from data.parity import make_parity
from utils.logging_utils import set_seed

rows = list(csv.DictReader(open("runs/trmfp_parity_seed0/train_log.csv")))
g = lambda k: np.array([float(r[k]) for r in rows])
s, tok, seq, jres, rho, dis = g("step"), g("tok"), g("seq"), g("joint_res"), g("rho"), g("disagree")

# full-test eval at the stopping checkpoint
set_seed(0)
m = TRMFPModel(2, 2, d_model=64, num_heads=4, n_inner=8, T_outer=4, tol=1e-3,
               neumann_steps=12, max_seq_len=16)
ck = torch.load("runs/trmfp_parity_seed0/checkpoint.pt", weights_only=False)
m.load_state_dict(ck["m"]); m.eval()
_, (xte, yte) = make_parity(20000, 4000, 16, seed=1234)
ct = cs = 0
with torch.no_grad():
    for i in range(0, 4000, 512):
        lg, _ = m(xte[i:i+512]); p = lg.argmax(-1)
        ct += (p == yte[i:i+512]).sum().item(); cs += (p == yte[i:i+512]).all(1).sum().item()
full_tok, full_seq = ct / yte.numel(), cs / 4000

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(s, tok, label="token"); ax[0].plot(s, seq, ls=":", label="sequence")
ax[0].set_title(f"TRM-FP accuracy (stopped @ {int(s[-1])}; full-test tok {full_tok:.3f})")
ax[0].set_xlabel("step"); ax[0].set_ylim(0, 1.02); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(s, jres, color="tab:red", label="joint residual")
ax[1].plot(s, rho, color="tab:purple", label="rho(comparison)")
ax[1].axhline(1.0, color="k", ls="--", lw=0.8)
ax[1].set_yscale("log"); ax[1].set_title("Convergence collapse (zero machinery)")
ax[1].set_xlabel("step"); ax[1].legend(); ax[1].grid(alpha=0.3)
r = np.corrcoef(dis, rho)[0, 1]
ax[2].scatter(dis, rho, c=s, cmap="viridis", s=28)
ax[2].set_xlabel("disagreement  mean||LN(z)-LN(x)||_inf"); ax[2].set_ylabel("rho")
ax[2].set_title(f"Prediction test: rho vs disagreement (Pearson r = {r:.3f})")
ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/trmfp_results.png", dpi=130)
json.dump({"stopped_at": int(s[-1]), "full_tok": full_tok, "full_seq": full_seq,
           "pearson_rho_disagree": float(r), "final_rho": float(rho[-1]),
           "final_joint_res": float(jres[-1])},
          open("runs/trmfp_parity_seed0/final_truncated.json", "w"), indent=2)
print(f"full-test tok {full_tok:.4f} seq {full_seq:.4f} | Pearson(rho,disagree) = {r:.3f}")

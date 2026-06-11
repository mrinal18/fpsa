"""Length generalization + anytime-compute probe.

Models trained on L=16 prefix parity are evaluated on L in {16, 20, 24}
at iteration budgets k in {8, 16, 24, 48} (tol exit still active, so
reported iters-used shows adaptive depth). Equilibrium hypothesis: more
inference iterations should help, especially beyond training length.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.model import FPSASeqModel
from data.parity import make_parity
from utils.logging_utils import set_seed

torch.set_flush_denormal(True)
MODELS = {
    "fixed_ffn": ("runs/abl_fixed_ffn_seed0", "fixed_ffn"),
    "blended":   ("runs/abl_blended_seed0", "blended"),
    "evolving":  ("runs/abl_evolving_gv2_seed0", "evolving"),
}
BUDGETS = [8, 16, 24, 48]
LENGTHS = [16, 20, 24]
N_TEST = 1024

results = {}
for name, (run, vm) in MODELS.items():
    cfg = json.load(open(os.path.join(run, "config.json")))
    set_seed(0)
    m = FPSASeqModel(2, 2, d_model=cfg["d_model"], num_heads=cfg["num_heads"],
                     value_mode=vm, damping=cfg["damping"], max_iter=48,
                     tol=cfg["tol"], backward=cfg["backward"],
                     neumann_steps=cfg["neumann_steps"],
                     use_spectral_norm=cfg["spectral_norm"], use_rope=True,
                     max_seq_len=32, ffn_mult=cfg["ffn_mult"])
    sd = torch.load(os.path.join(run, "model.pt"))
    m.load_state_dict(sd)
    m.eval()
    for L in LENGTHS:
        _, (xte, yte) = make_parity(0, N_TEST, L, seed=777 + L)
        for k in BUDGETS:
            accs, iters = [], []
            with torch.no_grad():
                for i in range(0, N_TEST, 512):
                    logits, st = m(xte[i:i+512], max_iter=k)
                    accs.append((logits.argmax(-1) == yte[i:i+512]).float().mean().item())
                    iters.append(st.iterations)
            acc = sum(accs)/len(accs)
            it = sum(iters)/len(iters)
            results[f"{name}|L{L}|k{k}"] = (acc, it)
            print(f"{name:<10} L={L}  budget={k:>2}  tok_acc={acc:.4f}  iters_used={it:.1f}")
    print()
json.dump(results, open("runs/length_gen.json", "w"), indent=1)

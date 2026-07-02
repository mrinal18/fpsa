"""TRM-FP validation: stacked-implicit gradcheck + coupled-contraction audit.
float64, fixed iteration budgets (no early stop) for FD determinism."""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.trmfp import TRMFPModel, coupled_audit
from utils.logging_utils import set_seed

torch.set_default_dtype(torch.float64)
set_seed(0)

N_IN, T_OUT = 30, 12   # deep fixed budget -> tight joint fixed point
m = TRMFPModel(vocab_size=2, num_classes=2, d_model=16, num_heads=2,
               n_inner=N_IN, T_outer=T_OUT, tol=0.0, neumann_steps=60,
               neumann_tol=1e-12, max_seq_len=8)
m.eval()
tok = torch.randint(0, 2, (2, 6))
tgt = torch.cumsum(tok, 1) % 2


def loss_of(model, backward):
    lg, rec = model(tok, backward=backward)
    return torch.nn.functional.cross_entropy(lg.view(-1, 2), tgt.view(-1)), rec


def grads(model, backward):
    mm = copy.deepcopy(model); mm.zero_grad()
    l, rec = loss_of(mm, backward)
    l.backward()
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                   for p in mm.parameters()])
    return g, rec

g_b, rec = grads(m, "bptt")
print(f"[fwd] rounds={rec['rounds']} joint_res={rec['joint_res']:.3e} "
      f"gate(bx,by,bz)={[round(v,3) for v in rec['gate_mean'].tolist()]}")
g_n, rec_n = grads(m, "neumann")
rel = ((g_n - g_b).norm() / g_b.norm()).item()
cos = torch.nn.functional.cosine_similarity(g_n, g_b, dim=0).item()
print(f"[stacked-neumann vs BPTT] rel_err={rel:.3e} cos={cos:.8f} "
      f"adjoint_iters={len(rec_n['bwd_res'])}")

# FD vs BPTT on random params (hybrid criterion)
params = list(m.parameters())
cum = torch.cumsum(torch.tensor([p.numel() for p in params]), 0)
set_seed(1)
idxs = torch.randperm(int(cum[-1]))[:14]
eps, worst = 1e-6, 0.0
for fi in idxs.tolist():
    pi = int(torch.searchsorted(cum, fi, right=True))
    loc = fi - (int(cum[pi - 1]) if pi > 0 else 0)
    ls = []
    for sg in (1, -1):
        mm = copy.deepcopy(m)
        with torch.no_grad():
            list(mm.parameters())[pi].view(-1)[loc] += sg * eps
        l, _ = loss_of(mm, "bptt"); ls.append(l.item())
    fd = (ls[0] - ls[1]) / (2 * eps); an = g_b[fi].item()
    err = 0.0 if abs(fd - an) < 1e-9 else abs(fd - an) / max(abs(an), abs(fd), 1e-12)
    worst = max(worst, err)
print(f"[FD vs BPTT] 14 params, hybrid max err {worst:.3e} -> "
      f"{'PASS' if worst < 1e-5 else 'FAIL'}")

est, rho = coupled_audit(m, tok)
print(f"[coupled audit] a={est['a']:.3f} b={est['b']:.3f} c={est['c']:.3f} "
      f"d={est['d']:.3f}  rho(comparison)={rho:.3f} -> "
      f"{'contractive' if rho < 1 else 'NOT certified (norm bound)'}")
ok = worst < 1e-5 and rel < 1e-3
print("\nTRM-FP GRADCHECK:", "PASS" if ok else "FAIL")

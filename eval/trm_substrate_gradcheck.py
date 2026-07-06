"""TRM-substrate gradcheck (float64): FD vs bptt_full vs neumann_k ladder."""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.trm_substrate import TRMSubstrate
from utils.logging_utils import set_seed

torch.set_default_dtype(torch.float64)
set_seed(0)
KW = dict(vocab_size=2, num_classes=2, d_model=16, num_heads=2,
          value_mode="fixed", n_inner=60, T_outer=3, tol=0.0,
          bwd_k=40, max_seq_len=8, alpha=0.5)
m = TRMSubstrate(backward="bptt_full", **KW); m.eval()  # eval LN stats n/a; keep train
m.train()
tok = torch.randint(0, 2, (2, 6)); tgt = torch.cumsum(tok, 1) % 2

def loss_of(model):
    logits, _ = model(tok)
    return sum(torch.nn.functional.cross_entropy(l.view(-1, 2), tgt.view(-1))
               for l in logits) / len(logits)

def grads(model):
    mm = copy.deepcopy(model); mm.zero_grad()
    loss_of(mm).backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                      for p in mm.parameters()])

g_full = grads(m)

for bw, k in [("phantom1", 1), ("neumann_k", 1), ("neumann_k", 5),
              ("neumann_k", 15), ("neumann_k", 40)]:
    mm = copy.deepcopy(m); mm.backward = bw; mm.bwd_k = k
    g = grads(mm)
    rel = ((g - g_full).norm() / g_full.norm()).item()
    cos = torch.nn.functional.cosine_similarity(g, g_full, dim=0).item()
    print(f"[{bw:>9} k={k:>2}] vs bptt_full: rel={rel:.3e} cos={cos:.8f}")

# phantom1 must equal neumann_k=1 exactly
m1 = copy.deepcopy(m); m1.backward = "phantom1"
m2 = copy.deepcopy(m); m2.backward = "neumann_k"; m2.bwd_k = 1
d = (grads(m1) - grads(m2)).abs().max().item()
print(f"[consistency] phantom1 vs neumann_k=1: max abs diff {d:.2e} -> "
      f"{'PASS' if d < 1e-12 else 'FAIL'}")

# FD vs bptt_full (hybrid criterion) — T_outer=1: with deep supervision's
# inter-segment detach, FD measures the TRUE total derivative (values still
# propagate through a detach) while autograd deliberately truncates it, so
# multi-segment FD/analytic disagreement is BY DESIGN. Single segment has
# no detach, so exact agreement is required there.
KW1 = dict(KW); KW1["T_outer"] = 1
set_seed(0)
m = TRMSubstrate(backward="bptt_full", **KW1); m.train()
g_full = grads(m)
params = list(m.parameters())
cum = torch.cumsum(torch.tensor([p.numel() for p in params]), 0)
set_seed(3); idxs = torch.randperm(int(cum[-1]))[:12]
eps, worst = 1e-6, 0.0
for fi in idxs.tolist():
    pi = int(torch.searchsorted(cum, fi, right=True))
    loc = fi - (int(cum[pi-1]) if pi > 0 else 0)
    ls = []
    for sg in (1, -1):
        mm = copy.deepcopy(m)
        with torch.no_grad():
            list(mm.parameters())[pi].view(-1)[loc] += sg * eps
        ls.append(loss_of(mm).item())
    fd = (ls[0] - ls[1]) / (2 * eps); an = g_full[fi].item()
    err = 0.0 if abs(fd - an) < 1e-9 else abs(fd - an) / max(abs(an), abs(fd), 1e-12)
    worst = max(worst, err)
print(f"[FD vs bptt_full] 12 params hybrid max err {worst:.3e} -> "
      f"{'PASS' if worst < 1e-5 else 'FAIL'}")

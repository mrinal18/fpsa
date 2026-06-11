"""Localize the implicit-vs-BPTT gradient discrepancy on a tiny instance.

Builds the dense Jacobian J = df/dz at z*, solves (I - J^T) lam = g exactly,
forms the implicit param gradient with the exact adjoint, and compares
per-parameter against full BPTT.
"""
import copy
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.utils.parametrize as parametrize

from models.model import FPSASeqModel
from utils.logging_utils import set_seed

torch.set_default_dtype(torch.float64)
set_seed(0)

MAX_ITER = 60
model = FPSASeqModel(vocab_size=2, num_classes=2, d_model=16, num_heads=2,
                     value_mode="fixed_ffn", damping=0.5, max_iter=MAX_ITER,
                     tol=0.0, use_spectral_norm=True, use_rope=True, max_seq_len=8)
model.eval()
tokens = torch.randint(0, 2, (2, 6))
targets = torch.cumsum(tokens, dim=1) % 2
names = [n for n, _ in model.named_parameters()]


def loss_at(z, m):
    logits = m.head(m.out_norm(z))
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1))


# ---------- BPTT reference ----------
m1 = copy.deepcopy(model)
x = m1.embed(tokens)
with parametrize.cached():
    static = m1.block.precompute(x)
    z = x
    for _ in range(MAX_ITER):
        z = m1.block.f(z, x, static, None)
loss = loss_at(z, m1)
g_bptt = torch.autograd.grad(loss, list(m1.parameters()), allow_unused=True)

# ---------- implicit with EXACT dense adjoint ----------
m2 = copy.deepcopy(model)
x2 = m2.embed(tokens)
with parametrize.cached():
    static2 = m2.block.precompute(x2)
    with torch.no_grad():
        z2 = x2
        for _ in range(MAX_ITER):
            z2 = m2.block.f(z2, x2, static2, None)
    z_star = z2.detach()

    # dL/dz at z* (treat z as free input to the head)
    z_free = z_star.clone().requires_grad_(True)
    g = torch.autograd.grad(loss_at(z_free, m2), z_free)[0]

    # dense J^T via vjp rows
    B, N, D = z_star.shape
    dim = B * N * D
    z0 = z_star.clone().requires_grad_(True)
    f0 = m2.block.f(z0, x2, static2, None)
    JT = torch.zeros(dim, dim)
    eye = torch.eye(dim)
    for i in range(dim):
        v = eye[i].view(B, N, D)
        JT[:, i] = torch.autograd.grad(f0, z0, v, retain_graph=True)[0].flatten()
    lam = torch.linalg.solve(torch.eye(dim) - JT, g.flatten()).view(B, N, D)
    sr = torch.linalg.eigvals(JT).abs().max().item()
    print(f"spectral radius of J at z*: {sr:.4f}")

    # param grads: head/out_norm direct + block via exact adjoint, embed via x-path
    direct = torch.autograd.grad(loss_at(z_star, m2), list(m2.parameters()),
                                 allow_unused=True, retain_graph=True)
    z_in = z_star.clone().requires_grad_(True)
    f_out = m2.block.f(z_in, x2, static2, None)
    via_f = torch.autograd.grad(f_out, list(m2.parameters()), grad_outputs=lam,
                                allow_unused=True)

g_impl = []
for p, d, v in zip(m2.parameters(), direct, via_f):
    t = (d if d is not None else torch.zeros_like(p)) + \
        (v if v is not None else torch.zeros_like(p))
    g_impl.append(t)

print(f"{'param':<34}{'relerr':>12}{'|bptt|':>12}")
tot_sq, ref_sq = 0.0, 0.0
for n, gb, gi in zip(names, g_bptt, g_impl):
    gb = torch.zeros_like(gi) if gb is None else gb
    e = (gi - gb).norm().item()
    r = gb.norm().item()
    tot_sq += e * e
    ref_sq += r * r
    print(f"{n:<34}{e / (r + 1e-30):>12.3e}{r:>12.3e}")
print(f"\nTOTAL rel err (exact adjoint vs BPTT): {tot_sq**0.5 / ref_sq**0.5:.3e}")

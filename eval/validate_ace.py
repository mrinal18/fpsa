"""ACE validation gate V1+V2 (+component checks + certificate calibration).

V-comp: softnorm/groupsort/SpectralCap satisfy their claimed properties
        (numerical, randomized).
V2    : sup ||J_F|| <= 1 - beta_min, audited by power iteration on the TRUE
        Jacobian at many random (x, z) pairs — not just at z*.
V1    : float64 FD vs Neumann gradients through the solve.
V-cert: T2 certificate ||z_k - z*|| <= ||z_{k+1}-z_k||/(1-rho) holds with
        z* taken as a 300-iteration reference solve.
Run: python3 eval/validate_ace.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.ace_block import ACEBlock, ACESeqModel, softnorm, groupsort
from utils.logging_utils import set_seed

torch.set_flush_denormal(True)


def vcomp():
    set_seed(0)
    # softnorm: 1-Lipschitz + bounded (randomized pairs)
    worst = 0.0
    for _ in range(2000):
        u, v = torch.randn(8) * 10 ** torch.randint(-2, 3, (1,)).item(), None
        v = u + torch.randn(8) * 1e-3
        worst = max(worst, ((softnorm(u) - softnorm(v)).norm() /
                            (u - v).norm()).item())
        assert softnorm(u).norm() < 1.0 + 1e-6
    assert worst <= 1.0 + 1e-4, worst
    # groupsort: norm-preserving
    u = torch.randn(64)
    assert abs(groupsort(u).norm() - u.norm()) < 1e-6
    # SpectralCap: exact <= 1
    lin = torch.nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        lin.weight.mul_(7.0)
    from models.ace_block import sn_exact
    lin = sn_exact(lin)
    assert torch.linalg.matrix_norm(lin.weight, 2).item() <= 1.0 + 1e-5
    print(f"PASS V-comp: softnorm Lip<=1 (worst {worst:.6f}), groupsort "
          f"isometric, SpectralCap exact")


def v2_jacobian_audit(n_points=20, power_iters=30, d=32, heads=4, N=12):
    set_seed(0)
    blk = ACEBlock(d, heads, beta_min=0.2, beta_max=0.9, max_seq_len=N)
    blk.eval()
    rho = blk.rho()
    worst = 0.0
    for p in range(n_points):
        x = torch.randn(1, N, d) * (10 ** ((p % 5) - 2))   # scales 1e-2..1e2
        z = torch.randn(1, N, d) * (10 ** ((p % 3) - 1))
        static = blk.precompute(x)
        z0 = z.clone().requires_grad_(True)
        f0 = blk.f(z0, x, static)
        u = torch.randn_like(z0)
        u = u / u.norm()
        for _ in range(power_iters):
            u = torch.autograd.grad(f0, z0, u, retain_graph=True)[0]
            nu = u.norm()
            u = u / (nu + 1e-30)
        worst = max(worst, nu.item())
    ok = worst <= rho + 1e-4
    print(f"{'PASS' if ok else 'FAIL'} V2: sup sigma(J) over {n_points} random "
          f"(x,z) at mixed scales = {worst:.4f}  (certified bound {rho:.4f})")
    assert ok


def v1_gradcheck(n_fd=12, eps=1e-6):
    """Quantitative T3 test: FD error must respect the truncation bound
    rho^{k+1}/(1-rho) at each adjoint depth k, and decay geometrically."""
    torch.set_default_dtype(torch.float64)
    set_seed(0)
    m = ACESeqModel(vocab_size=2, num_classes=2, d_model=16, num_heads=2,
                    tau=0.5, beta_min=0.2, max_iter=120, tol=0.0,
                    neumann_steps=40, max_seq_len=8)
    m.neumann_tol = 0.0   # disable early break: test pure truncation
    m.eval()
    tokens = torch.randint(0, 2, (2, 6))
    targets = torch.cumsum(tokens, 1) % 2

    def loss_of(model):
        lg, _ = model(tokens, max_iter=80)
        return torch.nn.functional.cross_entropy(lg.view(-1, 2), targets.view(-1))

    base = copy.deepcopy(m)
    rho = m.block.rho()
    params = list(base.parameters())
    cum = torch.cumsum(torch.tensor([p.numel() for p in params]), 0)
    idxs = torch.randperm(int(cum[-1]))[:n_fd]

    def grads_at_depth(k):
        mm = copy.deepcopy(base)
        mm.neumann_tol = 0.0
        mm.neumann_steps = k
        lg = loss_of(mm)
        lg.backward()
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                          for p in mm.parameters()])

    def fd_grad():
        out = {}
        for fi in idxs.tolist():
            pi = int(torch.searchsorted(cum, fi, right=True))
            local = fi - (int(cum[pi - 1]) if pi > 0 else 0)
            ls = []
            for sgn in (+1, -1):
                mm = copy.deepcopy(base)
                with torch.no_grad():
                    list(mm.parameters())[pi].view(-1)[local] += sgn * eps
                ls.append(loss_of(mm).item())
            out[fi] = (ls[0] - ls[1]) / (2 * eps)
        return out

    fd = fd_grad()
    errs = {}
    for k in [10, 25, 60]:
        g = grads_at_depth(k)
        errs[k] = max(abs(fd[fi] - g[fi].item()) /
                      max(abs(fd[fi]), abs(g[fi].item()), 1e-12)
                      for fi in idxs.tolist())
        bound = rho ** (k + 1) / (1 - rho)
        ok_k = errs[k] <= bound * 5 + 1e-7   # 5x slack: bound is per-gradient-norm
        print(f"  k={k:>3}: FD rel err {errs[k]:.2e}  T3 bound {bound:.2e}  "
              f"{'OK' if ok_k else 'VIOLATION'}")
        assert ok_k
    assert errs[60] < 1e-5, errs[60]
    torch.set_default_dtype(torch.float32)
    print(f"PASS V1: gradient error respects T3 truncation bound at every "
          f"depth and reaches {errs[60]:.1e} at k=60")


def vcert(trials=5):
    set_seed(1)
    m = ACESeqModel(vocab_size=2, num_classes=2, d_model=32, num_heads=4,
                    beta_min=0.2, max_iter=300, tol=0.0, max_seq_len=16)
    m.eval()
    rho = m.block.rho()
    viol = 0
    margins = []
    with torch.no_grad():
        for t in range(trials):
            tokens = torch.randint(0, 2, (2, 16))
            x = m.encoder(m.embed(tokens))
            static = m.block.precompute(x)
            z = x.clone()
            traj = [z]
            for _ in range(300):
                z = m.block.f(z, x, static)
                traj.append(z)
            z_star = traj[-1]
            for k in [2, 5, 10, 20, 40]:
                true_d = (traj[k] - z_star).norm(dim=-1).max().item()  # token-max norm
                step = (traj[k + 1] - traj[k]).norm(dim=-1).max().item()
                bound = step / (1 - rho)
                margins.append(bound / max(true_d, 1e-12))
                if true_d > bound * (1 + 1e-6):
                    viol += 1
    print(f"{'PASS' if viol == 0 else 'FAIL'} V-cert: certificate held in "
          f"{trials * 5}/{trials * 5 - viol} checks; bound/true ratio "
          f"min {min(margins):.2f} median {sorted(margins)[len(margins)//2]:.2f}")
    assert viol == 0


if __name__ == "__main__":
    vcomp()
    v2_jacobian_audit()
    v1_gradcheck()
    vcert()
    print("\nACE V1, V2, V-cert, V-comp: ALL PASS")

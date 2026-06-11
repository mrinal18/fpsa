"""ACE validation gate, V1 + V2.

V1: float64 gradcheck — FD vs full BPTT vs Neumann implicit on a tiny ACE.
V2: constraint audit — power-iteration estimate of ||J(z*)|| over many
    random inputs; THEOREM predicts <= rho = 1 - beta_min, always.
"""
import argparse, copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from models.ace import ACESeqModel, ball_project
from utils.logging_utils import set_seed


def build(d=16, heads=2, max_iter=80, tol=0.0, neumann=30, seq=6, beta_min=0.2):
    return ACESeqModel(vocab_size=2, num_classes=2, d_model=d, num_heads=heads,
                       beta_min=beta_min, beta_max=0.8, max_iter=max_iter,
                       tol=tol, neumann_steps=neumann, max_seq_len=seq)


def flat_grads(m):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                      for p in m.parameters()])


def loss_of(m, tok, tgt, K):
    lg, st = m(tok, max_iter=K)
    return torch.nn.functional.cross_entropy(lg.view(-1, 2), tgt.view(-1)), st


def v1_gradcheck(seed=0, K=80, n_fd=16, eps=1e-6):
    torch.set_default_dtype(torch.float64)
    set_seed(seed)
    m = build(max_iter=K)
    m.eval()
    tok = torch.randint(0, 2, (2, 6))
    tgt = torch.cumsum(tok, 1) % 2

    mb = copy.deepcopy(m); mb.backward_mode = "bptt"; mb.zero_grad()
    lb, st = loss_of(mb, tok, tgt, K); lb.backward()
    gb = flat_grads(mb)
    print(f"[V1 forward] iters={st.iterations} final mean res={st.rel_residual_mean[-1]:.3e}")

    for ns in [1, 5, 15, 30]:
        mn = copy.deepcopy(m); mn.backward_mode = "neumann"; mn.neumann_steps = ns
        mn.zero_grad()
        ln, _ = loss_of(mn, tok, tgt, K); ln.backward()
        gn = flat_grads(mn)
        rel = ((gn - gb).norm() / gb.norm()).item()
        cos = torch.nn.functional.cosine_similarity(gn, gb, dim=0).item()
        print(f"[V1 neumann k={ns:>2}] rel_err={rel:.3e} cos={cos:.6f}")

    params = list(m.parameters())
    cum = torch.cumsum(torch.tensor([p.numel() for p in params]), 0)
    idxs = torch.randperm(int(cum[-1]))[:n_fd]
    max_err = 0.0
    for fi in idxs.tolist():
        pi = int(torch.searchsorted(cum, fi, right=True))
        loc = fi - (int(cum[pi-1]) if pi > 0 else 0)
        ls = []
        for sg in (+1, -1):
            mm = copy.deepcopy(m); mm.backward_mode = "bptt"
            with torch.no_grad():
                list(mm.parameters())[pi].view(-1)[loc] += sg * eps
            l, _ = loss_of(mm, tok, tgt, K)
            ls.append(l.item())
        fd = (ls[0] - ls[1]) / (2 * eps)
        an = gb[fi].item()
        # hybrid criterion: relative where the gradient is resolvable,
        # absolute where it sits at/below the FD roundoff floor (~1e-10
        # for O(1) double-precision losses at eps=1e-6)
        rel = abs(fd - an) / max(abs(an), abs(fd), 1e-12)
        err = 0.0 if abs(fd - an) < 1e-9 else rel
        max_err = max(max_err, err)
    print(f"[V1 FD vs BPTT] {n_fd} params: max err (hybrid) {max_err:.3e} -> "
          f"{'PASS' if max_err < 1e-5 else 'FAIL'}")
    torch.set_default_dtype(torch.float32)
    return max_err < 1e-5


def sigma_at_zstar(model, tok, iters=40):
    """Power-iteration estimate of ||J|| at the solved z* (lower bound)."""
    x = model.encoder(model.embed(tok))
    static = model.block.precompute(x)
    z = ball_project(x, model.block.R)
    with torch.no_grad():
        for _ in range(model.max_iter):
            z = model.block.f(z, x, static, None)
    z0 = z.detach().requires_grad_(True)
    fz = model.block.f(z0, x, static, None)
    u = torch.randn_like(z0); u = u / u.norm()
    for _ in range(iters):
        u = torch.autograd.grad(fz, z0, u, retain_graph=True)[0]
        nu = u.norm()
        u = u / nu.clamp(min=1e-30)
    return nu.item()


def v2_audit(n_inputs=20, seed=0, model=None, label="untrained"):
    set_seed(seed)
    m = model or build(d=32, heads=4, max_iter=40, tol=1e-4, seq=16)
    m.eval()
    rho = m.rho
    worst = 0.0
    for i in range(n_inputs):
        tok = torch.randint(0, 2, (2, 16) if model is None else
                            (2, m.embed.num_embeddings and 16))
        s = sigma_at_zstar(m, tok)
        worst = max(worst, s)
    ok = worst <= rho + 1e-5
    print(f"[V2 {label}] worst sigma(J at z*) over {n_inputs} inputs: "
          f"{worst:.4f}  (theorem bound rho={rho:.2f}) -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_v1", action="store_true")
    args = ap.parse_args()
    ok1 = True if args.skip_v1 else v1_gradcheck()
    ok2 = v2_audit()
    print("\nACE V1+V2:", "PASS" if (ok1 and ok2) else "FAIL")

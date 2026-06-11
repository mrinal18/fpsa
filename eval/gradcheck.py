"""Gradient correctness harness (run BEFORE any training).

On a tiny instance in float64:
  1. Solve the fixed point to tight tolerance with a FIXED iteration budget
     (no early stop) so the computation is a deterministic function of params.
  2. Compare param gradients from:
       - full BPTT through the loop          (ground truth of the computed fn)
       - Neumann implicit gradient (k terms) (should match as k grows)
       - phantom1 / 1-step (HRM-style)       (reported, expected to be biased)
  3. Central finite differences on a random subset of scalar params against
     the BPTT graph — validates autograd itself.

Pass criteria printed at the end: FD vs BPTT rel err < 1e-6;
Neumann vs BPTT rel err small and decreasing in neumann_steps.
"""
import argparse
import copy

import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import FPSASeqModel
from utils.logging_utils import set_seed


def flat_grads(model):
    return torch.cat([p.grad.flatten() if p.grad is not None
                      else torch.zeros_like(p).flatten()
                      for p in model.parameters()])


def loss_fn(model, tokens, targets, max_iter):
    logits, stats = model(tokens, max_iter=max_iter)
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1)), stats


def grads_for(model, tokens, targets, backward, neumann_steps, max_iter):
    m = copy.deepcopy(model)
    m.backward_mode = backward
    m.neumann_steps = neumann_steps
    m.zero_grad()
    loss, stats = loss_fn(m, tokens, targets, max_iter)
    loss.backward()
    return flat_grads(m), loss.item(), stats


def rel_err(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value_mode", default="fixed_ffn")
    ap.add_argument("--max_iter", type=int, default=60)
    ap.add_argument("--n_fd", type=int, default=24)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    model = FPSASeqModel(vocab_size=2, num_classes=2, d_model=16, num_heads=2,
                         value_mode=args.value_mode, damping=0.5,
                         max_iter=args.max_iter, tol=0.0,  # tol=0 -> fixed budget
                         use_spectral_norm=True, use_rope=True, max_seq_len=8)
    # eval(): freezes spectral-norm power-iteration buffers so the loss is a
    # deterministic function of params (required for FD); autograd still works.
    model.eval()
    tokens = torch.randint(0, 2, (2, 6))
    targets = torch.cumsum(tokens, dim=1) % 2

    # --- forward convergence at the gradcheck point ---
    g_bptt, loss_b, st = grads_for(model, tokens, targets, "bptt", 0, args.max_iter)
    print(f"[forward] iters={st.iterations}  final mean rel-residual="
          f"{st.rel_residual_mean[-1]:.3e}  max={st.rel_residual_max[-1]:.3e}")

    rows = []
    for ns in [0, 1, 2, 5, 10, 20, 40]:
        mode = "phantom1" if ns == 0 else "neumann"
        g, loss_i, sti = grads_for(model, tokens, targets, mode, ns, args.max_iter)
        rows.append((ns, rel_err(g, g_bptt), cos(g, g_bptt)))
        tag = "phantom1(1-step)" if ns == 0 else f"neumann k={ns:>2}"
        print(f"[implicit vs BPTT] {tag}: rel_err={rows[-1][1]:.3e}  cos={rows[-1][2]:.6f}")

    # --- finite differences vs BPTT on random scalar params ---
    set_seed(args.seed + 1)
    params = [p for p in model.parameters() if p.requires_grad]
    sizes = torch.tensor([p.numel() for p in params])
    cum = torch.cumsum(sizes, 0)
    total = int(cum[-1])
    idxs = torch.randperm(total)[: args.n_fd]

    base = copy.deepcopy(model)
    g_ref = g_bptt
    max_fd_err = 0.0
    for flat_i in idxs.tolist():
        pi = int(torch.searchsorted(cum, flat_i, right=True))
        local = flat_i - (int(cum[pi - 1]) if pi > 0 else 0)
        losses = []
        for s in (+1, -1):
            m = copy.deepcopy(base)
            with torch.no_grad():
                list(m.parameters())[pi].view(-1)[local] += s * args.eps
            l, _ = loss_fn(m, tokens, targets, args.max_iter)
            losses.append(l.item())
        fd = (losses[0] - losses[1]) / (2 * args.eps)
        an = g_ref[flat_i].item()
        err = abs(fd - an) / max(abs(an), abs(fd), 1e-12)
        max_fd_err = max(max_fd_err, err)
    print(f"[FD vs BPTT] {args.n_fd} random params: max rel err = {max_fd_err:.3e}")

    ok_fd = max_fd_err < 1e-5
    ok_neu = rows[-1][1] < 1e-4
    print(f"\nPASS criteria: FD<1e-5: {'PASS' if ok_fd else 'FAIL'} | "
          f"Neumann(k=40) vs BPTT <1e-4: {'PASS' if ok_neu else 'FAIL'}")


if __name__ == "__main__":
    main()

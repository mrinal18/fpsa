"""V4 (token freezing) + certificate calibration for ACE.

Calibration: certified-v0 model uses the A-PRIORI bound
  ||z_k - z*|| <= ||z_{k+1} - z_k|| / (1 - rho),  rho = 1 - beta_min.
Relaxed model has no a-priori rho; we report the same bound with an
A-POSTERIORI per-solve rate lam_hat = max_{k>=5} ||d_{k+1}||/||d_k||
(labeled heuristic) and check empirical validity.

V4 freezing (relaxed model): freeze token i once its residual < tol_f;
frozen tokens stop updating but remain attention context. Report
prediction agreement vs full solve, accuracy, and compute saved.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from data.parity import make_parity
from models.ace import ACESeqModel, ball_project
from utils.logging_utils import set_seed


def load(run, certified):
    cfg = json.load(open(f"runs/{run}/config.json"))
    m = ACESeqModel(vocab_size=2, num_classes=2, d_model=cfg["d_model"],
                    num_heads=cfg["num_heads"], beta_min=cfg["beta_min"],
                    beta_max=cfg["beta_max"], R=cfg["R"], g_max=cfg["g_max"],
                    ffn_mult=cfg["ffn_mult"], certified=certified,
                    max_iter=cfg["max_iter"], tol=cfg["tol"],
                    neumann_steps=cfg["neumann_steps"], max_seq_len=cfg["seq_len"])
    m.load_state_dict(torch.load(f"runs/{run}/model.pt"))
    m.eval()
    return m, cfg


def trajectory(m, tok, K=40, Kref=200):
    x = m.encoder(m.embed(tok))
    s = m.block.precompute(x)
    z = ball_project(x, m.block.R)
    traj = [z]
    with torch.no_grad():
        for _ in range(Kref):
            z = m.block.f(z, x, s, None)
            traj.append(z)
    return traj


def calibrate(m, tok, rho=None, label=""):
    traj = trajectory(m, tok)
    zref = traj[-1]
    viol = 0; ratios = []
    lam_hat = max(((traj[k+1]-traj[k]).norm() / (traj[k]-traj[k-1]).norm()).item()
                  for k in range(5, 40))
    use_rho = rho if rho is not None else lam_hat
    for k in range(1, 39):
        true = (traj[k] - zref).norm().item()
        bound = (traj[k+1] - traj[k]).norm().item() / max(1 - use_rho, 1e-6)
        if true > bound * (1 + 1e-6):
            viol += 1
        if true > 1e-9:
            ratios.append(bound / true)
    import statistics
    print(f"[calib {label}] rho_used={use_rho:.3f} ({'a-priori' if rho else 'a-posteriori lam_hat'}) "
          f"violations {viol}/38  median tightness x{statistics.median(ratios):.1f}")
    return viol


def freeze_solve(m, tok, tol_f, K=40):
    x = m.encoder(m.embed(tok))
    s = m.block.precompute(x)
    z = ball_project(x, m.block.R)
    B, N, _ = z.shape
    frozen = torch.zeros(B, N, dtype=torch.bool)
    freeze_iter = torch.full((B, N), K, dtype=torch.long)
    with torch.no_grad():
        for k in range(K):
            zn = m.block.f(z, x, s, None)
            r = (zn - z).norm(dim=-1) / z.norm(dim=-1).clamp(min=1e-8)
            newly = (~frozen) & (r < tol_f)
            freeze_iter[newly] = k + 1
            z = torch.where(frozen.unsqueeze(-1), z, zn)
            frozen |= newly
            if frozen.all():
                break
    logits = m.head(m.out_norm(z))
    return logits, freeze_iter.float().mean().item() / K, z


if __name__ == "__main__":
    set_seed(0)
    _, (xte, yte) = make_parity(20000, 4000, 16, seed=1234)
    tok = xte[:256]; tgt = yte[:256]

    mc, _ = load("ace_v3_parity_seed0", certified=True)
    calibrate(mc, tok[:32], rho=mc.rho, label="certified-v0 (a-priori rho=0.8)")

    mr, _ = load("ace_v3_relaxed_seed0", certified=False)
    calibrate(mr, tok[:32], rho=None, label="relaxed")

    # V4 on the relaxed (working) model
    with torch.no_grad():
        lg_full, st = mr(tok)
    p_full = lg_full.argmax(-1)
    acc_full = (p_full == tgt).float().mean().item()
    print(f"\n[V4 baseline] full 40-iter solve: tok acc {acc_full:.4f}")
    for tf in [3e-3, 1e-3, 3e-4]:
        lg_f, frac, _ = freeze_solve(mr, tok, tf)
        p_f = lg_f.argmax(-1)
        agree = (p_f == p_full).float().mean().item()
        acc = (p_f == tgt).float().mean().item()
        print(f"[V4 tol_f={tf:.0e}] agreement {agree:.4f}  tok acc {acc:.4f}  "
              f"mean token-iters used {frac*100:.1f}% of full budget")

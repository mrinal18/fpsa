"""Unified Sudoku/Maze benchmark harness for all TRMSubstrate arms.
Metrics match the ablation dashboards: cell/board (sudoku), path P/R/F1 +
copy baseline (maze), mean_steps, converged_frac, final_residual."""
import argparse, csv, gc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F, yaml
from models.trm_substrate import TRMSubstrate
from data.maze import (load_trm_maze, synthetic_maze, maze_metrics,
                       sudoku_metrics, VOCAB_SIZE as MAZE_VOCAB)
from data.sudoku import load_trm_dataset, synthetic_sudoku, encode, VOCAB_SIZE as SUD_VOCAB
from utils.logging_utils import CSVLogger, set_seed


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()
    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def get_data(task, synthetic, data_root, hw):
    if task == "sudoku":
        vocab, grid = SUD_VOCAB, (9, 9)
        if synthetic:
            p, s = synthetic_sudoku(synthetic, seed=42)
            xtr = xte = encode(p); ytr = yte = encode(s)
        else:
            xtr, ytr = load_trm_dataset(data_root, "train")
            xte, yte = load_trm_dataset(data_root, "test")
    else:
        vocab, grid = MAZE_VOCAB, (hw, hw)
        if synthetic:
            xtr, ytr = synthetic_maze(synthetic, hw=hw, seed=42)
            xte, yte = synthetic_maze(max(64, synthetic // 4), hw=hw, seed=43)
        else:
            xtr, ytr = load_trm_maze(data_root, "train")
            xte, yte = load_trm_maze(data_root, "test")
            grid = (int(xtr.shape[1] ** 0.5),) * 2
    return xtr, ytr, xte, yte, vocab, grid


@torch.no_grad()
def k_sweep(model, task, x, y, tol, max_n=512):
    # Test-time iteration extrapolation: cell acc at n/2, n, 2n inner iters.
    # Equilibrium-trained models should be ~monotone (anytime property);
    # unrolled-regime models peak at trained depth.
    n0 = model.n_inner
    out = {}
    for label, n in [("k_half", max(1, n0 // 2)), ("k_train", n0), ("k_2x", 2 * n0)]:
        model.n_inner = n
        lg, _ = model(x[:max_n])
        out[label] = (lg[-1].argmax(-1) == y[:max_n]).float().mean().item()
    model.n_inner = n0
    return out


@torch.no_grad()
def evaluate(model, task, x, y, tol, batch=256, max_n=1024):
    model.eval()
    x, y = x[:max_n], y[:max_n]
    preds, conv, steps, res = [], [], [], []
    for i in range(0, len(x), batch):
        lg, rec = model(x[i:i+batch])
        preds.append(lg[-1].argmax(-1))
        conv.append((rec["res_sample"] < tol).float())
        steps.append(rec["total_iters"])
        res.append(rec["res_sample"].mean().item())
    pred = torch.cat(preds)
    fn = sudoku_metrics if task == "sudoku" else maze_metrics
    m = fn(pred, y, x)
    m["converged_frac"] = torch.cat(conv).mean().item()
    m["mean_steps"] = sum(steps) / len(steps)
    m["final_residual"] = sum(res) / len(res)
    # k-extrapolation: equilibrium models must NOT degrade past trained depth
    n0 = model.n_inner
    for mult, key in [(0.5, "acc_k0.5x"), (2, "acc_k2x"), (4, "acc_k4x")]:
        model.n_inner = max(1, int(n0 * mult))
        lg, _ = model(x[:256])
        m[key] = (lg[-1].argmax(-1) == y[:256]).float().mean().item()
    model.n_inner = n0
    model.train()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sudoku", "maze"], required=True)
    ap.add_argument("--arm", default="ace_relaxed",
                    choices=["fixed", "fixed_ffn", "evolving", "blended", "ace_relaxed"])
    ap.add_argument("--backward", default="neumann_k",
                    choices=["neumann_k", "bptt_k", "phantom1"])
    ap.add_argument("--config", default="configs/bench_default.yaml")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--data_root", default="data/trm_out")
    ap.add_argument("--hw", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max_minutes", type=float, default=2.8)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.steps: cfg["steps"] = a.steps
    torch.set_flush_denormal(True); set_seed(a.seed)
    name = f"{a.task}_{a.arm}_{a.backward}_s{a.seed}"
    rd = f"runs/{name}"; os.makedirs(rd, exist_ok=True)
    json.dump({**cfg, **vars(a)}, open(f"{rd}/config.json", "w"), indent=2)
    ck, dn = f"{rd}/checkpoint.pt", f"{rd}/final.json"
    if os.path.exists(dn):
        print("complete:", open(dn).read()); return
    xtr, ytr, xte, yte, vocab, grid = get_data(a.task, a.synthetic, a.data_root, a.hw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TRMSubstrate(vocab, vocab, d_model=cfg["d_model"], num_heads=cfg["num_heads"],
                     value_mode=a.arm, backward=a.backward, bwd_k=cfg["bwd_k"],
                     n_inner=cfg["n_inner"], T_outer=cfg["T_outer"], tol=cfg["tol"],
                     pos_mode="2d", grid_hw=grid, ffn_mult=cfg.get("ffn_mult", 2.0),
                     freeze_eps=cfg.get("freeze_eps", 0.0),
                     jac_reg=cfg.get("jac_reg_weight", 0.0) > 0,
                     jac_power_iters=cfg.get("jac_power_iters", 4),
                     inject_x=cfg.get("inject_x", True)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    ema = EMA(m, cfg.get("ema_decay", 0.999))
    g = torch.Generator().manual_seed(a.seed); start = 0
    fields = ["step", "loss"] + (["cell_accuracy", "board_accuracy"] if a.task == "sudoku"
              else ["cell_accuracy", "solved_accuracy", "path_precision",
                    "path_recall", "path_f1", "copy_cell_baseline"]) + \
             ["converged_frac", "mean_steps", "final_residual",
              "acc_k0.5x", "acc_k2x", "acc_k4x", "sps"]
    if os.path.exists(ck):
        c = torch.load(ck, weights_only=False, map_location=device)
        m.load_state_dict(c["m"]); opt.load_state_dict(c["o"]); ema.shadow = c["e"]
        g.set_state(c["g"]); torch.set_rng_state(c["r"]); start = c["s"]
        fh = open(f"{rd}/train_log.csv", "a", newline=""); w = csv.DictWriter(fh, fieldnames=fields)
        class L:
            def log(s, **k): w.writerow({x: k.get(x, "") for x in fields}); fh.flush()
            def close(s): fh.close()
        log = L(); print(f"resumed {start}")
    else:
        log = CSVLogger(f"{rd}/train_log.csv", fields)
        print(f"{name}: params {sum(p.numel() for p in m.parameters()):,} "
              f"grid {grid} vocab {vocab} train {len(xtr)}")
    def sv(s): torch.save({"m": m.state_dict(), "o": opt.state_dict(), "e": ema.shadow,
                           "g": g.get_state(), "r": torch.get_rng_state(), "s": s}, ck)
    # optional class weighting (maze: upweight PATH cells to escape the
    # copy local optimum; sanity runs plateau at copy_cell_baseline without it)
    cw = None
    if a.task == "maze" and cfg.get("path_weight", 1.0) != 1.0:
        cw = torch.ones(vocab, device=device); cw[5] = cfg["path_weight"]
    m.train(); t0 = time.time(); step = start
    fallback_steps, fb_next = 0, False
    auto_fb = cfg.get("auto_fallback", True) and a.backward == "neumann_k"
    for step in range(start + 1, cfg["steps"] + 1):
        idx = torch.randint(0, len(xtr), (cfg["batch"],), generator=g)
        xb, yb = xtr[idx].to(device), ytr[idx].to(device)
        if auto_fb:
            m.backward = "bptt_k" if fb_next else "neumann_k"
            fallback_steps += int(fb_next)
        logits, rec = m(xb)
        ce = sum(F.cross_entropy(l.view(-1, vocab), yb.view(-1), ignore_index=0,
                                 weight=cw) for l in logits) / len(logits)
        loss, sig = ce, float("nan")
        if rec.get("sigma_hats"):
            sig_t = torch.stack(rec["sigma_hats"]).mean()
            sig = sig_t.item()
            pen = cfg.get("jac_reg_weight", 0.0) *                 F.relu(sig_t - cfg.get("jac_rho_target", 0.85)) ** 2
            # guard v2: on diverged solves, train stability terms instead of CE
            if cfg.get("divergence_guard", False) and                     rec["res_sample"].mean().item() > cfg.get("guard_thresh", 0.15):
                loss = pen + cfg.get("guard_res_weight", 1.0) *                     torch.stack(rec["exit_res"]).mean()
            else:
                loss = ce + pen
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); ema.update(m)
        if auto_fb:
            tails = rec.get("adjoint_tail") or [0.0]
            fb_next = (max(tails) > 1.0) or ((sig == sig) and sig > 1.0)
        rec["sigma_hats"] = rec["exit_res"] = None   # break graph refs (leak hygiene)
        del logits
        if step % 50 == 0: gc.collect()
        if step % cfg["eval_every"] == 0 or step == 1:
            import copy as _c
            em = _c.deepcopy(m); ema.copy_to(em)
            mt = evaluate(em, a.task, xte.to(device), yte.to(device), cfg["tol"])
            del em
            sps = (time.time() - t0) / max(step - start, 1)
            log.log(step=step, loss=f"{loss.item():.4f}", sps=f"{sps:.2f}",
                    fallback_steps=fallback_steps,
                    **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in mt.items()})
            key = "board_accuracy" if a.task == "sudoku" else "path_f1"
            print(f"step {step:5d} loss {loss.item():.4f} cell {mt['cell_accuracy']:.4f} "
                  f"{key} {mt[key]:.4f} conv {mt['converged_frac']:.2f} "
                  f"steps {mt['mean_steps']:.0f} res {mt['final_residual']:.2e} ({sps:.2f}s)")
            sv(step)
        if (time.time() - t0) / 60 > a.max_minutes:
            sv(step); print(f"TIME LIMIT {step}"); log.close(); return
    import copy as _c
    em = _c.deepcopy(m); ema.copy_to(em)
    mt = evaluate(em, a.task, xte.to(device), yte.to(device), cfg["tol"], max_n=len(xte))
    json.dump({"final": mt, "steps": step}, open(dn, "w"), indent=2)
    sv(step); print("FINAL", {k: round(v, 4) for k, v in mt.items()}); log.close()


if __name__ == "__main__":
    main()

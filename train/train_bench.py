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
                     freeze_eps=cfg.get("freeze_eps", 0.0)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    ema = EMA(m, cfg.get("ema_decay", 0.999))
    g = torch.Generator().manual_seed(a.seed); start = 0
    fields = ["step", "loss"] + (["cell_accuracy", "board_accuracy"] if a.task == "sudoku"
              else ["cell_accuracy", "solved_accuracy", "path_precision",
                    "path_recall", "path_f1", "copy_cell_baseline"]) + \
             ["converged_frac", "mean_steps", "final_residual", "sps"]
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
    for step in range(start + 1, cfg["steps"] + 1):
        idx = torch.randint(0, len(xtr), (cfg["batch"],), generator=g)
        xb, yb = xtr[idx].to(device), ytr[idx].to(device)
        logits, rec = m(xb)
        loss = sum(F.cross_entropy(l.view(-1, vocab), yb.view(-1), ignore_index=0,
                                   weight=cw) for l in logits) / len(logits)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); ema.update(m)
        del logits
        if step % 50 == 0: gc.collect()
        if step % cfg["eval_every"] == 0 or step == 1:
            import copy as _c
            em = _c.deepcopy(m); ema.copy_to(em)
            mt = evaluate(em, a.task, xte.to(device), yte.to(device), cfg["tol"])
            del em
            sps = (time.time() - t0) / max(step - start, 1)
            log.log(step=step, loss=f"{loss.item():.4f}", sps=f"{sps:.2f}",
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

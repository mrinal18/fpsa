"""ACE V3: prefix parity with ZERO stability machinery.
No Jacobian regularization, no divergence guard, no damping knob — the
theorems are the stability story. Logs a power-iteration sigma audit at
every eval (V2-during-training)."""
import argparse, csv, gc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import yaml

from data.parity import make_parity
from models.ace import ACESeqModel
from eval.ace_validate import sigma_at_zstar
from utils.logging_utils import CSVLogger, set_seed


def evaluate(model, x, y, batch=512):
    model.eval()
    ct = cs = 0
    curve = None
    with torch.no_grad():
        for i in range(0, len(x), batch):
            lg, st = model(x[i:i+batch])
            p = lg.argmax(-1)
            ct += (p == y[i:i+batch]).sum().item()
            cs += (p == y[i:i+batch]).all(1).sum().item()
            if curve is None:
                curve = st.rel_residual_mean
    model.train()
    return ct / y.numel(), cs / len(x), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ace_v3_parity.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_minutes", type=float, default=2.5)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed

    torch.set_flush_denormal(True)
    set_seed(cfg["seed"])
    run_dir = os.path.join("runs", f"{cfg['name']}_seed{cfg['seed']}")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    done = os.path.join(run_dir, "final.json")
    if os.path.exists(done):
        print("complete:", open(done).read())
        return

    (xtr, ytr), (xte, yte) = make_parity(cfg["n_train"], cfg["n_test"],
                                         cfg["seq_len"], seed=cfg["data_seed"])
    xte_s, yte_s = xte[:1024], yte[:1024]
    model = ACESeqModel(vocab_size=2, num_classes=2, d_model=cfg["d_model"],
                        num_heads=cfg["num_heads"], beta_min=cfg["beta_min"],
                        beta_max=cfg["beta_max"], R=cfg["R"], g_max=cfg["g_max"],
                        ffn_mult=cfg["ffn_mult"],
                        certified=cfg.get("certified", True),
                        max_iter=cfg["max_iter"],
                        tol=cfg["tol"], neumann_steps=cfg["neumann_steps"],
                        max_seq_len=cfg["seq_len"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    g = torch.Generator().manual_seed(cfg["seed"])
    start = 0
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        g.set_state(ck["gen"]); torch.set_rng_state(ck["torch_rng"])
        start = ck["step"]
        print(f"resumed from {start}")
    else:
        print(f"params: {sum(p.numel() for p in model.parameters()):,} | "
              f"certified rho = {model.rho}")

    fields = ["step", "loss", "test_tok_acc", "test_seq_acc", "fpi_iters",
              "res_last", "converged_frac", "sigma_audit", "sec_per_step"]
    lp = os.path.join(run_dir, "train_log.csv")
    if start > 0 and os.path.exists(lp):
        fh = open(lp, "a", newline=""); w = csv.DictWriter(fh, fieldnames=fields)
        class _L:
            def log(self, **kw): w.writerow({k: kw.get(k, "") for k in fields}); fh.flush()
            def close(self): fh.close()
        log = _L()
    else:
        log = CSVLogger(lp, fields)

    def save(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "gen": g.get_state(), "torch_rng": torch.get_rng_state(),
                    "step": step}, ckpt)

    model.train()
    t0 = time.time()
    step = start
    for step in range(start + 1, cfg["steps"] + 1):
        idx = torch.randint(0, len(xtr), (cfg["batch_size"],), generator=g)
        lg, st = model(xtr[idx])
        loss = F.cross_entropy(lg.view(-1, 2), ytr[idx].view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        del lg
        if step % 50 == 0:
            gc.collect()
        if step % cfg["eval_every"] == 0 or step == 1:
            ta, sa, _ = evaluate(model, xte_s, yte_s)
            model.eval()
            sig = sigma_at_zstar(model, xte_s[:64], iters=15)
            model.train()
            sps = (time.time() - t0) / max(step - start, 1)
            log.log(step=step, loss=f"{loss.item():.4f}", test_tok_acc=f"{ta:.4f}",
                    test_seq_acc=f"{sa:.4f}", fpi_iters=st.iterations,
                    res_last=f"{st.rel_residual_mean[-1]:.3e}",
                    converged_frac=f"{st.converged_frac:.3f}",
                    sigma_audit=f"{sig:.4f}", sec_per_step=f"{sps:.2f}")
            print(f"step {step:5d} loss {loss.item():.4f} tok {ta:.4f} seq {sa:.4f} "
                  f"iters {st.iterations} res {st.rel_residual_mean[-1]:.2e} "
                  f"sigma {sig:.4f} ({sps:.2f}s/step)")
            save(step)
            if ta >= cfg.get("early_stop_acc", 1.01):
                break
        if (time.time() - t0) / 60.0 > args.max_minutes:
            save(step)
            print(f"TIME LIMIT at {step}")
            log.close()
            return
    ta, sa, _ = evaluate(model, xte, yte)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    save(step)
    json.dump({"final_test_tok_acc": ta, "final_test_seq_acc": sa,
               "steps_run": step}, open(done, "w"), indent=2)
    print(f"FINAL tok {ta:.4f} seq {sa:.4f}")
    log.close()


if __name__ == "__main__":
    main()

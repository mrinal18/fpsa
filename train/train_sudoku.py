"""GATE 2: Sudoku-Extreme training (HRM/TRM protocol data).

Consumes the HRM/TRM builder output (zero protocol drift) or synthetic
data (--synthetic N, CPU smoke tests only). Reports cell accuracy and
exact-match (all 81 cells) accuracy; full instrumentation as always.
Resumable; EMA evaluation copy (TRM uses ema=True).
"""
import argparse
import copy
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml

from data.sudoku import (load_trm_dataset, synthetic_sudoku, encode,
                         VOCAB_SIZE, IGNORE_LABEL_ID)
from models.model import FPSASeqModel
from models.ace import ACESeqModel
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


def evaluate(model, x, y, batch=256, max_iter=None, max_examples=None):
    model.eval()
    if max_examples:
        x, y = x[:max_examples], y[:max_examples]
    cells = exact = 0
    res_curve = None
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i+batch].to(next(model.parameters()).device)
            yb = y[i:i+batch].to(xb.device)
            logits, st = model(xb, max_iter=max_iter)
            pred = logits.argmax(-1)
            cells += (pred == yb).sum().item()
            exact += (pred == yb).all(dim=1).sum().item()
            if res_curve is None:
                res_curve = st.rel_residual_mean
    model.train()
    return cells / y.numel(), exact / len(x), res_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gate2_sudoku.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--synthetic", type=int, default=0,
                    help="use N synthetic puzzles (CPU smoke only)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for k, v in [("seed", args.seed), ("steps", args.steps),
                 ("data_dir", args.data_dir), ("name", args.name)]:
        if v is not None:
            cfg[k] = v

    torch.set_flush_denormal(True)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join("runs", f"{cfg['name']}_seed{cfg['seed']}")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    done_path = os.path.join(run_dir, "final.json")
    if os.path.exists(done_path):
        print("run already complete:", open(done_path).read())
        return

    if args.synthetic:
        puz, sol = synthetic_sudoku(args.synthetic, n_givens=cfg.get("syn_givens", 30),
                                    seed=cfg["seed"])
        xtr = xte = encode(puz)
        ytr = yte = encode(sol)
        print(f"SYNTHETIC smoke mode: {args.synthetic} puzzles (train==test, overfit check)")
    else:
        xtr, ytr = load_trm_dataset(cfg["data_dir"], "train")
        xte, yte = load_trm_dataset(cfg["data_dir"], "test")
        print(f"train {len(xtr)} examples, test {len(xte)}")

    arch = cfg.get("arch", "fpsa")
    if arch == "fpsa":
        model = FPSASeqModel(
            vocab_size=VOCAB_SIZE, num_classes=VOCAB_SIZE,
            d_model=cfg["d_model"], num_heads=cfg["num_heads"],
            value_mode=cfg["value_mode"], damping=cfg["damping"],
            max_iter=cfg["max_iter"], tol=cfg["tol"], backward=cfg["backward"],
            neumann_steps=cfg["neumann_steps"],
            use_spectral_norm=cfg["spectral_norm"], use_rope=True,
            pos_mode="2d", grid_hw=(9, 9), ffn_mult=cfg["ffn_mult"],
            jac_reg=cfg.get("jac_reg_weight", 0.0) > 0,
            freeze_sn=cfg.get("freeze_sn", True),
            jac_power_iters=cfg.get("jac_power_iters", 4)).to(device)
    elif arch in ("ace_relaxed", "ace_certified"):
        # ACE: NO guard, NO jac penalty (stability is architectural /
        # monitored); sigma audited at eval below.
        model = ACESeqModel(
            vocab_size=VOCAB_SIZE, num_classes=VOCAB_SIZE,
            d_model=cfg["d_model"], num_heads=cfg["num_heads"],
            beta_min=cfg.get("beta_min", 0.2), beta_max=cfg.get("beta_max", 0.8),
            R=cfg.get("R", 3.0), g_max=cfg.get("g_max", 0.97),
            ffn_mult=cfg["ffn_mult"], certified=(arch == "ace_certified"),
            max_iter=cfg["max_iter"], tol=cfg["tol"],
            neumann_steps=cfg["neumann_steps"],
            pos_mode="2d", grid_hw=(9, 9)).to(device)
    else:
        raise ValueError(arch)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    ema = EMA(model, cfg.get("ema_decay", 0.999)) if cfg.get("ema", True) else None
    g = torch.Generator().manual_seed(cfg["seed"])

    start_step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        g.set_state(ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        if ema and "ema" in ck:
            ema.shadow = ck["ema"]
        start_step = ck["step"]
        print(f"resumed from step {start_step}")
    else:
        print(f"params: {n_params:,}  device: {device}")

    fields = ["step", "loss", "cell_acc", "exact_acc", "ema_cell_acc",
              "ema_exact_acc", "fpi_iters", "res_first", "res_last",
              "converged_frac", "jac_sigma", "guard", "sec_per_step"]
    mode = "a" if (start_step > 0 and os.path.exists(os.path.join(run_dir, "train_log.csv"))) else "w"
    if mode == "a":
        import csv as _csv
        fh = open(os.path.join(run_dir, "train_log.csv"), "a", newline="")
        w = _csv.DictWriter(fh, fieldnames=fields)
        class _L:
            def log(self, **kw): w.writerow({k: kw.get(k, "") for k in fields}); fh.flush()
            def close(self): fh.close()
        log = _L()
    else:
        log = CSVLogger(os.path.join(run_dir, "train_log.csv"), fields)

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "gen": g.get_state(), "torch_rng": torch.get_rng_state(),
                    "step": step, **({"ema": ema.shadow} if ema else {})}, ckpt_path)

    model.train()
    t0 = time.time()
    n_guard = 0
    step = start_step
    stop_reason = "steps_complete"
    eval_n = cfg.get("eval_subset", 2048)
    for step in range(start_step + 1, cfg["steps"] + 1):
        idx = torch.randint(0, len(xtr), (cfg["batch_size"],), generator=g)
        xb, yb = xtr[idx].to(device), ytr[idx].to(device)
        logits, st = model(xb)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1),
                               ignore_index=IGNORE_LABEL_ID)
        pen = None
        sigma = float("nan")
        if arch != "fpsa":
            pass  # no penalty/guard for ACE
        elif st.jac_sigma_est is not None:
            sigma = st.jac_sigma_est.item()
            pen = F.relu(st.jac_sigma_est - cfg.get("jac_rho_target", 0.85)) ** 2
        diverged = (arch == "fpsa" and cfg.get("divergence_guard", True)
                    and st.rel_residual_mean[-1] > cfg.get("guard_thresh", 0.15))
        if diverged:
            n_guard += 1
            loss = 0.0
            if pen is not None:
                loss = loss + cfg["jac_reg_weight"] * pen
            if st.exit_residual_t is not None:
                loss = loss + cfg.get("guard_res_weight", 1.0) * st.exit_residual_t
            if not torch.is_tensor(loss):
                st.jac_sigma_est = None
                st.exit_residual_t = None
                del logits
                continue
        elif pen is not None:
            loss = loss + cfg["jac_reg_weight"] * pen
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema:
            ema.update(model)
        st.jac_sigma_est = None
        st.exit_residual_t = None
        if step % 50 == 0:
            gc.collect()

        if step % cfg["eval_every"] == 0 or step == 1:
            ca, ea, _ = evaluate(model, xte, yte, max_examples=eval_n)
            if arch != "fpsa":
                from eval.ace_validate import sigma_at_zstar
                model.eval()
                sigma = sigma_at_zstar(model, xte[:32].to(device), iters=12)
                model.train()
            eca = eea = float("nan")
            if ema:
                em = copy.deepcopy(model)
                ema.copy_to(em)
                eca, eea, _ = evaluate(em, xte, yte, max_examples=eval_n)
                del em
            sps = (time.time() - t0) / max(step - start_step, 1)
            log.log(step=step, loss=f"{float(loss):.4f}", cell_acc=f"{ca:.4f}",
                    exact_acc=f"{ea:.4f}", ema_cell_acc=f"{eca:.4f}",
                    ema_exact_acc=f"{eea:.4f}", fpi_iters=st.iterations,
                    res_first=f"{st.rel_residual_mean[0]:.3e}",
                    res_last=f"{st.rel_residual_mean[-1]:.3e}",
                    converged_frac=f"{st.converged_frac:.3f}",
                    jac_sigma=f"{sigma:.3f}", guard=n_guard,
                    sec_per_step=f"{sps:.2f}")
            print(f"step {step:6d} loss {float(loss):.4f} cell {ca:.4f} exact {ea:.4f} "
                  f"ema_exact {eea:.4f} iters {st.iterations} "
                  f"res {st.rel_residual_mean[-1]:.2e} sigma {sigma:.3f} "
                  f"guard {n_guard} ({sps:.2f}s/step)")
            save_ckpt(step)
        if (time.time() - t0) / 60.0 > args.max_minutes:
            save_ckpt(step)
            print(f"TIME LIMIT: checkpointed at step {step}; rerun to resume")
            log.close()
            return

    # final full-test eval (EMA copy if enabled, matching TRM reporting)
    final_model = model
    if ema:
        final_model = copy.deepcopy(model)
        ema.copy_to(final_model)
    ca, ea, _ = evaluate(final_model, xte, yte)
    torch.save(final_model.state_dict(), os.path.join(run_dir, "model.pt"))
    save_ckpt(step)
    json.dump({"final_cell_acc": ca, "final_exact_acc": ea, "params": n_params,
               "steps_run": step, "stop_reason": stop_reason,
               "n_test": len(xte), "guard_steps": n_guard},
              open(done_path, "w"), indent=2)
    print(f"FINAL (full test, {'EMA' if ema else 'raw'}): cell {ca:.4f} exact {ea:.4f}")
    log.close()


if __name__ == "__main__":
    main()

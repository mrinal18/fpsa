"""GATE 1: prefix-parity toy task with full convergence instrumentation.

Resumable: checkpoints (model/opt/step/data-generator state) every eval;
--max_minutes bounds wall time per invocation and exits cleanly with a
checkpoint, so long runs can be chunked without losing determinism of the
data order. A completed run writes final.json (the done marker).
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml

from data.parity import make_parity
from models.model import FPSASeqModel
from models.ace_block import ACESeqModel
from utils.logging_utils import CSVLogger, set_seed


def evaluate(model, x, y, batch=512, max_iter=None):
    model.eval()
    correct_tok = correct_seq = 0
    res_curve = None
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits, st = model(x[i:i + batch], max_iter=max_iter)
            pred = logits.argmax(-1)
            correct_tok += (pred == y[i:i + batch]).sum().item()
            correct_seq += (pred == y[i:i + batch]).all(dim=1).sum().item()
            if res_curve is None:
                res_curve = st.rel_residual_mean
    model.train()
    return correct_tok / y.numel(), correct_seq / len(x), res_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gate1_parity.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max_minutes", type=float, default=5.0)
    ap.add_argument("--value_mode", default=None)
    ap.add_argument("--freeze_sn", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.value_mode is not None:
        cfg["value_mode"] = args.value_mode
    if args.freeze_sn is not None:
        cfg["freeze_sn"] = bool(args.freeze_sn)
    if args.name is not None:
        cfg["name"] = args.name

    torch.set_flush_denormal(True)  # denormals cause 10-100x CPU stalls
    set_seed(cfg["seed"])
    run_dir = os.path.join("runs", f"{cfg['name']}_seed{cfg['seed']}")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    snap_path = os.path.join(run_dir, "residual_snapshots.json")
    done_path = os.path.join(run_dir, "final.json")
    if os.path.exists(done_path):
        print("run already complete:", open(done_path).read())
        return

    (xtr, ytr), (xte, yte) = make_parity(
        cfg["n_train"], cfg["n_test"], cfg["seq_len"], seed=cfg["data_seed"])
    xte_small, yte_small = xte[:1024], yte[:1024]  # cheap in-training eval

    if cfg.get("arch", "fpsa") == "ace":
        model = ACESeqModel(
            vocab_size=2, num_classes=2, d_model=cfg["d_model"],
            num_heads=cfg["num_heads"], tau=cfg.get("tau", 0.5),
            beta_min=cfg.get("beta_min", 0.2), beta_max=cfg.get("beta_max", 0.9),
            max_iter=cfg["max_iter"], tol=cfg["tol"], backward=cfg["backward"],
            neumann_steps=cfg["neumann_steps"], max_seq_len=cfg["seq_len"],
            ffn_mult=cfg["ffn_mult"])
    else:
        model = FPSASeqModel(
        vocab_size=2, num_classes=2, d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], value_mode=cfg["value_mode"],
        damping=cfg["damping"], max_iter=cfg["max_iter"], tol=cfg["tol"],
        backward=cfg["backward"], neumann_steps=cfg["neumann_steps"],
        use_spectral_norm=cfg["spectral_norm"], use_rope=True,
        max_seq_len=cfg["seq_len"], ffn_mult=cfg["ffn_mult"],
        jac_reg=cfg.get("jac_reg_weight", 0.0) > 0,
        freeze_sn=cfg.get("freeze_sn", True),
        jac_power_iters=cfg.get("jac_power_iters", 2))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    g = torch.Generator().manual_seed(cfg["seed"])

    start_step = 0
    res_snapshots = {}
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        g.set_state(ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        start_step = ck["step"]
        if os.path.exists(snap_path):
            res_snapshots = json.load(open(snap_path))
        print(f"resumed from step {start_step}")
    else:
        print(f"params: {n_params:,}")

    log_path = os.path.join(run_dir, "train_log.csv")
    fields = ["step", "loss", "train_tok_acc", "test_tok_acc", "test_seq_acc",
              "fpi_iters", "res_first", "res_last", "converged_frac",
              "bwd_res_last", "jac_sigma", "sec_per_step"]
    if start_step > 0 and os.path.exists(log_path):
        fh = open(log_path, "a", newline="")
        import csv
        w = csv.DictWriter(fh, fieldnames=fields)
        class _L:
            def log(self, **kw): w.writerow({k: kw.get(k, "") for k in fields}); fh.flush()
            def close(self): fh.close()
        log = _L()
    else:
        log = CSVLogger(log_path, fields)

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "gen": g.get_state(), "torch_rng": torch.get_rng_state(),
                    "step": step}, ckpt_path)
        json.dump(res_snapshots, open(snap_path, "w"))

    model.train()
    t0 = time.time()
    step = start_step
    n_guard = 0
    stop_reason = "steps_complete"
    for step in range(start_step + 1, cfg["steps"] + 1):
        idx = torch.randint(0, len(xtr), (cfg["batch_size"],), generator=g)
        xb, yb = xtr[idx], ytr[idx]
        logits, st = model(xb)
        loss = F.cross_entropy(logits.view(-1, 2), yb.view(-1))
        sigma = float('nan')
        pen = None
        if st.jac_sigma_est is not None:
            sigma = st.jac_sigma_est.item()
            pen = F.relu(st.jac_sigma_est - cfg.get("jac_rho_target", 0.85)) ** 2
        diverged = (cfg.get("divergence_guard", False)
                    and st.rel_residual_mean[-1] > cfg.get("guard_thresh", 0.15))
        if diverged:
            n_guard += 1
            loss = 0.0
            if pen is not None:
                loss = loss + cfg["jac_reg_weight"] * pen
            if cfg.get("guard_version", 1) >= 2 and st.exit_residual_t is not None:
                loss = loss + cfg.get("guard_res_weight", 1.0) * st.exit_residual_t
            if not torch.is_tensor(loss):
                continue
        elif pen is not None:
            loss = loss + cfg["jac_reg_weight"] * pen
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # break hook-closure <-> stats reference cycles (leak ~20MB/step)
        st.jac_sigma_est = None
        st.exit_residual_t = None
        if step % 50 == 0:
            gc.collect()

        if step % cfg["eval_every"] == 0 or step == 1:
            with torch.no_grad():
                tr_acc = (logits.argmax(-1) == yb).float().mean().item()
            te_tok, te_seq, curve = evaluate(model, xte_small, yte_small)
            res_snapshots[str(step)] = curve
            sps = (time.time() - t0) / max(step - start_step, 1)
            log.log(step=step, loss=f"{loss.item():.4f}",
                    train_tok_acc=f"{tr_acc:.4f}", test_tok_acc=f"{te_tok:.4f}",
                    test_seq_acc=f"{te_seq:.4f}", fpi_iters=st.iterations,
                    res_first=f"{st.rel_residual_mean[0]:.3e}",
                    res_last=f"{st.rel_residual_mean[-1]:.3e}",
                    converged_frac=f"{st.converged_frac:.3f}",
                    bwd_res_last=f"{st.backward_residual[-1]:.3e}" if st.backward_residual else "",
                    jac_sigma=f"{sigma:.3f}", sec_per_step=f"{sps:.2f}")
            print(f"step {step:5d} loss {loss.item():.4f} train {tr_acc:.3f} "
                  f"test_tok {te_tok:.4f} test_seq {te_seq:.4f} iters {st.iterations} "
                  f"res {st.rel_residual_mean[-1]:.2e} sigma {sigma:.3f} guard {n_guard} ({sps:.2f}s/step)")
            save_ckpt(step)
            if te_tok >= cfg.get("early_stop_acc", 1.01):
                stop_reason = "early_stop_acc"
                break
        if (time.time() - t0) / 60.0 > args.max_minutes:
            save_ckpt(step)
            print(f"TIME LIMIT: checkpointed at step {step}; rerun to resume")
            log.close()
            return

    # full final eval
    te_tok, te_seq, _ = evaluate(model, xte, yte)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    json.dump({"final_test_tok_acc": te_tok, "final_test_seq_acc": te_seq,
               "params": n_params, "steps_run": step, "stop_reason": stop_reason},
              open(done_path, "w"), indent=2)
    save_ckpt(step)
    print(f"FINAL test token acc {te_tok:.4f} | seq acc {te_seq:.4f} ({stop_reason})")
    log.close()


if __name__ == "__main__":
    main()

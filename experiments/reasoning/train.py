"""Unified trainer. Every architecture in the study goes through this exact
code path, so any difference in the results comes from the architecture and the
gradient scheme -- not from the recipe.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.fpsa_r import build_model                                  # noqa: E402
from src.fpsa_r.diagnostics import (ActivationMemory,               # noqa: E402
                                    empirical_spectral_radius)
from src.fpsa_r.implicit import BACKWARD_STATS                      # noqa: E402
from experiments.reasoning.tasks import make_task                   # noqa: E402


def stablemax(x: torch.Tensor) -> torch.Tensor:
    """s(x) = x+1 for x>=0, 1/(1-x) otherwise; normalised. Used by HRM/TRM/FPRM
    in place of softmax for the loss -- it does not saturate, which matters when
    a puzzle is nearly solved."""
    s = torch.where(x >= 0, x + 1.0, 1.0 / (1.0 - x))
    return s / s.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def stablemax_ce(logits: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    logp = torch.log(stablemax(logits.float()).clamp_min(1e-12))
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    m = mask.float()
    return (nll * m).sum() / m.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader, device, max_iter: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    tok_ok = tok_n = 0
    seq_ok = seq_n = 0
    iters_sum = n_batch = 0
    res_sum = 0.0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        out = model(x, max_iter=max_iter)
        pred = out["logits"].argmax(-1)
        ok = (pred == y) | ~m
        tok_ok += int((ok & m).sum())
        tok_n += int(m.sum())
        seq_ok += int(ok.all(dim=-1).sum())
        seq_n += x.shape[0]
        iters_sum += out["info"].n_iters
        res_sum += out["info"].rel_residual if math.isfinite(out["info"].rel_residual) else 0.0
        n_batch += 1
    return {"token_acc": 100.0 * tok_ok / max(tok_n, 1),
            "exact_match": 100.0 * seq_ok / max(seq_n, 1),
            "mean_iters": iters_sum / max(n_batch, 1),
            "mean_residual": res_sum / max(n_batch, 1)}


def run(args) -> Dict:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cpu"

    task = make_task(args.task, args.n_train, args.n_test, seed=args.seed,
                     n_blank=args.n_blank, size=args.maze_size, length=args.length,
                     group=args.group, n_gen=args.n_gen,
                     extra_blanks=args.extra_blanks, extra_lengths=args.extra_lengths,
                     extra_sizes=args.extra_sizes)

    # model sequence length must cover the longest evaluation sequence
    max_len = task.seq_len
    if task.extra_evals:
        max_len = max([max_len] + [d.seq_len for d in task.extra_evals.values()])

    cfg_kw = dict(vocab_size=task.vocab_size, out_vocab_size=task.out_vocab,
                  seq_len=max_len, conv_type=task.conv_type, causal=task.causal,
                  hidden_size=args.hidden, num_heads=args.heads,
                  n_block_layers=args.block_layers, expansion=args.expansion,
                  max_iter=args.max_iter, max_iter_eval=args.max_iter_eval,
                  n_backwards=args.n_backwards, fp_thresh=args.fp_thresh,
                  inner_damping=args.inner_damping,
                  contraction_lambda=args.contraction_lambda,
                  contraction_target=args.contraction_target, mlp_sigma=args.mlp_sigma)
    if args.arch == "transformer":
        # depth-matched non-recursive control: one pass through max_iter stacked
        # sub-blocks instead of max_iter passes through one.
        cfg_kw.update(n_block_layers=args.transformer_depth, max_iter=1, max_iter_eval=1)

    model = build_model(args.arch, **cfg_kw).to(device)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if getattr(p, "_no_weight_decay", False) or p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    train_loader = DataLoader(task.train, batch_size=args.bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(task.test, batch_size=args.bs)

    total_steps = args.steps
    warmup = max(1, int(0.05 * total_steps))

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (args.lr_min_ratio + (1 - args.lr_min_ratio)
                          * 0.5 * (1 + math.cos(math.pi * t)))

    # -- one-off measurements -------------------------------------------------
    xb, yb, mb = next(iter(train_loader))
    probe = ActivationMemory()
    model.train()
    model.zero_grad(set_to_none=True)
    with probe.track():
        out = model(xb)
        loss = stablemax_ce(out["logits"], yb, mb)
    loss.backward()
    act_mb = probe.mb
    model.zero_grad(set_to_none=True)

    t0 = time.time()
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        out = model(xb)
        stablemax_ce(out["logits"], yb, mb).backward()
    step_time = (time.time() - t0) / 3
    model.zero_grad(set_to_none=True)

    # -- training loop --------------------------------------------------------
    history = []
    step = 0
    t_start = time.time()
    it = iter(train_loader)
    best = {"exact_match": -1.0}
    while step < total_steps:
        try:
            x, y, m = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y, m = next(it)
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = stablemax_ce(out["logits"], y, m)
        if "ponder_cost" in out:
            loss = loss + args.ponder_coeff * out["ponder_cost"]
        if "contraction_loss" in out:
            loss = loss + model.cfg.contraction_lambda * out["contraction_loss"]
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        step += 1

        if step % args.eval_every == 0 or step == total_steps:
            ev = evaluate(model, test_loader, device)
            rec = {"step": step, "loss": float(loss.detach()), "gnorm": float(gnorm.detach()),
                   "elapsed": time.time() - t_start,
                   "train_iters": out["info"].n_iters,
                   "bwd_iters": BACKWARD_STATS["iters"],
                   "rho_train": model.last_rho, **ev}
            history.append(rec)
            if ev["exact_match"] > best["exact_match"]:
                best = dict(ev)
            if args.verbose:
                print(f"[{args.arch}/{args.task}/s{args.seed}] step {step:5d} "
                      f"loss {float(loss.detach()):.4f} tok {ev['token_acc']:.2f} "
                      f"em {ev['exact_match']:.2f} it {ev['mean_iters']:.1f}",
                      flush=True)

    # -- final diagnostics ----------------------------------------------------
    final = evaluate(model, test_loader, device)

    scaling = {}
    if args.arch != "transformer":
        for T in args.scaling_iters:
            scaling[str(T)] = evaluate(model, test_loader, device, max_iter=T)

    extra = {}
    if task.extra_evals:
        for name, ds in task.extra_evals.items():
            dl = DataLoader(ds, batch_size=args.bs)
            extra[name] = evaluate(model, dl, device)

    # measured contraction factor at the solution
    rho = None
    try:
        model.eval()
        xin = model._inputs(xb, None)
        si = model._seq_info(xin.shape[1])
        stepf = (model.block.nested_step if model.cfg.solver_mode == "nested"
                 else model.block.joint_step)
        step_fn = lambda s: stepf(s, xin, si)
        s = model.block.init_state(xb.shape[0], xin.shape[1], xin.device, xin.dtype)
        with torch.no_grad():
            for _ in range(args.max_iter_eval):
                s = s + (step_fn(s) - s)
        rho = empirical_spectral_radius(step_fn, s)
    except Exception:  # diagnostics must never sink a run
        rho = float("nan")

    result = {
        "arch": args.arch, "task": args.task, "seed": args.seed,
        "params": model.n_params(),
        "activation_mb": act_mb,
        "step_time_s": step_time,
        "total_time_s": time.time() - t_start,
        "final": final, "best": best,
        "scaling": scaling, "extra": extra,
        "spectral_radius": rho,
        "lipschitz_bound": model.lipschitz_report(),
        "history": history,
        "config": model.cfg.to_dict(),
        "args": vars(args),
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    return result


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="fpsa_r")
    p.add_argument("--task", default="sudoku")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ponder_coeff", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--block_layers", type=int, default=1)
    p.add_argument("--transformer_depth", type=int, default=8)
    p.add_argument("--expansion", type=float, default=2.0)
    p.add_argument("--max_iter", type=int, default=8)
    p.add_argument("--max_iter_eval", type=int, default=16)
    p.add_argument("--n_backwards", type=int, default=4)
    p.add_argument("--fp_thresh", type=float, default=1e-3)
    p.add_argument("--inner_damping", type=float, default=0.8)
    p.add_argument("--contraction_lambda", type=float, default=10.0)
    p.add_argument("--contraction_target", type=float, default=0.9)
    p.add_argument("--mlp_sigma", type=float, default=0.5)
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=512)
    p.add_argument("--n_blank", type=int, default=30)
    p.add_argument("--maze_size", type=int, default=9)
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--group", default="a5", choices=["s3", "s4", "a5", "s5"])
    p.add_argument("--n_gen", type=int, default=4)
    p.add_argument("--extra_blanks", type=int, nargs="*", default=[])
    p.add_argument("--extra_sizes", type=int, nargs="*", default=[])
    p.add_argument("--extra_lengths", type=int, nargs="*", default=[])
    p.add_argument("--scaling_iters", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default="")
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    r = run(build_parser().parse_args())
    print(json.dumps({k: r[k] for k in ("arch", "task", "seed", "params",
                                        "activation_mb", "step_time_s", "final")},
                     indent=1))

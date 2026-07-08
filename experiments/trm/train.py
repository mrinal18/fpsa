"""Training harness for TRM baseline and Implicit TRM on puzzle datasets.

Protocol matches the TRM reference (deep supervision with ACT: one supervision
step per optimizer step, per-slot carry persisting across steps, sequences
scored when they halt; EMA weights for eval; stablemax CE).

Single-GPU with optional gradient accumulation: pass --micro_batch_size to
split the global batch into independent carry streams (each micro-batch keeps
its own ACT carry, so semantics are preserved exactly).

Examples:
    # TRM baseline on Sudoku-Extreme (reference protocol)
    python train.py --config configs/sudoku_trm.yaml

    # Implicit TRM
    python train.py --config configs/sudoku_itrm.yaml

    # CPU smoke test
    python train.py --config configs/smoke_itrm.yaml
"""

import argparse

import json
import math
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import SplitData, TrainBatcher, iter_test_batches, load_split
from losses import ACTLossHead
from models.trm import TRMConfig, build_trm
from models.implicit_trm import build_implicit_trm


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DEFAULTS = dict(
    # data / run
    data_dir="data/sudoku-extreme-1k-aug-1000",
    arch="itrm",                    # trm | itrm
    run_name=None,
    results_dir="results",
    seed=0,
    # training
    epochs=50000,
    eval_interval=5000,
    global_batch_size=768,
    micro_batch_size=None,          # None = no accumulation
    lr=1e-4,
    lr_min_ratio=1.0,
    lr_warmup_steps=2000,
    weight_decay=1.0,
    puzzle_emb_lr=1e-4,
    puzzle_emb_weight_decay=1.0,
    beta1=0.9,
    beta2=0.95,
    optimizer="adamw",              # adamw | adam_atan2
    ema=True,
    ema_rate=0.999,
    grad_clip=None,                 # optional max-norm
    # arch (TRMConfig fields; see models/trm.py)
    H_cycles=3,
    L_cycles=6,
    L_layers=2,
    hidden_size=512,
    expansion=4.0,
    num_heads=8,
    pos_encodings="rope",
    puzzle_emb_len=16,
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    forward_dtype=None,             # None = bfloat16 on GPU, float32 on CPU
    mlp_t=False,
    # implicit arch
    norm_style=None,                # None = post for trm, pre for itrm
    residual_scale=None,            # None = False for trm, True for itrm
    alpha_1_init=0.75,
    alpha_2_init=0.25,
    spectral_norm=False,
    damping=0.9,
    stepsize_decay=0.9,
    decay_patience=5,
    inner_tol=1e-3,
    inner_max_iter=16,
    inner_max_iter_eval=64,
    adjoint_steps=10,
    adjoint_tol=1e-4,
    grad_mode="neumann",            # neumann | phantom | bptt
    bptt_steps=6,
    mask_nonconverged=True,
    jacobian_reg_lambda=0.0,
    jacobian_eps=1e-3,
    n_jacobian_samples=1,
    # loss
    loss_type="stablemax_cross_entropy",
    q_loss_coeff=0.5,
)


# Types for options whose DEFAULTS value is None (argparse can't infer them).
_NONE_DEFAULT_TYPES = {
    "run_name": str,
    "micro_batch_size": int,
    "grad_clip": float,
    "forward_dtype": str,
    "norm_style": str,
    "residual_scale": bool,
    "inner_max_iter_eval": int,
}


def _parse_bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes"):
        return True
    if s.lower() in ("0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


def parse_config() -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="YAML/JSON file with config overrides")
    for k, v in DEFAULTS.items():
        t = _NONE_DEFAULT_TYPES[k] if v is None else type(v)
        if t is bool:
            p.add_argument(f"--{k}", type=_parse_bool, default=None)
        else:
            p.add_argument(f"--{k}", type=t, default=None)
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            if args.config.endswith(".json"):
                file_cfg = json.load(f)
            else:
                import yaml
                file_cfg = yaml.safe_load(f)
        unknown = set(file_cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(file_cfg)
    for k in DEFAULTS:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    return cfg


# --------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------

class EMAHelper:
    """Initialization-debiased EMA over trainable parameters only.

    The raw shadow after N updates is  s_N = mu^N theta_0 + (1-mu) sum_t
    mu^(N-t) theta_t — contaminated by the random init theta_0 with weight
    mu^N. With mu=0.999 that weight is still 0.79 after 240 steps, so an
    early eval sees a mostly-random model (chance-level accuracy) even though
    the live weights have learned. swap_in therefore evaluates the debiased
    average of only the VISITED weights (same trick as Adam's bias
    correction):

        theta_hat = (s_N - mu^N theta_0) / (1 - mu^N)

    which converges to the plain EMA as N grows, so the long-run reference
    protocol (first eval at ~6.5k steps, mu^N ~ 1e-3) is unchanged while
    pilots and smoke runs report meaningful numbers from the first eval.

    Buffers are deliberately excluded: spectral-norm parametrizations keep
    power-iteration vectors (_u/_v) as buffers, and EMA-averaging those unit
    vectors would corrupt the sigma estimate used to normalize the weight at
    eval time. Init-constant buffers (z_init/y_init, rotary caches) don't
    change during training, so excluding them is lossless.

    Evaluation swaps EMA weights into the live model and restores afterwards —
    no deepcopy of the module.
    """

    def __init__(self, model: torch.nn.Module, mu: float):
        self.mu = mu
        self.n_updates = 0
        self.shadow = {
            k: p.detach().clone().float()
            for k, p in model.named_parameters()
            if p.requires_grad
        }
        self.init = {k: v.clone() for k, v in self.shadow.items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, p in model.named_parameters():
            if k in self.shadow:
                self.shadow[k].mul_(self.mu).add_(p.detach().float(), alpha=1 - self.mu)
        self.n_updates += 1

    def _debiased(self, k: str) -> torch.Tensor:
        decay_n = self.mu ** self.n_updates
        if self.n_updates == 0 or decay_n < 1e-4:
            return self.shadow[k]
        return (self.shadow[k] - decay_n * self.init[k]) / (1.0 - decay_n)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module):
        assert self._backup is None, "swap_in called twice without swap_out"
        self._backup = {}
        for k, p in model.named_parameters():
            if k in self.shadow:
                self._backup[k] = p.detach().clone()
                p.copy_(self._debiased(k).to(p.dtype))

    @torch.no_grad()
    def swap_out(self, model: torch.nn.Module):
        for k, p in model.named_parameters():
            if k in self._backup:
                p.copy_(self._backup[k])
        self._backup = None

    def state(self) -> dict:
        """Checkpointable state; debiased weights recoverable on load."""
        return {
            "shadow": self.shadow,
            "debiased": {k: self._debiased(k) for k in self.shadow},
            "n_updates": self.n_updates,
            "mu": self.mu,
        }


# --------------------------------------------------------------------------
# LR schedule
# --------------------------------------------------------------------------

def compute_lr(base_lr: float, step: int, total_steps: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if min_ratio >= 1.0:
        return base_lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(loss_head: ACTLossHead, test_data: SplitData, batch_size: int,
             device: torch.device) -> Dict[str, float]:
    loss_head.eval()
    model = loss_head.model

    totals = {"count": 0.0, "accuracy": 0.0, "exact_accuracy": 0.0,
              "inner_iters": 0.0, "iters_batches": 0.0}
    for batch, n_real in iter_test_batches(test_data, batch_size, device):
        carry = model.initial_carry(batch)  # allocated on the batch's device

        while True:
            carry, outputs = model(carry=carry, batch=batch)
            if bool(carry.halted.all()):
                break

        labels = carry.current_data["labels"]
        preds = outputs["logits"].argmax(dim=-1)
        mask = labels != -100
        loss_counts = mask.sum(-1)
        is_correct = mask & (preds == labels)
        seq_is_correct = is_correct.sum(-1) == loss_counts

        real = torch.zeros_like(carry.halted)
        real[:n_real] = True
        valid = real & (loss_counts > 0)

        totals["count"] += valid.sum().item()
        totals["accuracy"] += torch.where(
            valid, is_correct.float().sum(-1) / loss_counts.clamp_min(1), torch.zeros_like(loss_counts, dtype=torch.float)
        ).sum().item()
        totals["exact_accuracy"] += (valid & seq_is_correct).sum().item()
        if "stat_inner_iters" in outputs:
            totals["inner_iters"] += float(outputs["stat_inner_iters"])
            totals["iters_batches"] += 1

    count = max(totals["count"], 1)
    metrics = {
        "eval/accuracy": totals["accuracy"] / count,
        "eval/exact_accuracy": totals["exact_accuracy"] / count,
        "eval/count": count,
    }
    if totals["iters_batches"] > 0:
        metrics["eval/inner_iters_per_step"] = totals["inner_iters"] / totals["iters_batches"]
    return metrics


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    cfg = parse_config()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["forward_dtype"] is None:
        cfg["forward_dtype"] = "bfloat16" if device.type == "cuda" else "float32"
    if cfg["norm_style"] is None:
        cfg["norm_style"] = "pre" if cfg["arch"] == "itrm" else "post"
    if cfg["residual_scale"] is None:
        cfg["residual_scale"] = cfg["arch"] == "itrm"

    run_name = cfg["run_name"] or f"{cfg['arch']}-{os.path.basename(cfg['data_dir'])}-seed{cfg['seed']}"
    results_dir = os.path.join(cfg["results_dir"], run_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    log_path = os.path.join(results_dir, "log.jsonl")

    # Data
    train_data = load_split(cfg["data_dir"], "train")
    test_data = load_split(cfg["data_dir"], "test")
    meta = train_data.metadata

    micro_bs = cfg["micro_batch_size"] or cfg["global_batch_size"]
    assert cfg["global_batch_size"] % micro_bs == 0
    n_micro = cfg["global_batch_size"] // micro_bs

    model_config = TRMConfig(
        seq_len=meta["seq_len"],
        vocab_size=meta["vocab_size"],
        H_cycles=cfg["H_cycles"], L_cycles=cfg["L_cycles"], L_layers=cfg["L_layers"],
        hidden_size=cfg["hidden_size"], expansion=cfg["expansion"], num_heads=cfg["num_heads"],
        pos_encodings=cfg["pos_encodings"], puzzle_emb_len=cfg["puzzle_emb_len"],
        halt_max_steps=cfg["halt_max_steps"], halt_exploration_prob=cfg["halt_exploration_prob"],
        forward_dtype=cfg["forward_dtype"], mlp_t=cfg["mlp_t"],
        norm_style=cfg["norm_style"], residual_scale=cfg["residual_scale"],
        alpha_1_init=cfg["alpha_1_init"], alpha_2_init=cfg["alpha_2_init"],
        spectral_norm=cfg["spectral_norm"],
        damping=cfg["damping"], stepsize_decay=cfg["stepsize_decay"],
        decay_patience=cfg["decay_patience"], inner_tol=cfg["inner_tol"],
        inner_max_iter=cfg["inner_max_iter"], inner_max_iter_eval=cfg["inner_max_iter_eval"],
        adjoint_steps=cfg["adjoint_steps"], adjoint_tol=cfg["adjoint_tol"],
        grad_mode=cfg["grad_mode"], bptt_steps=cfg["bptt_steps"],
        mask_nonconverged=cfg["mask_nonconverged"],
        jacobian_reg_lambda=cfg["jacobian_reg_lambda"], jacobian_eps=cfg["jacobian_eps"],
        n_jacobian_samples=cfg["n_jacobian_samples"],
    )
    build = build_implicit_trm if cfg["arch"] == "itrm" else build_trm
    model = build(model_config).to(device)
    loss_head = ACTLossHead(
        model, loss_type=cfg["loss_type"], q_loss_coeff=cfg["q_loss_coeff"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{run_name}] arch={cfg['arch']} params={n_params:,} device={device} "
          f"dtype={cfg['forward_dtype']} groups={len(train_data.group_indices) - 1}")

    # Optimizer: puzzle prefix in its own group (matches reference puzzle_emb_lr)
    puzzle_params = [p for n, p in model.named_parameters() if "puzzle_emb" in n]
    main_params = [p for n, p in model.named_parameters() if "puzzle_emb" not in n]
    groups = [
        {"params": main_params, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
        {"params": puzzle_params, "lr": cfg["puzzle_emb_lr"], "weight_decay": cfg["puzzle_emb_weight_decay"]},
    ]
    if cfg["optimizer"] == "adam_atan2":
        try:
            from adam_atan2 import AdamATan2
            opt = AdamATan2(groups, betas=(cfg["beta1"], cfg["beta2"]))
        except ImportError:
            print("adam-atan2 not installed; falling back to AdamW")
            opt = torch.optim.AdamW(groups, betas=(cfg["beta1"], cfg["beta2"]))
    else:
        opt = torch.optim.AdamW(groups, betas=(cfg["beta1"], cfg["beta2"]))
    base_lrs = [g["lr"] for g in opt.param_groups]

    ema = EMAHelper(model, cfg["ema_rate"]) if cfg["ema"] else None

    num_groups = len(train_data.group_indices) - 1
    total_steps = int(cfg["epochs"] * num_groups / cfg["global_batch_size"])
    eval_every = max(1, int(cfg["eval_interval"] * num_groups / cfg["global_batch_size"]))
    print(f"total optimizer steps: {total_steps} (eval every {eval_every})")

    batchers = [TrainBatcher(train_data, micro_bs, seed=cfg["seed"] + i) for i in range(n_micro)]
    carries = [None] * n_micro

    best_exact = 0.0
    t0 = time.time()
    running = {}
    for step in range(1, total_steps + 1):
        loss_head.train()
        lr_now = None
        for g, base_lr in zip(opt.param_groups, base_lrs):
            lr_now = compute_lr(base_lr, step, total_steps, cfg["lr_warmup_steps"], cfg["lr_min_ratio"])
            g["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        for i in range(n_micro):
            batch = batchers[i].next_batch(device)
            if carries[i] is None:
                carries[i] = model.initial_carry(batch)  # on the batch device

            carries[i], loss, metrics, _, _ = loss_head(carry=carries[i], batch=batch)
            ((1.0 / cfg["global_batch_size"]) * loss).backward()
            # Accumulate as tensors — .float()/.item() here would force a
            # GPU->CPU sync per micro-batch; convert only at the log boundary.
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v.detach()

        if cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        if ema is not None:
            ema.update(model)

        running["_n"] = running.get("_n", 0) + 1

        if step % max(1, eval_every // 10) == 0:
            n = running.pop("_n")
            count = max(float(running.pop("count", 0.0)), 1.0)
            line = {"step": step, "lr": lr_now, "elapsed_s": round(time.time() - t0, 1)}
            for k, v in running.items():
                v = float(v)
                if k.endswith("loss"):
                    line[f"train/{k}"] = v / (cfg["global_batch_size"] * n)
                elif k.startswith("stat_"):
                    line[f"train/{k.removeprefix('stat_')}"] = v / (n * n_micro)
                else:
                    line[f"train/{k}"] = v / count  # per halted sequence
            line["train/halted_per_step"] = count / n
            if device.type == "cuda":
                line["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in line.items()}))
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            running = {}

        if step % eval_every == 0 or step == total_steps:
            if ema is not None:
                ema.swap_in(model)
            metrics = evaluate(loss_head, test_data, micro_bs, device)
            if ema is not None:
                ema.swap_out(model)
                # Live-weights eval alongside the EMA eval: separates "model
                # hasn't learned" from "EMA is lagging the model" at a glance.
                live = evaluate(loss_head, test_data, micro_bs, device)
                metrics.update({k.replace("eval/", "eval_live/"): v for k, v in live.items()})
            metrics["step"] = step
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()}))
            with open(log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

            if metrics["eval/exact_accuracy"] >= best_exact:
                best_exact = metrics["eval/exact_accuracy"]
                ckpt = {
                    "config": cfg,
                    "model": model.state_dict(),
                    "ema": ema.state() if ema is not None else None,
                    "step": step,
                    "eval": metrics,
                }
                torch.save(ckpt, os.path.join(results_dir, "best.pt"))
            torch.save({"config": cfg, "model": model.state_dict(),
                        "ema": ema.state() if ema is not None else None, "step": step},
                       os.path.join(results_dir, "last.pt"))

    print(f"done. best eval/exact_accuracy={best_exact:.4f}")


if __name__ == "__main__":
    main()

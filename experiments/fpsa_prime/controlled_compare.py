"""One controlled trainer for FPSA-Prime and the existing FPSA-R baselines.

Every arm uses the same datasets, batching, optimizer, learning-rate schedule,
loss, seed handling, evaluation code, and JSON schema.  The only family-specific
code is the model construction and the structural arguments required by the
pure attention model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from experiments.fpsa_prime.utils import (  # noqa: E402
    masked_loss,
    resolve_device,
    task_structure,
)
from experiments.reasoning.tasks import make_task  # noqa: E402
from src.fpsa_prime import build_model as build_prime_model  # noqa: E402
from src.fpsa_prime.implicit import BACKWARD_STATS as PRIME_BACKWARD_STATS  # noqa: E402
from src.fpsa_r import build_model as build_block_model  # noqa: E402
from src.fpsa_r.implicit import BACKWARD_STATS as BLOCK_BACKWARD_STATS  # noqa: E402


def _forward(
    family: str,
    model,
    tokens: torch.Tensor,
    *,
    relation_ids=None,
    attention_bias=None,
    max_iter: Optional[int] = None,
):
    if family == "prime":
        return model(
            tokens,
            relation_ids=relation_ids,
            attention_bias=attention_bias,
            max_iter=max_iter,
        )
    return model(tokens, max_iter=max_iter)


@torch.no_grad()
def evaluate(
    family: str,
    model,
    loader,
    *,
    device: torch.device,
    relation_ids=None,
    attention_bias=None,
    max_iter: Optional[int] = None,
) -> dict[str, float]:
    model.eval()
    exact = total = token_ok = token_total = 0
    residual = iterations = convergence = function_evals = 0.0
    batches = 0
    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        output = _forward(
            family,
            model,
            x,
            relation_ids=relation_ids,
            attention_bias=attention_bias,
            max_iter=max_iter,
        )
        logits = output["logits"]
        prediction = logits.argmax(-1)
        correct = (prediction == y) | ~mask
        exact += int(correct.all(dim=1).sum())
        total += x.shape[0]
        token_ok += int((correct & mask).sum())
        token_total += int(mask.sum())
        info = output["info"]
        residual += float(info.rel_residual)
        iterations += float(info.n_iters)
        convergence += float(getattr(info, "converged_frac", 1.0))
        function_evals += float(getattr(info, "n_function_evals", info.n_iters))
        batches += 1
    denominator = max(1, batches)
    return {
        "exact_match": 100.0 * exact / max(1, total),
        "token_accuracy": 100.0 * token_ok / max(1, token_total),
        "mean_residual": residual / denominator,
        "mean_iterations": iterations / denominator,
        "mean_function_evals": function_evals / denominator,
        "mean_converged_fraction": convergence / denominator,
    }


def _build_model(args, task, max_seq_len: int, relation_types: int):
    max_iter_eval = args.max_iter if args.max_iter_eval is None else args.max_iter_eval
    if args.family == "prime":
        return build_prime_model(
            args.arch,
            vocab_size=task.vocab_size,
            out_vocab_size=task.out_vocab,
            max_seq_len=max_seq_len,
            hidden_size=args.hidden,
            num_heads=args.heads,
            num_global_slots=args.global_slots,
            num_relation_types=relation_types,
            causal=task.causal,
            max_iter=args.max_iter,
            max_iter_eval=max_iter_eval,
            fp_tol=args.fp_tol,
            solver_damping=args.damping,
            forward_solver=args.prime_forward_solver,
            backward_solver=args.prime_backward_solver,
            backward_max_iter=args.backward_max_iter,
            backward_tol=args.backward_tol,
            gmres_restart=args.gmres_restart,
            stability_weight=args.stability_weight,
            stability_target=args.stability_target,
            stability_power_steps=args.stability_power_steps,
            stability_fd_eps=args.stability_fd_eps,
            require_convergence=not args.allow_nonconvergence,
            require_backward_convergence=not args.allow_inexact_backward,
        )

    return build_block_model(
        args.arch,
        vocab_size=task.vocab_size,
        out_vocab_size=task.out_vocab,
        seq_len=max_seq_len,
        hidden_size=args.hidden,
        num_heads=args.heads,
        n_block_layers=args.block_layers,
        expansion=args.expansion,
        conv_type=task.conv_type,
        causal=task.causal,
        max_iter=args.max_iter,
        max_iter_eval=max_iter_eval,
        fp_thresh=args.fp_tol,
        stepsize=args.damping,
        forward_solver=args.block_forward_solver,
        backward_solver=args.block_backward_solver,
        backward_max_iter=args.backward_max_iter,
        backward_tol=args.backward_tol,
        contraction_lambda=args.block_contraction_weight,
        contraction_target=args.block_contraction_target,
        inner_damping=args.inner_damping,
        spectral_norm=args.block_spectral_norm,
    )


def run(args) -> dict:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    task = make_task(
        args.task,
        args.n_train,
        args.n_test,
        seed=args.seed,
        n_blank=args.n_blank,
        size=args.maze_size,
        length=args.state_length,
        group=args.state_group,
        n_gen=args.state_generators,
        extra_blanks=args.extra_blanks,
        extra_sizes=args.extra_sizes,
        extra_lengths=args.extra_lengths,
    )
    max_seq_len = task.seq_len
    if task.extra_evals:
        max_seq_len = max(
            [max_seq_len] + [dataset.seq_len for dataset in task.extra_evals.values()]
        )

    relation_ids = attention_bias = None
    relation_types = 0
    if args.family == "prime":
        relation_ids, attention_bias, relation_types = task_structure(
            task,
            heads=args.heads,
            global_heads=args.global_heads,
            global_slots=args.global_slots,
            device=device,
        )

    model = _build_model(args, task, max_seq_len, relation_types).to(device)
    parameter_count = (
        model.n_params()
        if hasattr(model, "n_params")
        else sum(parameter.numel() for parameter in model.parameters())
    )

    decay, no_decay = [], []
    for parameter in model.parameters():
        if getattr(parameter, "_no_weight_decay", False) or parameter.ndim <= 1:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        task.train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        task.test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    warmup = max(1, int(0.05 * args.steps))

    def learning_rate(step: int) -> float:
        if step < warmup:
            return args.lr * step / warmup
        fraction = (step - warmup) / max(1, args.steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        return args.lr * (args.lr_min_ratio + (1.0 - args.lr_min_ratio) * cosine)

    print(
        f"family={args.family} arch={args.arch} params={parameter_count} "
        f"task={args.task} seed={args.seed}",
        flush=True,
    )
    history = []
    iterator = iter(train_loader)
    start = time.time()
    final_metrics = None

    for step in range(1, args.steps + 1):
        try:
            x, y, mask = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y, mask = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        model.train()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        output = _forward(
            args.family,
            model,
            x,
            relation_ids=relation_ids,
            attention_bias=attention_bias,
        )
        logits = output["logits"]
        loss = masked_loss(logits, y, mask)
        stability_loss = output.get("stability_loss")
        if stability_loss is not None:
            loss = loss + model.cfg.stability_weight * stability_loss
        contraction_loss = output.get("contraction_loss")
        if contraction_loss is not None:
            loss = loss + model.cfg.contraction_lambda * contraction_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("training loss became non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("gradient norm became non-finite")
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                args.family,
                model,
                test_loader,
                device=device,
                relation_ids=relation_ids,
                attention_bias=attention_bias,
            )
            backward_stats = (
                PRIME_BACKWARD_STATS if args.family == "prime" else BLOCK_BACKWARD_STATS
            )
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "learning_rate": learning_rate(step),
                "backward_iterations": int(backward_stats.get("iters", 0)),
                "backward_relative_residual": float(
                    backward_stats.get(
                        "relative_residual", backward_stats.get("rel", 0.0)
                    )
                ),
                **metrics,
            }
            history.append(record)
            final_metrics = metrics
            print(json.dumps(record, sort_keys=True), flush=True)

    extra = {}
    if task.extra_evals:
        for name, dataset in task.extra_evals.items():
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            extra_relations = extra_bias = None
            if args.family == "prime":
                extra_task = SimpleNamespace(name=task.name, seq_len=dataset.seq_len)
                extra_relations, extra_bias, _ = task_structure(
                    extra_task,
                    heads=args.heads,
                    global_heads=args.global_heads,
                    global_slots=args.global_slots,
                    device=device,
                )
            extra[name] = evaluate(
                args.family,
                model,
                loader,
                device=device,
                relation_ids=extra_relations,
                attention_bias=extra_bias,
            )

    result = {
        "family": args.family,
        "arch": args.arch,
        "task": args.task,
        "seed": args.seed,
        "params": parameter_count,
        "elapsed_seconds": time.time() - start,
        "final": final_metrics,
        "extra": extra,
        "history": history,
        "config": model.cfg.to_dict(),
        "args": vars(args),
    }
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2))
    if args.save_model:
        destination = Path(args.save_model)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": model.cfg.to_dict(), "state_dict": model.state_dict()},
            destination,
        )
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=("prime", "block"), required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--task", choices=("sudoku", "maze", "state_track"), default="maze")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_train", type=int, default=4096)
    p.add_argument("--n_test", type=int, default=512)
    p.add_argument("--n_blank", type=int, default=45)
    p.add_argument("--extra_blanks", type=int, nargs="*", default=[])
    p.add_argument("--maze_size", type=int, default=7)
    p.add_argument("--extra_sizes", type=int, nargs="*", default=[9, 11])
    p.add_argument("--state_length", type=int, default=16)
    p.add_argument("--extra_lengths", type=int, nargs="*", default=[])
    p.add_argument("--state_group", default="a5")
    p.add_argument("--state_generators", type=int, default=4)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--block_layers", type=int, default=1)
    p.add_argument("--expansion", type=float, default=2.0)
    p.add_argument("--global_heads", type=int, default=1)
    p.add_argument("--global_slots", type=int, default=0)
    p.add_argument("--max_iter", type=int, default=16)
    p.add_argument("--max_iter_eval", type=int, default=None)
    p.add_argument("--fp_tol", type=float, default=1e-4)
    p.add_argument("--damping", type=float, default=0.8)
    p.add_argument("--backward_max_iter", type=int, default=40)
    p.add_argument("--backward_tol", type=float, default=1e-5)
    p.add_argument("--gmres_restart", type=int, default=20)
    p.add_argument("--prime_forward_solver", choices=("picard", "anderson"), default="anderson")
    p.add_argument("--prime_backward_solver", choices=("gmres", "neumann"), default="gmres")
    p.add_argument("--stability_weight", type=float, default=1.0)
    p.add_argument("--stability_target", type=float, default=0.95)
    p.add_argument("--stability_power_steps", type=int, default=1)
    p.add_argument("--stability_fd_eps", type=float, default=1e-3)
    p.add_argument("--block_forward_solver", choices=("picard", "anderson", "broyden"), default="anderson")
    p.add_argument("--block_backward_solver", choices=("gmres", "anderson", "neumann"), default="gmres")
    p.add_argument("--block_contraction_weight", type=float, default=10.0)
    p.add_argument("--block_contraction_target", type=float, default=0.9)
    p.add_argument("--block_spectral_norm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--inner_damping", type=float, default=0.8)
    p.add_argument("--allow_nonconvergence", action="store_true")
    p.add_argument("--allow_inexact_backward", action="store_true")
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--output", default="")
    p.add_argument("--save_model", default="")
    return p


if __name__ == "__main__":
    run(parser().parse_args())

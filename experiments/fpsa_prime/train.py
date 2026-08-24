"""Small benchmark trainer for the pure FPSA-Prime architecture.

The script reuses the repository's generated Sudoku, Maze, and state-tracking
datasets while keeping the model and recurrence entirely in ``src.fpsa_prime``.
It is a correctness-first launcher, not yet the final benchmark harness.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from experiments.fpsa_prime.utils import (  # noqa: E402
    evaluate,
    masked_loss,
    resolve_device,
    task_structure,
)
from experiments.reasoning.tasks import make_task  # noqa: E402
from src.fpsa_prime import build_model  # noqa: E402
from src.fpsa_prime.losses import attractor_margin_loss, sudoku_energy  # noqa: E402


def main(args):
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
    )
    relation_ids, attention_bias, relation_types = task_structure(
        task,
        heads=args.heads,
        global_heads=args.global_heads,
        global_slots=args.global_slots,
        device=device,
    )
    if args.attractor_weight > 0 and task.name != "sudoku":
        raise ValueError("the v0 attractor curriculum currently supports Sudoku only")

    model = build_model(
        args.arch,
        vocab_size=task.vocab_size,
        out_vocab_size=task.out_vocab,
        max_seq_len=task.seq_len,
        num_global_slots=args.global_slots,
        hidden_size=args.hidden,
        num_heads=args.heads,
        num_relation_types=relation_types,
        causal=task.causal,
        max_iter=args.max_iter,
        max_iter_eval=args.max_iter_eval,
        fp_tol=args.fp_tol,
        solver_damping=args.damping,
        backward_max_iter=args.backward_max_iter,
        backward_tol=args.backward_tol,
        require_convergence=not args.allow_nonconvergence,
        require_backward_convergence=not args.allow_inexact_backward,
    ).to(device)

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
    iterator = iter(train_loader)

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
        optimizer.zero_grad(set_to_none=True)
        output = model(
            x,
            relation_ids=relation_ids,
            attention_bias=attention_bias,
            return_context=args.attractor_weight > 0,
        )
        logits = output["logits"]
        assert isinstance(logits, torch.Tensor)
        loss = masked_loss(logits, y, mask)
        task_energy = None
        if task.name == "sudoku" and (
            args.sudoku_energy_weight > 0 or args.attractor_weight > 0
        ):
            task_energy = sudoku_energy(logits, x)
        if task_energy is not None and args.sudoku_energy_weight > 0:
            loss = loss + args.sudoku_energy_weight * task_energy.mean()
        attractor_loss_value = logits.new_zeros(())
        if task_energy is not None and args.attractor_weight > 0:
            context = output.get("context")
            residual = output["residual"]
            if context is None or not isinstance(residual, torch.Tensor):
                raise RuntimeError("attractor training requires residual and context")
            fixed_map = lambda state: model.attention.fixed_map(state, context)
            attractor = attractor_margin_loss(
                fixed_map,
                residual.detach(),
                task_energy,
                positive_threshold=args.positive_energy_threshold,
                negative_threshold=args.negative_energy_threshold,
                negative_margin=args.negative_fp_margin,
            )
            attractor_loss_value = attractor["loss"]
            loss = loss + args.attractor_weight * attractor_loss_value
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("training loss became non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("gradient norm became non-finite")
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                test_loader,
                device=device,
                relation_ids=relation_ids,
                attention_bias=attention_bias,
            )
            print(
                f"step={step} loss={float(loss.detach()):.4f} "
                f"attr={float(attractor_loss_value.detach()):.4f} "
                f"em={metrics['exact_match']:.2f} "
                f"tok={metrics['token_accuracy']:.2f} "
                f"iters={metrics['mean_iterations']:.1f} "
                f"res={metrics['mean_residual']:.2e} "
                f"conv={metrics['mean_converged_fraction']:.3f}",
                flush=True,
            )

    if args.save:
        destination = Path(args.save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": model.cfg.to_dict(),
                "state_dict": model.state_dict(),
                "task": args.task,
                "seed": args.seed,
            },
            destination,
        )


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--arch", default="fpsa_prime")
    argument_parser.add_argument(
        "--task",
        choices=("sudoku", "maze", "state_track"),
        default="sudoku",
    )
    argument_parser.add_argument("--device", default="auto")
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--n_train", type=int, default=4096)
    argument_parser.add_argument("--n_test", type=int, default=512)
    argument_parser.add_argument("--n_blank", type=int, default=45)
    argument_parser.add_argument("--maze_size", type=int, default=9)
    argument_parser.add_argument("--state_length", type=int, default=16)
    argument_parser.add_argument("--state_group", default="a5")
    argument_parser.add_argument("--state_generators", type=int, default=4)
    argument_parser.add_argument("--steps", type=int, default=2000)
    argument_parser.add_argument("--batch_size", type=int, default=64)
    argument_parser.add_argument("--num_workers", type=int, default=0)
    argument_parser.add_argument("--lr", type=float, default=3e-4)
    argument_parser.add_argument("--weight_decay", type=float, default=1e-2)
    argument_parser.add_argument("--clip", type=float, default=1.0)
    argument_parser.add_argument("--hidden", type=int, default=128)
    argument_parser.add_argument("--heads", type=int, default=8)
    argument_parser.add_argument("--global_heads", type=int, default=2)
    argument_parser.add_argument("--global_slots", type=int, default=0)
    argument_parser.add_argument("--max_iter", type=int, default=24)
    argument_parser.add_argument("--max_iter_eval", type=int, default=64)
    argument_parser.add_argument("--fp_tol", type=float, default=1e-4)
    argument_parser.add_argument("--damping", type=float, default=0.8)
    argument_parser.add_argument("--backward_max_iter", type=int, default=40)
    argument_parser.add_argument("--backward_tol", type=float, default=1e-5)
    argument_parser.add_argument(
        "--allow_nonconvergence",
        action="store_true",
        help="diagnostic only: allow implicit training from an unconverged forward state",
    )
    argument_parser.add_argument(
        "--allow_inexact_backward",
        action="store_true",
        help="diagnostic only: accept an adjoint solve above backward_tol",
    )
    argument_parser.add_argument("--sudoku_energy_weight", type=float, default=0.01)
    argument_parser.add_argument("--attractor_weight", type=float, default=0.0)
    argument_parser.add_argument(
        "--positive_energy_threshold", type=float, default=0.05
    )
    argument_parser.add_argument(
        "--negative_energy_threshold", type=float, default=0.5
    )
    argument_parser.add_argument("--negative_fp_margin", type=float, default=0.05)
    argument_parser.add_argument("--eval_every", type=int, default=100)
    argument_parser.add_argument("--save", default="")
    return argument_parser


if __name__ == "__main__":
    main(parser().parse_args())

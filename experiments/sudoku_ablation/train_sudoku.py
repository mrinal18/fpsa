"""
Training Script for FPSA Value-Stream Ablation on Sudoku
=========================================================

Trains all 4 FPSA variants (+ vanilla baseline) on Sudoku puzzles
to empirically test whether fixing V = W_V(x) prevents constraint
propagation in spatial reasoning tasks.

Variants:
  1. "fixed"     — V static (original FPSA)
  2. "evolving"  — V recomputed from z^(k) each iteration
  3. "blended"   — Learned gate blends fixed and evolved V
  4. "fixed_ffn" — V fixed, but shared FFN inside the FPI loop
  5. "vanilla"   — Standard attention, k_max=1 (no iteration, baseline)

Usage:
  python3 train_sudoku.py --variants fixed evolving blended fixed_ffn vanilla
  python3 train_sudoku.py --variants blended --difficulty hard --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json
import os
import time
from collections import defaultdict

from fpsa_spatial import SpatialFPSAModel, effective_rank
from data_sudoku import SudokuDataset, get_dataloaders


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, device, epoch_num):
    """Train for one epoch. Returns dict of averaged metrics."""
    model.train()
    totals = defaultdict(float)
    n_batches = 0
    correct_cells = 0
    total_cells = 0
    correct_boards = 0
    total_boards = 0

    for batch in dataloader:
        grid_tokens = batch["grid_tokens"].to(device)
        row_ids = batch["row_ids"].to(device)
        col_ids = batch["col_ids"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        out = model(grid_tokens, row_ids, col_ids, target=target)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # --- Metrics ---
        totals["loss"] += loss.item()
        totals["total_steps"] += out["total_steps"]

        # Cell accuracy (only on masked cells)
        with torch.no_grad():
            preds = out["logits"].argmax(dim=-1)  # [B, 81]
            # target has -100 for given cells; we only evaluate masked ones
            eval_mask = target != -100
            if eval_mask.sum() > 0:
                correct_cells += (preds[eval_mask] == target[eval_mask]).sum().item()
                total_cells += eval_mask.sum().item()

            # Board accuracy: all masked cells correct
            B = grid_tokens.shape[0]
            for b in range(B):
                b_mask = eval_mask[b]
                if b_mask.sum() > 0:
                    board_correct = (preds[b][b_mask] == target[b][b_mask]).all().item()
                    correct_boards += int(board_correct)
                total_boards += 1

        n_batches += 1

    cell_acc = 100.0 * correct_cells / max(total_cells, 1)
    board_acc = 100.0 * correct_boards / max(total_boards, 1)
    avg_loss = totals["loss"] / max(n_batches, 1)
    avg_steps = totals["total_steps"] / max(n_batches, 1)

    return {
        "loss": avg_loss,
        "cell_accuracy": cell_acc,
        "board_accuracy": board_acc,
        "mean_steps": avg_steps,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model. Returns metrics dict."""
    model.eval()
    correct_cells = 0
    total_cells = 0
    correct_boards = 0
    total_boards = 0
    total_steps = 0
    n_batches = 0

    for batch in dataloader:
        grid_tokens = batch["grid_tokens"].to(device)
        row_ids = batch["row_ids"].to(device)
        col_ids = batch["col_ids"].to(device)
        target = batch["target"].to(device)

        out = model(grid_tokens, row_ids, col_ids, target=target)
        preds = out["logits"].argmax(dim=-1)

        eval_mask = target != -100
        if eval_mask.sum() > 0:
            correct_cells += (preds[eval_mask] == target[eval_mask]).sum().item()
            total_cells += eval_mask.sum().item()

        B = grid_tokens.shape[0]
        for b in range(B):
            b_mask = eval_mask[b]
            if b_mask.sum() > 0:
                correct_boards += int((preds[b][b_mask] == target[b][b_mask]).all().item())
            total_boards += 1

        total_steps += out["total_steps"]
        n_batches += 1

    return {
        "cell_accuracy": 100.0 * correct_cells / max(total_cells, 1),
        "board_accuracy": 100.0 * correct_boards / max(total_boards, 1),
        "mean_steps": total_steps / max(n_batches, 1),
    }


@torch.no_grad()
def collect_convergence_trace(model, dataloader, device, k_max_override=30):
    """Run model with high k_max to trace convergence and rank evolution.

    Returns:
        residual_norms: list of lists (per layer, per iteration)
        effective_ranks: list of lists (per layer, per iteration)
    """
    model.eval()

    # Override k_max and disable early stopping
    original_settings = []
    for layer in model.layers:
        attn = layer.attn
        original_settings.append((attn.k_max, attn.epsilon))
        attn.k_max = k_max_override
        attn.epsilon = 0

    # Collect from first batch only
    batch = next(iter(dataloader))
    grid_tokens = batch["grid_tokens"].to(device)
    row_ids = batch["row_ids"].to(device)
    col_ids = batch["col_ids"].to(device)

    # We need to manually trace rank at each iteration
    # For now, just collect residual norms from the model output
    out = model(grid_tokens, row_ids, col_ids)
    residual_norms = [info["residual_norms"] for info in out["layer_infos"]]

    # Collect effective rank by running attention manually with tracking
    # (Simplified: compute rank of final hidden states per layer)
    x = model.token_embed(grid_tokens)
    ranks_per_layer = []
    for layer in model.layers:
        # Run one step at a time to track rank
        attn = layer.attn
        z = x.clone()
        if attn.value_mode in ("fixed", "fixed_ffn"):
            v = attn._get_values(x, x)
        else:
            v = None

        ranks = []
        for k in range(k_max_override):
            z_new = attn._one_step(z, x, v, row_ids, col_ids)
            ranks.append(effective_rank(z_new))
            z = z_new
        ranks_per_layer.append(ranks)

        # Pass through full layer for next iteration
        x, _ = layer(x, row_ids, col_ids)

    # Restore
    for layer, (km, eps) in zip(model.layers, original_settings):
        layer.attn.k_max = km
        layer.attn.epsilon = eps

    return {
        "residual_norms": residual_norms,
        "effective_ranks": ranks_per_layer,
    }


# =============================================================================
# Main Training Orchestration
# =============================================================================

def train_variant(variant, args, device):
    """Train a single variant and return results."""
    print(f"\n{'='*60}")
    print(f"  Training variant: {variant.upper()}")
    print(f"  Difficulty: {args.difficulty} | Epochs: {args.epochs}")
    print(f"{'='*60}")

    # Data
    train_loader, val_loader = get_dataloaders(
        difficulty=args.difficulty,
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
    )

    # Model
    k_max = 1 if variant == "vanilla" else args.k_max
    value_mode = "fixed" if variant == "vanilla" else variant

    model = SpatialFPSAModel(
        vocab_size=10,  # 0=MASK, 1-9=digits
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        value_mode=value_mode,
        k_max=k_max,
        epsilon=args.epsilon,
        damping=args.damping,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        use_spectral_norm=args.spectral_norm,
        max_grid_size=9,
        grad_mode=args.grad_mode,
        adjoint_steps=args.adjoint_steps,
    ).to(device)

    n_params = model.num_parameters()
    print(f"  Parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader),
    )

    # Training loop
    history = defaultdict(list)
    best_board_acc = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)

        # Record
        for k, v in train_metrics.items():
            history[k].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)

        elapsed = time.time() - t0

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{args.epochs} | "
                f"Loss={train_metrics['loss']:.4f} | "
                f"Cell={train_metrics['cell_accuracy']:.1f}% | "
                f"Board={train_metrics['board_accuracy']:.1f}% | "
                f"Steps={train_metrics['mean_steps']:.1f} | "
                f"Val Board={val_metrics['board_accuracy']:.1f}% | "
                f"{elapsed:.1f}s"
            )

        # Track best
        if val_metrics["board_accuracy"] > best_board_acc:
            best_board_acc = val_metrics["board_accuracy"]

    # Convergence diagnostics (post-training)
    print(f"  Collecting convergence trace...")
    conv_trace = collect_convergence_trace(
        model, val_loader, device, k_max_override=30
    )

    # Final eval
    final_eval = evaluate(model, val_loader, device)
    print(f"  Final — Cell: {final_eval['cell_accuracy']:.1f}% | "
          f"Board: {final_eval['board_accuracy']:.1f}% | "
          f"Steps: {final_eval['mean_steps']:.1f}")

    # Save checkpoint
    ckpt_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{variant}.pt"))

    return {
        "variant": variant,
        "difficulty": args.difficulty,
        "n_params": n_params,
        "train_history": dict(history),
        "eval_metrics": final_eval,
        "convergence_trace": conv_trace,
        "best_board_accuracy": best_board_acc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="FPSA Value-Stream Ablation on Sudoku"
    )

    # Variants
    parser.add_argument(
        "--variants", nargs="+",
        default=["fixed", "evolving", "blended", "fixed_ffn", "vanilla"],
        help="Which variants to train",
    )

    # Data
    parser.add_argument("--difficulty", default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--num_train", type=int, default=5000)
    parser.add_argument("--num_val", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--k_max", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--ffn_mult", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spectral_norm", action="store_true", default=True)
    parser.add_argument("--no_spectral_norm", dest="spectral_norm",
                        action="store_false")
    parser.add_argument("--grad_mode", default="unroll",
                        choices=["unroll", "phantom", "neumann"],
                        help="Backward through the FPI loop: full unrolled "
                             "autograd, 1-step phantom gradient, or Neumann-"
                             "refined implicit gradient (O(1) memory)")
    parser.add_argument("--adjoint_steps", type=int, default=10,
                        help="Neumann terms for the adjoint solve (neumann mode)")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=5)

    # Output
    parser.add_argument("--results_dir", default="results/sudoku")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)

    # Results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Train each variant
    all_results = {}
    for variant in args.variants:
        results = train_variant(variant, args, device)
        all_results[variant] = results

        # Save individual results
        result_path = os.path.join(args.results_dir, f"{variant}_results.json")
        # Convert non-serializable items
        serializable = {
            "variant": results["variant"],
            "difficulty": results["difficulty"],
            "n_params": results["n_params"],
            "train_history": results["train_history"],
            "eval_metrics": results["eval_metrics"],
            "convergence_trace": results["convergence_trace"],
            "best_board_accuracy": results["best_board_accuracy"],
        }
        with open(result_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  Saved: {result_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY — Difficulty: {args.difficulty}")
    print(f"{'='*70}")
    print(f"  {'Variant':<12} {'Params':>10} {'Cell Acc':>10} {'Board Acc':>10} {'Steps':>8}")
    print(f"  {'-'*52}")
    for variant, res in all_results.items():
        em = res["eval_metrics"]
        print(
            f"  {variant:<12} {res['n_params']:>10,} "
            f"{em['cell_accuracy']:>9.1f}% "
            f"{em['board_accuracy']:>9.1f}% "
            f"{em['mean_steps']:>7.1f}"
        )
    print()


if __name__ == "__main__":
    main()

"""
FPSA Value-Stream Ablation — Google Colab Runner
=================================================

Upload this single file to Google Colab with A100 GPU runtime.
It imports the local modules and runs the full ablation.

Prerequisites: upload fpsa_spatial.py, data_sudoku.py, train_sudoku.py
to the same directory (or clone your repo).

Usage in Colab:
    !python3 run_colab_ablation.py
"""

import subprocess
import sys
import os
import json
import time

# Ensure we're using GPU
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("   ⚠️  No GPU detected! Training will be slow.")
    print("   Go to Runtime → Change runtime type → A100 GPU")

print()

# ============================================================
# Run the full ablation
# ============================================================

VARIANTS = ["fixed", "evolving", "blended", "fixed_ffn", "vanilla"]
RESULTS_DIR = "results/sudoku"

# --- Configuration ---
# Increase these for a more thorough experiment on GPU:
CONFIG = {
    "difficulty": "medium",   # easy/medium/hard
    "epochs": 50,             # more epochs on GPU
    "num_train": 5000,        # more data on GPU
    "num_val": 500,
    "batch_size": 128,        # larger batch on GPU
    "k_max": 16,              # full FPI depth
    "hidden_dim": 128,
    "num_layers": 4,
    "num_heads": 4,
    "log_every": 10,
}

cmd = [
    sys.executable, "train_sudoku.py",
    "--variants", *VARIANTS,
    "--difficulty", CONFIG["difficulty"],
    "--epochs", str(CONFIG["epochs"]),
    "--num_train", str(CONFIG["num_train"]),
    "--num_val", str(CONFIG["num_val"]),
    "--batch_size", str(CONFIG["batch_size"]),
    "--k_max", str(CONFIG["k_max"]),
    "--hidden_dim", str(CONFIG["hidden_dim"]),
    "--num_layers", str(CONFIG["num_layers"]),
    "--num_heads", str(CONFIG["num_heads"]),
    "--log_every", str(CONFIG["log_every"]),
]

print(f"🚀 Running: {' '.join(cmd)}")
print(f"   Config: {CONFIG}")
print()

t0 = time.time()
result = subprocess.run(cmd, capture_output=False)
elapsed = time.time() - t0

print(f"\n⏱️  Total training time: {elapsed/60:.1f} minutes")

# ============================================================
# Print final comparison
# ============================================================

print(f"\n{'='*70}")
print(f"  FINAL RESULTS — {CONFIG['difficulty'].upper()} Sudoku")
print(f"{'='*70}")
print(f"  {'Variant':<12} {'Params':>10} {'Cell Acc':>10} {'Board Acc':>10} {'Steps':>8}")
print(f"  {'-'*52}")

for variant in VARIANTS:
    result_path = os.path.join(RESULTS_DIR, f"{variant}_results.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            d = json.load(f)
        em = d["eval_metrics"]
        print(
            f"  {variant:<12} {d['n_params']:>10,} "
            f"{em['cell_accuracy']:>9.1f}% "
            f"{em['board_accuracy']:>9.1f}% "
            f"{em['mean_steps']:>7.1f}"
        )
    else:
        print(f"  {variant:<12} {'MISSING':>10}")

print()

# ============================================================
# Generate analysis plots
# ============================================================

try:
    print("📊 Generating analysis plots...")
    subprocess.run([sys.executable, "analyze_sudoku.py"], capture_output=False)
    print("   Plots saved to results/sudoku/")
except Exception as e:
    print(f"   Plot generation failed: {e}")

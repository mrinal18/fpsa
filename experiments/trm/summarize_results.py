"""Summarize training runs into a markdown results table.

Scans results/<run_name>/log.jsonl files, extracts the best eval metrics and
compute statistics, and prints a table for the README/paper.

Usage:
    python summarize_results.py --results-dir results
"""

import argparse
import glob
import json
import os


def summarize_run(run_dir: str):
    log_path = os.path.join(run_dir, "log.jsonl")
    cfg_path = os.path.join(run_dir, "config.json")
    if not (os.path.exists(log_path) and os.path.exists(cfg_path)):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)

    best = None
    last_train = None
    peak_mem = None
    for line in open(log_path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval/exact_accuracy" in rec:
            if best is None or rec["eval/exact_accuracy"] > best["eval/exact_accuracy"]:
                best = rec
        if "train/lm_loss" in rec:
            last_train = rec
            peak_mem = rec.get("peak_mem_gb", peak_mem)

    if best is None:
        return None
    return {
        "run": os.path.basename(run_dir),
        "arch": cfg.get("arch"),
        "grad_mode": cfg.get("grad_mode") if cfg.get("arch") == "itrm" else "unroll(last cycle)",
        "seed": cfg.get("seed"),
        "exact_acc": best["eval/exact_accuracy"],
        "cell_acc": best.get("eval/accuracy"),
        "at_step": best.get("step"),
        "inner_iters_eval": best.get("eval/inner_iters_per_step"),
        "inner_iters_train": (last_train or {}).get("train/inner_iters"),
        "peak_mem_gb": peak_mem,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.results_dir, "*"))):
        if os.path.isdir(run_dir):
            row = summarize_run(run_dir)
            if row:
                rows.append(row)

    if not rows:
        print("no completed runs found")
        return

    cols = ["run", "arch", "grad_mode", "seed", "exact_acc", "cell_acc",
            "at_step", "inner_iters_train", "inner_iters_eval", "peak_mem_gb"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                v = f"{v:.4f}" if "acc" in c else f"{v:.2f}"
            cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

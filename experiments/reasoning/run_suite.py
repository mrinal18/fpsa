"""Run the full comparison grid, in parallel across CPU workers.

Usage:
    python experiments/reasoning/run_suite.py --stage main    --workers 4
    python experiments/reasoning/run_suite.py --stage ablation --workers 4
    python experiments/reasoning/run_suite.py --stage mechanism --workers 1
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN = os.path.join(HERE, "train.py")

MAIN_ARCHS = ["fpsa_r", "deq_block", "fprm", "looped_bptt", "ut_act", "transformer"]
ABLATIONS = ["fpsa_r", "fpsa_r_nested", "fpsa_r_bptt", "fpsa_r_onestep",
             "fpsa_r_nomask", "fpsa_r_neumann", "fpsa_r_nospec"]

TASKS = {
    # Primary benchmark: 7x7 shortest-path planning, with held-out larger grids
    # to test whether test-time iteration buys generalisation.
    "maze7": ["--task", "maze", "--maze_size", "7", "--extra_sizes", "9", "11",
              "--max_iter", "8", "--max_iter_eval", "32", "--bs", "32",
              "--n_train", "8000", "--n_test", "512"],
    "maze9": ["--task", "maze", "--maze_size", "9",
              "--max_iter", "8", "--max_iter_eval", "32", "--bs", "32",
              "--n_train", "8000", "--n_test", "512"],
    "sudoku": ["--task", "sudoku", "--n_blank", "25", "--extra_blanks", "35",
               "--max_iter", "8", "--max_iter_eval", "32", "--bs", "32",
               "--n_train", "6000", "--n_test", "256"],
    "state_track": ["--task", "state_track", "--length", "16", "--group", "a5",
                    "--extra_lengths", "24", "32",
                    "--max_iter", "8", "--max_iter_eval", "64",
                    "--n_train", "12000", "--n_test", "512"],
}


def job(arch, task, seed, steps, outdir, extra=()):
    out = os.path.join(outdir, f"{task}__{arch}__s{seed}.json")
    cmd = [sys.executable, TRAIN, "--arch", arch, "--seed", str(seed),
           "--steps", str(steps), "--threads", "1", "--out", out, "--verbose"]
    cmd += TASKS[task] + list(extra)
    return out, cmd


def run_all(jobs, workers):
    pending = list(jobs)
    running = []
    done = 0
    t0 = time.time()
    while pending or running:
        while pending and len(running) < workers:
            out, cmd = pending.pop(0)
            if os.path.exists(out):
                done += 1
                continue
            log = out.replace(".json", ".log")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            f = open(log, "w")
            running.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT), f, out))
        time.sleep(5)
        for p, f, out in list(running):
            if p.poll() is not None:
                f.close()
                running.remove((p, f, out))
                done += 1
                status = "ok" if os.path.exists(out) else f"FAILED(rc={p.returncode})"
                print(f"[{done}/{len(jobs)}] {os.path.basename(out)} {status} "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main",
                    choices=["main", "ablation", "mechanism", "all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--ablation_task", default="maze7")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "runs"))
    ap.add_argument("--tasks", nargs="*", default=["maze7", "maze9"])
    args = ap.parse_args()

    jobs = []
    if args.stage in ("main", "all"):
        for task, arch, seed in itertools.product(args.tasks, MAIN_ARCHS, args.seeds):
            jobs.append(job(arch, task, seed, args.steps, args.outdir))
    if args.stage in ("ablation", "all"):
        for arch, seed in itertools.product(ABLATIONS, args.seeds):
            jobs.append(job(arch, args.ablation_task, seed, args.steps,
                            os.path.join(args.outdir, "..", "ablation")))
    print(f"{len(jobs)} jobs, {args.workers} workers")
    run_all(jobs, args.workers)


if __name__ == "__main__":
    main()

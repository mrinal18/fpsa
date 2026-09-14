"""Run either FPSA ARC or unmodified TRM through the same pinned official trainer.

No scores are predicted and no fallback optimizer/precision/solver is selected
silently. A dry run prints exact commands and the planned exposure budget.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import platform
from .upstream import checkout, install_shims, TRM_SHA, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', default='.external/trm')
    p.add_argument('--data', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--arch', choices=['trm', 'fpsa_arc_refine', 'fpsa_arc_dual', 'fpsa_arc_bptt',
                                      'fpsa_arc_block', 'fpsa_arc_legacy', 'fpsa_arc_deq'], default='fpsa_arc_refine')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=768)
    p.add_argument('--epochs', type=int, default=100000)
    p.add_argument('--eval-interval', type=int, default=10000)
    p.add_argument('--voting', choices=['checkpoint', 'history'], default='checkpoint')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--override', action='append', default=[], help='Explicit Hydra override; recorded in manifest')
    a = p.parse_args()
    if min(a.devices, a.batch_size, a.epochs, a.eval_interval) < 1:
        raise ValueError('Counts must be positive')
    if a.batch_size % a.devices or a.epochs % a.eval_interval:
        raise ValueError('Batch must divide across devices; eval interval must divide epochs')
    data, run_dir = Path(a.data).resolve(), Path(a.run_dir).resolve()
    metadata = json.loads((data / 'train/dataset.json').read_text())
    upstream = checkout(a.upstream)
    install_shims(upstream)
    planned_steps = int(a.epochs * metadata['total_groups'] * metadata['mean_puzzle_examples'] / a.batch_size)
    overrides = [f'arch={a.arch}', f'data_paths=[{str(data)}]', f'seed={a.seed}',
                 f'global_batch_size={a.batch_size}', f'epochs={a.epochs}', f'eval_interval={a.eval_interval}',
                 f'+checkpoint_path={run_dir}', f'+run_name={run_dir.name}', 'ema=True',
                 '+eval_save_outputs=[inputs,puzzle_identifiers,q_halt_logits,preds]',
                 'evaluators=[{name:arc@ARC,aggregated_voting:' + str(a.voting == 'history').lower() + '}]']
    if a.arch == 'trm':
        overrides += ['arch.L_layers=2', 'arch.H_cycles=3', 'arch.L_cycles=4']
    overrides += a.override
    entry = ['-m', 'experiments.arc.official_entry', '--upstream', str(upstream), *overrides]
    # ARC's official evaluator uses collectives even at world_size=1.
    # torchrun initializes the process group for the single-GPU path too.
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={a.devices}', *entry]
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        git_sha = 'not-a-git-checkout'
    manifest = dict(command=command, argv=vars(a), fpsa_commit=git_sha, trm_commit=TRM_SHA,
                    python=platform.python_version(), official_estimated_optimizer_updates=planned_steps,
                    note='Data and optimizer/EMA/sampler are official; model compute differs and must be measured.',
                    compiler='disabled for both arms', fpsa_solver_dtype='float32',
                    candidate_history=a.voting, source_19_percent_run_reproduced=False)
    print(json.dumps(manifest, indent=2), flush=True)
    print('$ ' + shlex.join(command), flush=True)
    if a.dry_run:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Full official trainer requires CUDA; use experiments.arc.smoke on CPU')
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('Use a NEW run directory; old weights/results will not be overwritten')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', DISABLE_COMPILE='1',
               FPSA_ARC_DIAGNOSTICS=str(run_dir / 'diagnostics'), FPSA_ARC_SEED=str(a.seed))
    env.setdefault('WANDB_MODE', 'offline')
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + str(upstream) + os.pathsep + env.get('PYTHONPATH', '')
    log_path = run_dir / 'console.log'
    with log_path.open('w') as log:
        with subprocess.Popen(command, cwd=upstream, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            try:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = proc.wait()
            except BaseException:
                proc.terminate()
                proc.wait()
                raise
    (run_dir / 'status.json').write_text(json.dumps(dict(exit_code=code, completed=code == 0)))
    if code:
        print('\n'.join(log_path.read_text().splitlines()[-100:]), file=sys.stderr)
        raise SystemExit(code)

if __name__ == '__main__':
    main()

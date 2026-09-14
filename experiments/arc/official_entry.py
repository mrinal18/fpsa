"""Instrumentation around the official trainer, not a second training recipe.

Hooks add diagnostics and raw checkpoints. They do not alter the optimizer,
EMA, batches, losses, or ACT schedule. Full sampler state is not serialized;
checkpoints are NOT advertised as bit-exact resumable training snapshots.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import traceback
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--upstream', required=True)
    own, rest = p.parse_known_args()
    sys.path.insert(0, str(Path(own.upstream).resolve()))
    sys.argv = [str(Path(own.upstream) / 'pretrain.py'),
                '--config-path', str(Path(own.upstream).resolve() / 'config'), *rest]
    import pretrain
    seed = int(os.environ.get('FPSA_ARC_SEED', '0')) + int(os.environ.get('RANK', '0'))
    random.seed(seed)
    np.random.seed(seed)
    old_train, old_save = pretrain.train_batch, pretrain.save_train_state
    latest = {}
    root = Path(os.environ.get('FPSA_ARC_DIAGNOSTICS', 'results/arc_diagnostics'))
    root.mkdir(parents=True, exist_ok=True)

    def train(config, state, batch, global_batch_size, rank, world_size):
        latest['state'] = state
        try:
            result = old_train(config, state, batch, global_batch_size, rank, world_size)
        except BaseException as exc:
            path = root / f'train_failure_rank{rank}.pt'
            # Atomic model/optimizer/carry dump at the failing step, not an
            # attempt to continue using an invalid equilibrium gradient.
            torch.save(dict(model=state.model.state_dict(), step=state.step,
                            optimizers=[v.state_dict() for v in state.optimizers],
                            carry=state.carry, batch=batch, traceback=traceback.format_exc(),
                            exact_resume=False), path.with_suffix('.tmp'))
            path.with_suffix('.tmp').replace(path)
            raise
        module = state.model.model
        core = getattr(getattr(module, 'inner', None), 'core', None)
        if core is not None and (state.step <= 3 or state.step % 10 == 0):
            record = dict(step=state.step, phase='train', rank=rank,
                          solver=[i.as_dict() for i in core.last_infos])
            with (root / f'numerics_rank{rank}.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
        return result

    def save(config, evaluation_state):
        old_save(config, evaluation_state)
        if config.checkpoint_path and 'state' in latest:
            raw = latest['state']
            path = Path(config.checkpoint_path) / f'step_{raw.step}_raw.pt'
            torch.save(dict(model=raw.model.state_dict(), step=raw.step,
                            config=config.model_dump(), weights='raw', exact_resume=False), path)

    pretrain.train_batch, pretrain.save_train_state = train, save
    pretrain.launch()

if __name__ == '__main__':
    main()

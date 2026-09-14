"""Model/loss adapters for the PINNED official TRM pretrain.py.

Reuse upstream data sampling, sparse embedding optimizer, dense optimizer, EMA,
training-time ACT and max-segment evaluation. Do not import upstream until the
adapter is constructed; standalone core tests require only PyTorch.
"""
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import time
import torch
from torch import nn
import torch.nn.functional as F
from .model import ARCConfig, ARCReasoner, Carry
from .legacy import LegacyARC
from .losses import per_example_loss
from .numerics import ConvergenceError


@dataclass
class ACTCarry:
    inner_carry: Carry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: dict


class _Inner(nn.Module):
    def __init__(self, cfg, task_dim, num_ids, batch_size, kind):
        super().__init__()
        from models.sparse_embedding import CastedSparseEmbedding
        self.core = (LegacyARC(cfg, fpsa=kind == 'legacy_joint')
                     if kind in {'legacy_joint', 'legacy_block'} else ARCReasoner(cfg))
        self.puzzle_emb = (CastedSparseEmbedding(num_ids, task_dim, batch_size,
                                               init_std=0., cast_to=torch.float32)
                           if task_dim else None)


class FPSAARC_ACTV1(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        raw = dict(config_dict)
        allowed = ARCConfig.__dataclass_fields__
        self.arc_cfg = ARCConfig(**{k: v for k, v in raw.items() if k in allowed})
        self.kind = raw.get('kind', 'answer_refine')
        if self.kind not in {'answer_refine', 'legacy_joint', 'legacy_block'}:
            raise ValueError('Unknown ARC model kind')
        self.task_dim = int(raw.get('puzzle_emb_ndim', self.arc_cfg.hidden_size))
        self.num_ids = int(raw['num_puzzle_identifiers'])
        if bool(self.task_dim) != bool(self.arc_cfg.puzzle_emb_len):
            raise ValueError('Disable both puzzle_emb_ndim and puzzle_emb_len, or enable both')
        self.max_segments = int(raw.get('halt_max_steps', self.arc_cfg.refinement_steps))
        if self.kind.startswith('legacy') and self.max_segments != 1:
            raise ValueError('Legacy joint control solves once; repeated identical equilibria are not refinement')
        self.explore = float(raw.get('halt_exploration_prob', .1))
        if self.max_segments < 1 or not 0 <= self.explore <= 1:
            raise ValueError('Invalid halting configuration')
        self.inner = _Inner(self.arc_cfg, self.task_dim, self.num_ids, int(raw['batch_size']), self.kind)
        self._diagnostic_index = 0
        print('ARC_PARAMETER_REPORT=' + json.dumps(self.inner.core.parameter_report(self.num_ids, self.task_dim)), flush=True)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch):
        b, device = batch['inputs'].shape[0], batch['inputs'].device
        return ACTCarry(self.inner.core.initial_carry(b, device=device),
                        torch.zeros(b, device=device, dtype=torch.long),
                        torch.ones(b, device=device, dtype=torch.bool),
                        {k: torch.zeros_like(v) for k, v in batch.items()})

    def _failure(self, error, data, carry):
        root = os.environ.get('FPSA_ARC_DIAGNOSTICS')
        if not root:
            return
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        tag = f"failure_rank{os.environ.get('RANK', '0')}_{time.time_ns()}"
        record = dict(error=str(error), kind=self.kind, config=asdict(self.arc_cfg),
                      diagnostic_only=True, resume_supported=False)
        (path / f'{tag}.json').write_text(json.dumps(record, indent=2))
        # For diagnosis, not exact resume: no optimizer/dataloader state here.
        torch.save(dict(model=self.state_dict(), inputs=data['inputs'],
                        puzzle_identifiers=data['puzzle_identifiers'], carry=carry,
                        metadata=record), path / f'{tag}.pt')

    def forward(self, carry: ACTCarry, batch):
        reset = carry.halted
        fresh = self.inner.core.initial_carry(batch['inputs'].shape[0], device=batch['inputs'].device)
        previous = carry.inner_carry
        inner = Carry(torch.where(reset[:, None, None], fresh.answer, previous.answer),
                      tuple(torch.where(reset[:, None, None], a, b)
                            for a, b in zip(fresh.latent, previous.latent)))
        data = {k: torch.where(reset.reshape((-1,) + (1,) * (v.ndim-1)), v,
                               carry.current_data[k]) for k, v in batch.items()}
        steps = torch.where(reset, 0, carry.steps)
        task = self.puzzle_emb(data['puzzle_identifiers'].long()) if self.task_dim else None
        try:
            new, out = self.inner.core.forward_segment(data['inputs'], inner, task, strict=True)
        except (ConvergenceError, FloatingPointError) as error:
            self._failure(error, data, inner)
            raise
        steps = steps + 1
        halted = steps >= self.max_segments
        with torch.no_grad():
            if self.training and self.max_segments > 1:
                halted |= out['q_halt_logits'] > 0
                explore = torch.rand_like(out['q_halt_logits']) < self.explore
                minimum = torch.randint(2, self.max_segments+1, steps.shape, device=steps.device)
                halted &= steps >= torch.where(explore, minimum, 0)
        self._diagnostic_index += 1
        # The official harness receives tensors only in outputs it may export.
        return ACTCarry(new.detach(), steps.detach(), halted.detach(), data), out


class ARCLossHead(nn.Module):
    """Same per-example loss scaling as upstream, with finite checks and audit metrics."""
    def __init__(self, model, loss_type='stablemax_cross_entropy'):
        super().__init__()
        if loss_type != 'stablemax_cross_entropy':
            raise ValueError('This matched recipe uses StableMax')
        self.model = model

    def initial_carry(self, *a, **kw):
        return self.model.initial_carry(*a, **kw)

    def forward(self, return_keys=(), **kwargs):
        new, out = self.model(**kwargs)
        labels = new.current_data['labels']
        losses, valid, mask = per_example_loss(out['logits'], labels)
        with torch.no_grad():
            pred = out['logits'].argmax(-1)
            correct = ((pred == labels) | ~mask).all(-1) & valid
            halted_valid = new.halted & valid
            counts = mask.sum(-1).clamp_min(1)
            metrics = dict(count=halted_valid.sum(),
                    accuracy=(((pred == labels) & mask).sum(-1) / counts * halted_valid).sum(),
                    exact_accuracy=(correct & halted_valid).sum(),
                    q_halt_accuracy=(((out['q_halt_logits'] >= 0) == correct) & halted_valid).sum(),
                    steps=(new.steps * halted_valid).sum())
        halt_loss = F.binary_cross_entropy_with_logits(out['q_halt_logits'],
                           correct.to(out['q_halt_logits']), reduction='none')
        lm = losses[valid].sum()
        q = halt_loss[valid].sum()
        # pretrain.py divides the total by global batch size before backward.
        stability = out.get('stability_loss', lm.new_zeros(()))
        total = lm + .5 * q + self.model.arc_cfg.stability_weight * stability * valid.sum()
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite ARC training objective')
        metrics.update(lm_loss=lm.detach(), q_halt_loss=q.detach())
        out['preds'] = pred
        detached = {k: out[k].detach() for k in return_keys if k in out and isinstance(out[k], torch.Tensor)}
        return new, total, metrics, detached, new.halted.all()

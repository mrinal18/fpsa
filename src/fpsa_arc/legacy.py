"""Existing FPSA-R operator with audited solvers, without editing old checkpoints.

This is deliberately a SINGLE joint equilibrium, not answer-conditioned FPSA.
The old model weights are retained; only numerical forward/backward is replaced.
"""
from dataclasses import replace
import math
import torch
from src.fpsa_r import build_model
from .model import ARCConfig, Carry
from .numerics import SolverConfig, solve


class LegacyARC(torch.nn.Module):
    def __init__(self, cfg: ARCConfig, *, fpsa=True):
        super().__init__()
        self.cfg = cfg
        self.base = build_model('fpsa_r' if fpsa else 'deq_block',
                                hidden_size=cfg.hidden_size, num_heads=cfg.num_heads,
                                n_block_layers=cfg.layers, expansion=cfg.expansion,
                                vocab_size=cfg.vocab_size, out_vocab_size=cfg.vocab_size,
                                seq_len=cfg.seq_len + cfg.puzzle_emb_len,
                                puzzle_emb_len=0, spectral_norm=False, conv_type='none',
                                dropout=0., contraction_lambda=0.,
                                max_iter=cfg.max_iter, max_iter_eval=cfg.max_iter_eval)
        self.last_infos = []

    def initial_carry(self, batch_size, *, device=None):
        d, n = self.cfg.hidden_size, self.cfg.seq_len + self.cfg.puzzle_emb_len
        w = self.base.embed_tokens.weight
        zero = torch.zeros(batch_size, n, d, device=device or w.device, dtype=w.dtype)
        return Carry(zero, ())

    def forward_segment(self, tokens, carry=None, task_embedding=None, *, strict=True):
        x = self.base.embed_tokens(tokens.long())
        if self.cfg.puzzle_emb_len:
            if task_embedding is None:
                raise ValueError('Task embeddings must be supplied')
            capacity = self.cfg.puzzle_emb_len * self.cfg.hidden_size
            p = torch.nn.functional.pad(task_embedding.to(x), (0, capacity-task_embedding.shape[1]))
            x = torch.cat([p.reshape(x.shape[0], self.cfg.puzzle_emb_len, -1), x], 1)
        x = math.sqrt(self.cfg.hidden_size) * x
        b, n, d = x.shape
        s0 = self.base.block.init_state(b, n, x.device, x.dtype)
        rows = s0.shape[0]

        def unpack(t):
            return t.reshape(b, rows, n, d).transpose(0, 1)

        def pack(s):
            return s.transpose(0, 1).reshape(b, rows * n, d)

        seq_info = dict(cos_sin=self.base.rotary(n), attn_mask=None,
                        prefix_len=self.cfg.puzzle_emb_len)
        phi = lambda t: pack(self.base.block.joint_step(unpack(t), x, seq_info))
        cap = self.cfg.max_iter if self.training else self.cfg.max_iter_eval
        solver = SolverConfig(max_iter=cap, tol=self.cfg.fp_tol, damping=self.cfg.damping,
                              backward_max_iter=self.cfg.backward_max_iter,
                              backward_tol=self.cfg.backward_tol)
        solved, info = solve(phi, pack(s0), solver, mode=self.cfg.mode, strict=strict)
        z = unpack(solved)[0]
        logits = self.base._readout(z)[:, self.cfg.puzzle_emb_len:]
        self.last_infos = [info]
        return Carry(z.detach(), ()), dict(logits=logits,
                q_halt_logits=self.base.q_head(z.mean(1))[:, 0], infos=[info],
                stability_loss=z.new_zeros(()), nfe=info.nfe,
                mlp_calls=info.nfe * self.cfg.layers, stability_map_calls=0)

    def parameter_report(self, task_embedding_entries=0, task_embedding_dim=0):
        core = sum(p.numel() for p in self.parameters())
        task = task_embedding_entries * task_embedding_dim
        return dict(core_parameters=core, task_embedding_values=task,
                    total_learned_values=core+task)

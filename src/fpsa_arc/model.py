"""Answer-conditioned FPSA: attention equilibria BETWEEN feedforward updates.

The proposed ARC variant is separate from src/fpsa_r. Within a stage, evidence
and the current answer are frozen. Only attention recurs. After solving a stage,
one SwiGLU updates the answer. Detached carries enable supervised refinement
segments without claiming an exact gradient through the entire segment history.
"""
from dataclasses import dataclass, replace, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F
from src.fpsa_r.layers import RotaryEmbedding, apply_rotary, SwiGLU
from .numerics import SolverConfig, SolveInfo, solve


def norm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)


@dataclass
class ARCConfig:
    hidden_size: int = 512
    num_heads: int = 8
    layers: int = 2
    expansion: float = 4.0
    seq_len: int = 900
    vocab_size: int = 12
    puzzle_emb_len: int = 16
    value_mode: str = 'evidence'  # evidence or dual; dual is a distinct ablation
    mode: str = 'implicit'       # implicit, one_step, unroll
    operator: str = 'attention'  # block: same MLP, but inside every map call
    max_iter: int = 48
    max_iter_eval: int = 64
    fp_tol: float = 1e-4
    damping: float = 0.8
    backward_max_iter: int = 64
    backward_tol: float = 1e-5
    refinement_steps: int = 16
    stability_weight: float = 0.0
    stability_target: float = 0.95
    stability_probes: int = 2

    def __post_init__(self):
        if min(self.hidden_size, self.num_heads) < 1:
            raise ValueError('Widths and heads must be positive')
        if self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % 2:
            raise ValueError('hidden_size/num_heads must be even for RoPE')
        if self.value_mode not in {'evidence', 'dual'} or self.mode not in {'implicit', 'one_step', 'unroll'}:
            raise ValueError('Unknown value/gradient mode')
        if self.operator not in {'attention', 'block'}:
            raise ValueError('operator must be attention or block')
        if min(self.layers, self.seq_len, self.refinement_steps, self.max_iter, self.max_iter_eval) < 1:
            raise ValueError('Counts must be positive')
        if self.puzzle_emb_len < 0 or self.stability_weight < 0:
            raise ValueError('Invalid embedding length or penalty')
        if self.stability_probes < 1:
            raise ValueError('stability_probes must be positive')
        if self.mode == 'unroll' and self.max_iter != self.max_iter_eval:
            raise ValueError('Primary BPTT comparison requires identical train/eval depth')


@dataclass
class Carry:
    answer: torch.Tensor
    latent: tuple[torch.Tensor, ...]

    def detach(self):
        return Carry(self.answer.detach(), tuple(z.detach() for z in self.latent))


class AttentionMap(nn.Module):
    """Q/K evolve; evidence V is projected once and held fixed for the solve."""
    def __init__(self, cfg: ARCConfig):
        super().__init__()
        d, h = cfg.hidden_size, cfg.num_heads
        self.h, self.dh = h, d // h
        self.e_heads = h if cfg.value_mode == 'evidence' else max(1, h // 2)
        self.s_heads = h - self.e_heads
        self.q, self.k, self.o = [nn.Linear(d, d, bias=False) for _ in range(3)]
        self.ve = nn.Linear(d, self.e_heads * self.dh, bias=False)
        self.vs = nn.Linear(d, self.s_heads * self.dh, bias=False) if self.s_heads else None
        self.raw_tau = nn.Parameter(torch.full((h,), math.log(math.expm1(.2 - .03))))
        self.raw_gain = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.o.weight, std=.01 * math.sqrt(128 / d))

    def split(self, x, heads):
        return x.view(x.shape[0], x.shape[1], heads, self.dh).transpose(1, 2)

    def prepare(self, evidence):
        return self.split(self.ve(norm(evidence)), self.e_heads)

    def forward(self, r, anchor, values, cos_sin):
        s = norm(anchor + r)
        q = self.split(self.q(s), self.h)
        k = self.split(self.k(s), self.h)
        q, k = apply_rotary(q, k, cos_sin)
        q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
        tau = (.03 + F.softplus(self.raw_tau)).view(1, -1, 1, 1)
        a = torch.softmax((q @ k.transpose(-1, -2)) / tau, dim=-1)
        v = values
        if self.vs is not None:
            v = torch.cat([values, self.split(self.vs(s), self.s_heads)], dim=1)
        read = (a @ v).transpose(1, 2).reshape_as(r)
        # A bounded feedback gain helps initialization; NOT a contraction proof.
        return torch.sigmoid(self.raw_gain) * self.o(read)


class Stage(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = AttentionMap(cfg)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.expansion, spectral=False)


def local_gain_penalty(phi, point, *, target, probes):
    """Exact JVP probe penalty, not a largest-singular-value certificate.

    A finite number of directions can miss unstable modes. This opt-in auxiliary
    is separately costed and evaluated at a detached current state.
    """
    loss, maximum = point.new_zeros(()), point.new_zeros(())
    for _ in range(probes):
        v = torch.randn_like(point)
        v = v / v.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None]
        _, jv = torch.autograd.functional.jvp(phi, point.detach(), v, create_graph=True)
        gain = jv.flatten(1).norm(dim=1)
        loss = loss + torch.relu(gain - target).square().mean() / probes
        maximum = torch.maximum(maximum, gain.max())
    return loss, maximum


class ARCReasoner(nn.Module):
    def __init__(self, cfg: ARCConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        nn.init.normal_(self.embedding.weight, std=1 / math.sqrt(cfg.hidden_size))
        # Upstream task embeddings are zero-initialized and padded across prefix
        # slots. A zero anchor plus cosine Q/K has an ill-conditioned direction
        # at initialization. Fixed, nonzero slot encodings remove that artificial
        # singularity without adding learned capacity or recurring an MLP.
        positions = torch.arange(1, cfg.puzzle_emb_len + 1).float()[:, None]
        channels = torch.arange(cfg.hidden_size).float()[None, :]
        angles = positions / (10000. ** (2 * torch.floor(channels / 2) / cfg.hidden_size))
        prefix_anchor = torch.where((channels.long() % 2) == 0, angles.sin(), angles.cos())
        self.register_buffer('prefix_anchor', prefix_anchor / math.sqrt(cfg.hidden_size))
        self.stages = nn.ModuleList(Stage(cfg) for _ in range(cfg.layers))
        self.rotary = RotaryEmbedding(cfg.hidden_size // cfg.num_heads,
                                      cfg.seq_len + cfg.puzzle_emb_len)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.q_head = nn.Linear(cfg.hidden_size, 1)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.)
        self.last_infos: list[SolveInfo] = []

    def initial_carry(self, batch_size, *, device=None):
        w = self.embedding.weight
        shape = (batch_size, self.cfg.seq_len + self.cfg.puzzle_emb_len, self.cfg.hidden_size)
        zero = torch.zeros(shape, device=device or w.device, dtype=w.dtype)
        return Carry(zero, tuple(zero.clone() for _ in self.stages))

    def encode(self, tokens, task_embedding=None):
        if tokens.ndim != 2 or tokens.shape[1] != self.cfg.seq_len:
            raise ValueError('Tokens must use the configured canvas length')
        x = self.embedding(tokens.long())
        if self.cfg.puzzle_emb_len:
            if task_embedding is None:
                raise ValueError('Task embeddings enabled: caller must provide the actual task IDs/embeddings')
            if task_embedding.shape[0] != x.shape[0] or task_embedding.ndim != 2:
                raise ValueError('Task embeddings must have shape (B, embedding_dim)')
            capacity = self.cfg.puzzle_emb_len * self.cfg.hidden_size
            if task_embedding.shape[1] > capacity:
                raise ValueError('Task embedding exceeds prefix capacity')
            p = F.pad(task_embedding.to(x), (0, capacity - task_embedding.shape[1]))
            prefix = p.reshape(x.shape[0], self.cfg.puzzle_emb_len, -1) + self.prefix_anchor.to(x)
            x = torch.cat([prefix, x], dim=1)
        return math.sqrt(self.cfg.hidden_size) * x

    def forward_segment(self, tokens, carry: Carry | None = None, task_embedding=None,
                        *, strict=True):
        x = self.encode(tokens, task_embedding)
        carry = carry or self.initial_carry(tokens.shape[0], device=tokens.device)
        if carry.answer.shape != x.shape or len(carry.latent) != len(self.stages):
            raise ValueError('Carry is incompatible with this batch/configuration')
        y = carry.answer
        latent, infos = [], []
        penalty = x.new_zeros(())
        gain = x.new_zeros(())
        penalty_calls = 0
        cos_sin = self.rotary(x.shape[1])
        cap = self.cfg.max_iter if self.training else self.cfg.max_iter_eval
        solver = SolverConfig(max_iter=cap, tol=self.cfg.fp_tol, damping=self.cfg.damping,
                              backward_max_iter=self.cfg.backward_max_iter,
                              backward_tol=self.cfg.backward_tol)
        for i, layer in enumerate(self.stages):
            anchor = norm(x + y)  # fixed CURRENT answer within this stage
            values = layer.attention.prepare(x)  # evidence never includes the target

            # Default argument binding is essential: backward executes after the
            # loop and must not accidentally use the final layer/context.
            def phi(r, layer=layer, anchor=anchor, values=values, cos_sin=cos_sin):
                update = layer.attention(r, anchor, values, cos_sin)
                if self.cfg.operator == 'block':
                    update = update + layer.mlp(norm(anchor + update))
                return update

            r, info = solve(phi, carry.latent[i], solver, mode=self.cfg.mode, strict=strict)
            infos.append(info)
            if self.training and self.cfg.stability_weight:
                p, g = local_gain_penalty(phi, r, target=self.cfg.stability_target,
                                         probes=self.cfg.stability_probes)
                penalty, gain = penalty + p, torch.maximum(gain, g)
                penalty_calls += self.cfg.stability_probes
            y = norm(y + r)
            if self.cfg.operator == 'attention':
                y = norm(y + layer.mlp(y))  # exactly ONCE per stage/segment
            latent.append(r)
        self.last_infos = infos
        output = dict(logits=self.lm_head(y[:, self.cfg.puzzle_emb_len:]),
                      q_halt_logits=self.q_head(y[:, 0]).squeeze(-1),
                      infos=infos, stability_loss=penalty,
                      stability_probe_gain=gain,
                      stability_map_calls=penalty_calls,
                      mlp_calls=(len(self.stages) if self.cfg.operator == 'attention'
                                 else sum(v.nfe for v in infos)),
                      nfe=sum(v.nfe for v in infos))
        return Carry(y, tuple(latent)).detach(), output

    def parameter_report(self, task_embedding_entries=0, task_embedding_dim=0):
        core = sum(p.numel() for p in self.parameters())
        task = task_embedding_entries * task_embedding_dim
        return dict(core_parameters=core, task_embedding_values=task,
                    total_learned_values=core + task)

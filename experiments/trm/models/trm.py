"""TRM baseline: Tiny Recursive Model (Jolicoeur-Martineau, 2025).

Faithful port of the reference implementation
(github.com/SamsungSAILMontreal/TinyRecursiveModels, MIT license) so the
Implicit TRM can be compared against it inside one harness.

Recursion per deep-supervision step (arch defaults H_cycles=3, L_cycles=6,
L_layers=2):

    for h in range(H_cycles):            # first H_cycles-1 under no_grad
        for l in range(L_cycles):
            z = net(z, y + x_emb)        # latent update ("z_L")
        y = net(y, z)                    # answer update ("z_H")

Gradients flow through the LAST cycle only (L_cycles+1 = 7 network
applications), which is exactly what the Implicit TRM replaces with an
equilibrium solve + O(1)-memory implicit gradients.

Divergence from the reference (documented): the puzzle embedding is a plain
nn.Parameter prefix instead of CastedSparseEmbedding + sign-SGD. For
Sudoku-Extreme and Maze-Hard there is a single blank puzzle identifier, so the
sparse embedding degenerates to one learned prefix; a separate optimizer
param-group with `puzzle_emb_lr` reproduces the reference behavior.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
    trunc_normal_init_,
)


@dataclass
class TRMConfig:
    batch_size: int
    seq_len: int
    vocab_size: int

    H_cycles: int = 3
    L_cycles: int = 6
    L_layers: int = 2

    hidden_size: int = 512
    expansion: float = 4.0
    num_heads: int = 8
    pos_encodings: str = "rope"          # rope | learned | none
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    puzzle_emb_len: int = 16             # learned prefix tokens (0 disables)

    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    no_ACT_continue: bool = True

    forward_dtype: str = "float32"       # bfloat16 on GPU

    mlp_t: bool = False                  # token-mixing MLP instead of attention

    # --- Implicit TRM extras (ignored by the baseline) ---
    norm_style: str = "post"             # post (TRM) | pre (FPSA/FPRM)
    # Learnable residual scaling (FPRM): h <- a1*h + b1*out at each residual
    # junction and h <- a2*z + b2*injection at the loop input, with betas
    # weight-tied so the map's identity path has gain a2*a1^(2L) < 1. Without
    # this, the residual identity path makes the block map non-contractive and
    # the fixed-point iteration cannot converge.
    residual_scale: bool = False
    alpha_1_init: float = 0.75
    alpha_2_init: float = 0.25
    spectral_norm: bool = False
    damping: float = 1.0                 # solver stepsize
    stepsize_decay: float = 0.9
    decay_patience: int = 5
    inner_tol: float = 1e-3
    inner_max_iter: int = 16
    inner_max_iter_eval: Optional[int] = None
    adjoint_steps: int = 10
    adjoint_tol: float = 1e-4
    grad_mode: str = "neumann"           # neumann | phantom | bptt
    bptt_steps: int = 6                  # for grad_mode=bptt (FPRM-style)
    mask_nonconverged: bool = True
    jacobian_reg_lambda: float = 0.0     # FDA Jacobian penalty weight
    jacobian_eps: float = 1e-3
    n_jacobian_samples: int = 1


@dataclass
class InnerCarry:
    z: torch.Tensor
    y: torch.Tensor


@dataclass
class Carry:
    inner_carry: InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TRMBlock(nn.Module):
    """Attention (or token-mixing MLP) + SwiGLU. norm_style selects post-norm
    (TRM reference) or pre-norm (used by the Implicit TRM for contractivity)."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        if config.mlp_t:
            self.mlp_t = SwiGLU(
                hidden_size=config.seq_len + config.puzzle_emb_len,
                expansion=config.expansion,
                spectral_norm=config.spectral_norm,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                causal=False,
                spectral_norm=config.spectral_norm,
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
            spectral_norm=config.spectral_norm,
        )

    def _mix(self, cos_sin: Optional[CosSin], h: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            return self.mlp_t(h.transpose(1, 2)).transpose(1, 2)
        return self.self_attn(cos_sin=cos_sin, hidden_states=h)

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor,
                alpha_1: Optional[torch.Tensor] = None,
                beta_1: Optional[torch.Tensor] = None) -> torch.Tensor:
        if alpha_1 is None:
            alpha_1 = hidden_states.new_ones(())
            beta_1 = hidden_states.new_ones(())

        if self.config.norm_style == "post":
            # TRM reference: residual then RMS-norm
            hidden_states = rms_norm(
                alpha_1 * hidden_states + beta_1 * self._mix(cos_sin, hidden_states),
                self.norm_eps,
            )
            hidden_states = rms_norm(
                alpha_1 * hidden_states + beta_1 * self.mlp(hidden_states), self.norm_eps
            )
        else:
            # pre-norm keeps the update map's Jacobian well-scaled for the
            # fixed-point solve (FPSA Appx B; FPRM found the same).
            hidden_states = alpha_1 * hidden_states + beta_1 * self._mix(
                cos_sin, rms_norm(hidden_states, self.norm_eps)
            )
            hidden_states = alpha_1 * hidden_states + beta_1 * self.mlp(
                rms_norm(hidden_states, self.norm_eps)
            )
        return hidden_states


class ReasoningModule(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TRMBlock(config) for _ in range(config.L_layers)])

        if config.residual_scale:
            def logit(a: float) -> float:
                return math.log(a / (1 - a))
            self.alpha_1_param = nn.Parameter(
                torch.full((config.hidden_size,), logit(config.alpha_1_init))
            )
            self.alpha_2_param = nn.Parameter(
                torch.full((config.hidden_size,), logit(config.alpha_2_init))
            )
            self.alpha_1_param._no_weight_decay = True
            self.alpha_2_param._no_weight_decay = True

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor,
                cos_sin: Optional[CosSin] = None) -> torch.Tensor:
        if not self.config.residual_scale:
            hidden_states = hidden_states + input_injection
            for layer in self.layers:
                hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
            return hidden_states

        # FPRM residual scaling with asymptotic weight tying: the identity
        # path through the whole module has gain a2 * a1^(2L) < 1, and the
        # betas are tied so repeated application stays bounded.
        dtype = hidden_states.dtype
        alpha_1 = torch.sigmoid(self.alpha_1_param).to(dtype).view(1, 1, -1)
        alpha_2 = torch.sigmoid(self.alpha_2_param).to(dtype).view(1, 1, -1)
        two_l = 2 * len(self.layers)
        beta_2 = 1 - alpha_2 * alpha_1.pow(two_l)
        beta_1 = beta_2 * (1 - alpha_1) / (1 - alpha_1.pow(two_l) + 1e-5)

        hidden_states = alpha_2 * hidden_states + beta_2 * input_injection
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states,
                                  alpha_1=alpha_1, beta_1=beta_1)
        return hidden_states


class TRMInnerBase(nn.Module):
    """Embeddings, heads, and initial states shared by TRM and Implicit TRM."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            config.vocab_size, config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = config.puzzle_emb_len
        if self.puzzle_emb_len > 0:
            # Zero-init learned prefix (see module docstring).
            self.puzzle_emb = nn.Parameter(
                torch.zeros(self.puzzle_emb_len, config.hidden_size)
            )

        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len, config.hidden_size,
                init_std=embed_init_std, cast_to=self.forward_dtype,
            )

        self.z_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.y_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(input.to(torch.int32))
        if self.puzzle_emb_len > 0:
            prefix = self.puzzle_emb.to(self.forward_dtype).unsqueeze(0).expand(
                embedding.shape[0], -1, -1
            )
            embedding = torch.cat((prefix, embedding), dim=-2)
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def _seq_info(self) -> Dict:
        return {"cos_sin": self.rotary_emb() if hasattr(self, "rotary_emb") else None}

    def empty_carry(self, batch_size: int) -> InnerCarry:
        shape = (batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size)
        return InnerCarry(
            z=torch.empty(*shape, dtype=self.forward_dtype),
            y=torch.empty(*shape, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: InnerCarry) -> InnerCarry:
        flag = reset_flag.view(-1, 1, 1)
        return InnerCarry(
            z=torch.where(flag, self.z_init, carry.z),
            y=torch.where(flag, self.y_init, carry.y),
        )

    def _outputs(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.lm_head(y)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(y[:, 0]).to(torch.float32)
        return output, q_logits[..., 0], q_logits[..., 1]


class TRMInner(TRMInnerBase):
    """Reference TRM recursion: fixed cycles, grad through the last one."""

    def __init__(self, config: TRMConfig):
        super().__init__(config)
        self.L_level = ReasoningModule(config)

    def forward(self, carry: InnerCarry, batch: Dict[str, torch.Tensor]):
        seq_info = self._seq_info()
        input_embeddings = self._input_embeddings(batch["inputs"])

        z, y = carry.z, carry.y
        with torch.no_grad():
            for _h in range(self.config.H_cycles - 1):
                for _l in range(self.config.L_cycles):
                    z = self.L_level(z, y + input_embeddings, **seq_info)
                y = self.L_level(y, z, **seq_info)
        # final cycle with grad
        for _l in range(self.config.L_cycles):
            z = self.L_level(z, y + input_embeddings, **seq_info)
        y = self.L_level(y, z, **seq_info)

        new_carry = InnerCarry(z=z.detach(), y=y.detach())
        output, q_halt, q_continue = self._outputs(y)
        stats = {"inner_iters": torch.tensor(float(self.config.H_cycles * self.config.L_cycles)),
                 "converged_frac": torch.tensor(float("nan"))}
        return new_carry, output, (q_halt, q_continue), stats


class ACTWrapper(nn.Module):
    """Deep supervision + Q-learning halting (shared by TRM and Implicit TRM)."""

    def __init__(self, config: TRMConfig, inner: TRMInnerBase):
        super().__init__()
        self.config = config
        self.inner = inner

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> Carry:
        batch_size = batch["inputs"].shape[0]
        return Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry: Carry, batch: Dict[str, torch.Tensor]):
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits), stats = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            **{f"stat_{k}": v for k, v in stats.items()},
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and self.config.halt_max_steps > 1:
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: force some sequences to run at least k steps.
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

        return Carry(new_inner_carry, new_steps, halted, new_current_data), outputs


def build_trm(config: TRMConfig) -> ACTWrapper:
    return ACTWrapper(config, TRMInner(config))

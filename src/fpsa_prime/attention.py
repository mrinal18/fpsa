"""Pure attention-level fixed-point map for FPSA-Prime.

The recurrent state is a residual scratchpad ``R``.  The immutable anchor ``A``
is encoded once.  At every recurrent evaluation,

    S(R) = RMSNorm(A + R)
    Q, K = projections of S(R)

Evidence heads read values from the frozen anchor, while scratch heads read
values from the evolving state.  No MLP, convolution, or full Transformer block
is inside this map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FPSAPrimeConfig
from .layers import CosSin, RMSNorm, apply_rotary, inverse_sigmoid, inverse_softplus


@dataclass(frozen=True)
class FPSAContext:
    anchor: torch.Tensor
    evidence_values: Optional[torch.Tensor]
    cos_sin: Optional[CosSin]
    attention_bias: Optional[torch.Tensor]
    relation_ids: Optional[torch.Tensor]


class DualBankFixedPointAttention(nn.Module):
    """Architectural map ``Phi(R; A)`` used by the equilibrium solver."""

    def __init__(self, cfg: FPSAPrimeConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.num_evidence_heads = cfg.num_evidence_heads
        self.num_scratch_heads = cfg.num_scratch_heads

        self.state_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.W_Q = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.W_K = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.W_O = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.W_V_evidence = (
            nn.Linear(cfg.hidden_size, self.num_evidence_heads * self.head_dim, bias=False)
            if self.num_evidence_heads else None
        )
        self.W_V_scratch = (
            nn.Linear(cfg.hidden_size, self.num_scratch_heads * self.head_dim, bias=False)
            if self.num_scratch_heads else None
        )

        tau_offset = cfg.temperature_init - cfg.temperature_min
        self.raw_temperature = nn.Parameter(
            torch.full((cfg.num_heads,), inverse_softplus(tau_offset))
        )
        gate_values = [cfg.evidence_gate_init] * self.num_evidence_heads
        gate_values += [cfg.scratch_gate_init] * self.num_scratch_heads
        self.head_gate_logits = nn.Parameter(
            torch.tensor([inverse_sigmoid(v) for v in gate_values], dtype=torch.float32)
        )
        self.head_gate_logits._no_weight_decay = True
        self.raw_temperature._no_weight_decay = True

        if cfg.num_relation_types:
            self.relation_bias = nn.Parameter(
                torch.zeros(cfg.num_heads, cfg.num_relation_types)
            )
        else:
            self.register_parameter("relation_bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        if self.W_V_evidence is not None:
            nn.init.xavier_uniform_(self.W_V_evidence.weight)
        if self.W_V_scratch is not None:
            nn.init.xavier_uniform_(self.W_V_scratch.weight)
        # A small output projection makes the initial fixed-point map stable
        # without permanently constraining its spectral norm.
        nn.init.normal_(self.W_O.weight, mean=0.0, std=self.cfg.output_init_std)

    def _split_full(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _split_group(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _merge(x: torch.Tensor) -> torch.Tensor:
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d)

    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature) + self.cfg.temperature_min

    def head_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.head_gate_logits)

    def prepare(
        self,
        anchor: torch.Tensor,
        *,
        cos_sin: Optional[CosSin] = None,
        attention_bias: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
    ) -> FPSAContext:
        """Prepare quantities that are immutable for the entire fixed-point solve."""
        if anchor.ndim != 3 or anchor.shape[-1] != self.hidden_size:
            raise ValueError("anchor must have shape (B, N, hidden_size)")
        if attention_bias is not None:
            attention_bias = attention_bias.to(device=anchor.device)
        if relation_ids is not None:
            relation_ids = relation_ids.to(device=anchor.device)
        evidence_values = None
        if self.W_V_evidence is not None:
            evidence_values = self._split_group(
                self.W_V_evidence(anchor), self.num_evidence_heads
            )
        self._validate_structure(anchor, attention_bias, relation_ids)
        return FPSAContext(
            anchor=anchor,
            evidence_values=evidence_values,
            cos_sin=cos_sin,
            attention_bias=attention_bias,
            relation_ids=relation_ids,
        )

    def _validate_structure(
        self,
        anchor: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
        relation_ids: Optional[torch.Tensor],
    ) -> None:
        b, n, _ = anchor.shape
        if attention_bias is not None:
            if not attention_bias.dtype.is_floating_point:
                raise TypeError("attention_bias must be floating point and additive")
            if attention_bias.ndim not in (2, 3, 4):
                raise ValueError("attention_bias must have 2, 3, or 4 dimensions")
            if attention_bias.shape[-2:] != (n, n):
                raise ValueError("attention_bias has the wrong sequence dimensions")
            if attention_bias.ndim >= 3 and attention_bias.shape[0] not in (1, b):
                raise ValueError("attention_bias batch dimension is not broadcastable")
            if attention_bias.ndim == 4 and attention_bias.shape[1] not in (
                1,
                self.num_heads,
            ):
                raise ValueError("attention_bias head dimension is not broadcastable")
            if bool(torch.isnan(attention_bias).any() or torch.isposinf(attention_bias).any()):
                raise ValueError("attention_bias may contain finite values or -inf, not NaN/+inf")
            blocked = torch.isneginf(self._broadcast_bias(attention_bias))
            if self.cfg.causal:
                causal = torch.ones(n, n, dtype=torch.bool, device=anchor.device).triu(1)
                blocked = blocked | causal.view(1, 1, n, n)
            if bool(blocked.all(dim=-1).any()):
                raise ValueError("attention structure leaves at least one query row fully masked")
        if relation_ids is not None:
            if self.relation_bias is None:
                raise ValueError("relation_ids were supplied but num_relation_types is zero")
            if relation_ids.dtype != torch.long:
                raise TypeError("relation_ids must have dtype torch.long")
            if relation_ids.ndim not in (2, 3):
                raise ValueError("relation_ids must have shape (N,N) or (B,N,N)")
            if relation_ids.shape[-2:] != (n, n):
                raise ValueError("relation_ids has the wrong sequence dimensions")
            if relation_ids.ndim == 3 and relation_ids.shape[0] not in (1, b):
                raise ValueError("relation_ids batch dimension is not broadcastable")
            if relation_ids.numel() and (
                int(relation_ids.min()) < 0
                or int(relation_ids.max()) >= self.cfg.num_relation_types
            ):
                raise ValueError("relation_ids contains an out-of-range relation type")

    def _relation_bias(self, relation_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if relation_ids is None:
            return None
        assert self.relation_bias is not None
        if relation_ids.ndim == 2:
            # (H, R) indexed by (N, N) -> (H, N, N)
            return self.relation_bias[:, relation_ids].unsqueeze(0)
        # Advanced indexing gives (H, B, N, N); move B first.
        return self.relation_bias[:, relation_ids].permute(1, 0, 2, 3)

    @staticmethod
    def _broadcast_bias(bias: torch.Tensor) -> torch.Tensor:
        if bias.ndim == 2:
            return bias.unsqueeze(0).unsqueeze(0)
        if bias.ndim == 3:
            return bias.unsqueeze(1)
        return bias

    def fixed_map(
        self,
        residual: torch.Tensor,
        context: FPSAContext,
        *,
        return_aux: bool = False,
    ):
        """Evaluate ``Phi(R; A)`` once.

        Solver damping is intentionally absent here.  Changing damping must not
        change the fixed-point equation or the implicit Jacobian.
        """
        if residual.shape != context.anchor.shape:
            raise ValueError("residual and anchor must have the same shape")

        state = self.state_norm(context.anchor + residual)
        q = self._split_full(self.W_Q(state))
        k = self._split_full(self.W_K(state))
        q, k = apply_rotary(q, k, context.cos_sin)

        qf, kf = q.float(), k.float()
        if self.cfg.qk_normalize:
            qf = F.normalize(qf, dim=-1, eps=self.cfg.qk_norm_eps)
            kf = F.normalize(kf, dim=-1, eps=self.cfg.qk_norm_eps)
            scores = torch.matmul(qf, kf.transpose(-2, -1))
        else:
            scores = torch.matmul(qf, kf.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores / self.temperature().float().view(1, -1, 1, 1)

        relation_bias = self._relation_bias(context.relation_ids)
        if relation_bias is not None:
            scores = scores + relation_bias.float()
        if context.attention_bias is not None:
            scores = scores + self._broadcast_bias(context.attention_bias).float()
        if self.cfg.causal:
            n = scores.shape[-1]
            causal = torch.ones(n, n, dtype=torch.bool, device=scores.device).triu(1)
            scores = scores.masked_fill(causal, float("-inf"))

        attention = F.softmax(scores, dim=-1).to(state.dtype)
        values = []
        if context.evidence_values is not None:
            values.append(context.evidence_values)
        if self.W_V_scratch is not None:
            values.append(
                self._split_group(self.W_V_scratch(state), self.num_scratch_heads)
            )
        value_tensor = values[0] if len(values) == 1 else torch.cat(values, dim=1)
        retrieved = torch.matmul(attention, value_tensor)
        retrieved = retrieved * self.head_gates().to(retrieved.dtype).view(1, -1, 1, 1)
        update = self.W_O(self._merge(retrieved))

        if not return_aux:
            return update
        entropy = -(attention.float().clamp_min(1e-12).log() * attention.float()).sum(-1)
        return update, {
            "attention": attention,
            "entropy": entropy,
            "temperature": self.temperature(),
            "head_gates": self.head_gates(),
        }

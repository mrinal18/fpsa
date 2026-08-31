"""FPSA-Prime reasoner: one-time encode, attention equilibrium, one-time decode."""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import DualBankFixedPointAttention, FPSAContext
from .config import FPSAPrimeConfig, build_config
from .implicit import solve_equilibrium
from .layers import RMSNorm, RotaryEmbedding, SwiGLU
from .stability import local_jacobian_spectral_penalty


class FPSAPrimeReasoner(nn.Module):
    def __init__(self, cfg: FPSAPrimeConfig):
        super().__init__()
        self.cfg = cfg
        hidden_size = cfg.hidden_size
        self.embed_scale = math.sqrt(hidden_size)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, hidden_size)
        nn.init.trunc_normal_(
            self.embed_tokens.weight, std=1.0 / self.embed_scale
        )

        total_length = cfg.max_seq_len + cfg.num_global_slots
        if cfg.position_encoding in ("learned", "rope+learned"):
            self.position_embedding = nn.Embedding(total_length, hidden_size)
            nn.init.trunc_normal_(
                self.position_embedding.weight, std=1.0 / self.embed_scale
            )
        else:
            self.register_module("position_embedding", None)
        if cfg.position_encoding in ("rope", "rope+learned"):
            self.rotary = RotaryEmbedding(
                cfg.head_dim, total_length, cfg.rope_theta
            )
        else:
            self.register_module("rotary", None)

        if cfg.num_global_slots:
            self.global_slots = nn.Parameter(
                torch.zeros(cfg.num_global_slots, hidden_size)
            )
            nn.init.trunc_normal_(
                self.global_slots, std=1.0 / self.embed_scale
            )
        else:
            self.register_parameter("global_slots", None)

        # These modules execute once per example, outside the equilibrium.
        self.input_norm = RMSNorm(hidden_size, cfg.rms_norm_eps)
        self.input_mlp = (
            SwiGLU(hidden_size, cfg.input_expansion, cfg.dropout)
            if cfg.use_input_mlp
            else None
        )
        self.anchor_norm = RMSNorm(hidden_size, cfg.rms_norm_eps)

        # This is the only recurrent module.
        self.attention = DualBankFixedPointAttention(cfg)

        self.raw_residual_readout_gain = nn.Parameter(torch.tensor(0.0))
        self.raw_residual_readout_gain._no_weight_decay = True
        self.output_norm = RMSNorm(hidden_size, cfg.rms_norm_eps)
        self.output_mlp = (
            SwiGLU(hidden_size, cfg.output_expansion, cfg.dropout)
            if cfg.use_output_mlp
            else None
        )
        self.final_norm = RMSNorm(hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(
            hidden_size, cfg.out_vocab_size or cfg.vocab_size, bias=False
        )
        self.verifier_head = (
            nn.Linear(hidden_size, 1) if cfg.use_verifier_head else None
        )

        self.last_info = None
        self.last_context: Optional[FPSAContext] = None

    @property
    def output_vocab_size(self) -> int:
        return self.cfg.out_vocab_size or self.cfg.vocab_size

    def residual_readout_gain(self) -> torch.Tensor:
        # Range (0, 2), initialized to exactly 1.
        return 2.0 * torch.sigmoid(self.raw_residual_readout_gain)

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (B, N)")
        if tokens.shape[1] > self.cfg.max_seq_len:
            raise ValueError("input exceeds max_seq_len")
        hidden = self.embed_scale * self.embed_tokens(tokens.long())
        if self.global_slots is not None:
            slots = self.global_slots.to(hidden.dtype).unsqueeze(0).expand(
                hidden.shape[0], -1, -1
            )
            hidden = torch.cat([slots, hidden], dim=1)
        if self.position_embedding is not None:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            hidden = hidden + self.position_embedding(positions).to(
                hidden.dtype
            ).unsqueeze(0)
        if self.input_mlp is not None:
            hidden = hidden + self.input_mlp(self.input_norm(hidden))
        return hidden

    def _cos_sin(self, length: int):
        return None if self.rotary is None else self.rotary(length)

    @staticmethod
    def _detached_context(context: FPSAContext) -> FPSAContext:
        evidence = (
            None
            if context.evidence_values is None
            else context.evidence_values.detach()
        )
        return FPSAContext(
            anchor=context.anchor.detach(),
            evidence_values=evidence,
            cos_sin=context.cos_sin,
            attention_bias=(
                None
                if context.attention_bias is None
                else context.attention_bias.detach()
            ),
            relation_ids=context.relation_ids,
        )

    def _prepare_context(
        self,
        encoded: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
        relation_ids: Optional[torch.Tensor],
    ) -> FPSAContext:
        anchor = self.anchor_norm(encoded)
        context = self.attention.prepare(
            anchor,
            cos_sin=self._cos_sin(anchor.shape[1]),
            attention_bias=attention_bias,
            relation_ids=relation_ids,
        )
        # Keep diagnostics available without retaining the training graph after
        # the caller has finished the step.
        self.last_context = self._detached_context(context)
        return context

    def _decode_hidden(
        self, encoded: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        hidden = encoded + self.residual_readout_gain().to(
            encoded.dtype
        ) * residual
        if self.output_mlp is not None:
            hidden = hidden + self.output_mlp(self.output_norm(hidden))
        return self.final_norm(hidden)

    def decode_residual(
        self, encoded: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        hidden = self._decode_hidden(encoded, residual)
        logits = self.lm_head(hidden)
        return logits[:, self.cfg.num_global_slots :]

    def local_stability_penalty(
        self, residual: torch.Tensor, context: FPSAContext
    ) -> dict[str, torch.Tensor]:
        """Measure and softly penalise local recurrent feedback gain."""
        detached_context = self._detached_context(context)
        fixed_map = lambda state: self.attention.fixed_map(state, detached_context)
        return local_jacobian_spectral_penalty(
            fixed_map,
            residual.detach(),
            target=self.cfg.stability_target,
            power_steps=self.cfg.stability_power_steps,
            epsilon=self.cfg.stability_fd_eps,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        attention_bias: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
        initial_residual: Optional[torch.Tensor] = None,
        max_iter: Optional[int] = None,
        record_trace: bool = False,
        return_context: bool = False,
        require_convergence: Optional[bool] = None,
    ) -> dict[str, object]:
        encoded = self._encode(tokens)
        context = self._prepare_context(encoded, attention_bias, relation_ids)
        fixed_map = lambda residual: self.attention.fixed_map(residual, context)

        if initial_residual is None:
            if self.cfg.init_std > 0:
                initial_residual = torch.randn_like(encoded) * self.cfg.init_std
            else:
                initial_residual = torch.zeros_like(encoded)
        else:
            if initial_residual.shape != encoded.shape:
                raise ValueError(
                    "initial_residual must match the encoded state shape"
                )
            initial_residual = initial_residual.to(
                device=encoded.device, dtype=encoded.dtype
            )

        residual, info = solve_equilibrium(
            fixed_map,
            initial_residual,
            self.cfg,
            training=self.training,
            max_iter=max_iter,
            record_trace=record_trace,
            require_convergence=require_convergence,
        )
        self.last_info = info
        hidden = self._decode_hidden(encoded, residual)
        token_hidden = hidden[:, self.cfg.num_global_slots :]
        output: dict[str, object] = {
            "logits": self.lm_head(token_hidden),
            "residual": residual,
            "encoded": encoded,
            "info": info,
        }
        if return_context:
            output["context"] = context
        if self.verifier_head is not None:
            output["learned_energy"] = F.softplus(
                self.verifier_head(hidden.mean(dim=1))
            ).squeeze(-1)
        if self.training and self.cfg.stability_weight > 0:
            stability = self.local_stability_penalty(residual, context)
            output["stability_loss"] = stability["loss"]
            output["stability_estimate"] = stability["estimate"]
            output["stability_max_estimate"] = stability["max_estimate"]
        return output

    @torch.no_grad()
    def forward_particles(
        self,
        tokens: torch.Tensor,
        *,
        num_particles: int,
        init_std: float,
        energy_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        attention_bias: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
        max_iter: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Run independent initial states and select the lowest-energy answer."""
        if self.training:
            raise RuntimeError("particle inference requires model.eval()")
        if num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if init_std < 0:
            raise ValueError("init_std cannot be negative")
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (B, N)")
        batch, length = tokens.shape
        expanded_tokens = tokens[:, None, :].expand(
            batch, num_particles, length
        ).reshape(batch * num_particles, length)

        def expand_structure(
            value: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            if value is None or value.ndim == 2 or value.shape[0] == 1:
                return value
            return value[:, None].expand(
                batch, num_particles, *value.shape[1:]
            ).reshape(batch * num_particles, *value.shape[1:])

        total_length = length + self.cfg.num_global_slots
        initial = torch.randn(
            batch * num_particles,
            total_length,
            self.cfg.hidden_size,
            device=tokens.device,
            dtype=self.embed_tokens.weight.dtype,
        ) * init_std
        output = self.forward(
            expanded_tokens,
            attention_bias=expand_structure(attention_bias),
            relation_ids=expand_structure(relation_ids),
            initial_residual=initial,
            max_iter=max_iter,
            require_convergence=False,
        )
        logits = output["logits"]
        assert isinstance(logits, torch.Tensor)
        logits = logits.view(
            batch, num_particles, length, self.output_vocab_size
        )
        if energy_fn is not None:
            expanded_input = tokens[:, None, :].expand(
                batch, num_particles, length
            )
            energies = energy_fn(
                logits.reshape(
                    batch * num_particles, length, self.output_vocab_size
                ),
                expanded_input.reshape(batch * num_particles, length),
            ).view(batch, num_particles)
        elif "learned_energy" in output:
            learned_energy = output["learned_energy"]
            assert isinstance(learned_energy, torch.Tensor)
            energies = learned_energy.view(batch, num_particles)
        else:
            raise ValueError(
                "particle selection needs energy_fn or a trained verifier head"
            )
        info = output["info"]
        if info.converged is None or info.per_token_residual is None:
            raise RuntimeError("particle solver did not return convergence diagnostics")
        converged = info.converged.view(batch, num_particles)
        sample_residual = info.per_token_residual.amax(dim=1).view(
            batch, num_particles
        )
        finite_logits = torch.isfinite(logits).all(dim=(-1, -2))
        finite_energy = torch.isfinite(energies)
        valid_particles = converged & finite_logits & finite_energy
        selection_energy = energies.masked_fill(~valid_particles, float("inf"))
        no_valid_particle = ~valid_particles.any(dim=1)
        if bool(no_valid_particle.any()):
            indices = no_valid_particle.nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "no finite, converged particle for batch indices "
                f"{indices}; increase max_iter, reduce init_std, or relax fp_tol"
            )
        best = selection_energy.argmin(dim=1)
        batch_index = torch.arange(batch, device=tokens.device)
        selected_logits = logits[batch_index, best]
        return {
            "logits": selected_logits,
            "all_logits": logits,
            "energies": energies,
            "selection_energy": selection_energy,
            "converged": converged,
            "valid_particles": valid_particles,
            "fixed_point_residual": sample_residual,
            "best_particle": best,
        }

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(
    arch: str = "fpsa_prime", **config_overrides
) -> FPSAPrimeReasoner:
    return FPSAPrimeReasoner(build_config(arch, **config_overrides))

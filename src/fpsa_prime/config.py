"""Configuration for FPSA-Prime.

FPSA-Prime has exactly one recurrent state: a residual attention scratchpad.
The token encoder and output decoder run once; only the attention map is
repeated by the fixed-point solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class FPSAPrimeConfig:
    # I/O
    vocab_size: int = 10
    out_vocab_size: int = 0
    max_seq_len: int = 81
    num_global_slots: int = 0

    # Width and one-time encoder/decoder
    hidden_size: int = 128
    num_heads: int = 8
    value_mode: Literal["dual", "evidence", "scratch"] = "dual"
    evidence_fraction: float = 0.5
    input_expansion: float = 2.0
    output_expansion: float = 2.0
    use_input_mlp: bool = True
    use_output_mlp: bool = True
    dropout: float = 0.0

    # Position and structural bias
    position_encoding: Literal["rope", "learned", "rope+learned", "none"] = "rope"
    rope_theta: float = 10000.0
    num_relation_types: int = 0
    causal: bool = False

    # Recurrent attention map Phi(R; A)
    qk_normalize: bool = True
    qk_norm_eps: float = 1e-6
    temperature_init: float = 0.20
    temperature_min: float = 0.03
    evidence_gate_init: float = 0.90
    scratch_gate_init: float = 0.20
    output_init_std: float = 0.02

    # Forward equilibrium solver. Damping belongs to the numerical solver, not
    # the architectural fixed-point equation R = Phi(R; A).
    forward_solver: Literal["picard", "anderson"] = "anderson"
    max_iter: int = 24
    max_iter_eval: int = 64
    fp_tol: float = 1e-4
    solver_damping: float = 0.8
    min_damping: float = 0.05
    damping_decay: float = 0.5
    stall_patience: int = 4
    anderson_m: int = 6
    anderson_beta: float = 1.0
    anderson_lam: float = 1e-4
    init_std: float = 0.0

    # Gradient through the equilibrium
    grad_mode: Literal["implicit", "bptt", "one_step"] = "implicit"
    backward_solver: Literal["gmres", "neumann"] = "gmres"
    backward_max_iter: int = 40
    backward_tol: float = 1e-5
    gmres_restart: int = 20

    # Readout and diagnostics. The learned verifier is opt-in because it must be
    # trained with a task-specific correctness/energy target before it can rank
    # particles meaningfully.
    use_verifier_head: bool = False
    rms_norm_eps: float = 1e-5
    # Implicit differentiation is an equilibrium gradient only after the
    # forward and adjoint systems meet their tolerances.  The safe default is
    # therefore fail-fast; experiments may opt out explicitly for diagnostics.
    require_convergence: bool = True
    require_backward_convergence: bool = True

    def __post_init__(self) -> None:
        valid_options = {
            "value_mode": ({"dual", "evidence", "scratch"}, self.value_mode),
            "position_encoding": (
                {"rope", "learned", "rope+learned", "none"},
                self.position_encoding,
            ),
            "forward_solver": ({"picard", "anderson"}, self.forward_solver),
            "grad_mode": ({"implicit", "bptt", "one_step"}, self.grad_mode),
            "backward_solver": ({"gmres", "neumann"}, self.backward_solver),
        }
        for name, (allowed, value) in valid_options.items():
            if value not in allowed:
                raise ValueError(f"invalid {name}={value!r}; choose from {sorted(allowed)}")
        if self.vocab_size <= 0 or self.out_vocab_size < 0:
            raise ValueError("vocabulary sizes must be positive (or zero for tied output)")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.hidden_size <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.position_encoding in ("rope", "rope+learned"):
            if (self.hidden_size // self.num_heads) % 2:
                raise ValueError("RoPE requires an even head dimension")
        if self.value_mode == "dual" and self.num_heads < 2:
            raise ValueError("dual value mode needs at least two heads")
        if not 0.0 < self.evidence_fraction < 1.0 and self.value_mode == "dual":
            raise ValueError("evidence_fraction must be in (0, 1) for dual mode")
        if self.input_expansion <= 0 or self.output_expansion <= 0:
            raise ValueError("MLP expansion factors must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.qk_norm_eps <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("normalization epsilons must be positive")
        if self.temperature_min <= 0 or self.temperature_init <= self.temperature_min:
            raise ValueError("temperature_init must exceed a positive temperature_min")
        for name, value in (
            ("evidence_gate_init", self.evidence_gate_init),
            ("scratch_gate_init", self.scratch_gate_init),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.output_init_std <= 0:
            raise ValueError("output_init_std must be positive")
        if not 0.0 < self.solver_damping <= 1.0:
            raise ValueError("solver_damping must be in (0, 1]")
        if not 0.0 < self.min_damping <= self.solver_damping:
            raise ValueError("min_damping must be in (0, solver_damping]")
        if not 0.0 < self.damping_decay <= 1.0:
            raise ValueError("damping_decay must be in (0, 1]")
        if self.stall_patience <= 0:
            raise ValueError("stall_patience must be positive")
        if self.max_iter <= 0 or self.max_iter_eval <= 0:
            raise ValueError("iteration budgets must be positive")
        if self.fp_tol <= 0 or self.backward_tol <= 0:
            raise ValueError("solver tolerances must be positive")
        if self.anderson_m <= 0 or self.backward_max_iter <= 0:
            raise ValueError("solver iteration counts must be positive")
        if not 0.0 <= self.anderson_beta <= 1.0:
            raise ValueError("anderson_beta must be in [0, 1]")
        if self.anderson_lam < 0:
            raise ValueError("anderson_lam cannot be negative")
        if self.gmres_restart <= 0:
            raise ValueError("gmres_restart must be positive")
        if self.init_std < 0:
            raise ValueError("init_std cannot be negative")
        if self.num_relation_types < 0 or self.num_global_slots < 0:
            raise ValueError("relation types and global slots cannot be negative")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_evidence_heads(self) -> int:
        if self.value_mode == "evidence":
            return self.num_heads
        if self.value_mode == "scratch":
            return 0
        count = int(round(self.num_heads * self.evidence_fraction))
        return min(self.num_heads - 1, max(1, count))

    @property
    def num_scratch_heads(self) -> int:
        return self.num_heads - self.num_evidence_heads

    def to_dict(self) -> dict:
        return asdict(self)


ARCH_PRESETS = {
    "fpsa_prime": dict(value_mode="dual", grad_mode="implicit"),
    "fpsa_fixed_v": dict(value_mode="evidence", grad_mode="implicit"),
    "fpsa_dynamic_v": dict(value_mode="scratch", grad_mode="implicit"),
    "fpsa_prime_bptt": dict(value_mode="dual", grad_mode="bptt"),
    "fpsa_prime_one_step": dict(value_mode="dual", grad_mode="one_step"),
}


def build_config(arch: str = "fpsa_prime", **overrides) -> FPSAPrimeConfig:
    if arch not in ARCH_PRESETS:
        raise KeyError(f"unknown architecture {arch!r}; choose from {sorted(ARCH_PRESETS)}")
    values = dict(ARCH_PRESETS[arch])
    values.update(overrides)
    return FPSAPrimeConfig(**values)

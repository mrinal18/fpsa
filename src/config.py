from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BertConfig:
    """BERT architecture config, compatible with both vanilla and FPSA attention."""

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0

    # Attention mechanism: "vanilla", "fpsa", or "iterated"
    attention_type: str = "fpsa"

    # ---- Iterated Attention settings (attention_type="iterated") ----
    # Number of attention iterations per layer. T=1 is equivalent to vanilla.
    # Typical values: 2-4. V is computed once from input, Q/K recomputed each step.
    iter_T: int = 3

    # ---- Looped Encoder settings ----
    # When True, wrap the encoder in a fixed-point loop (iterate the whole encoder).
    # Each layer uses vanilla attention; the loop provides adaptive depth.
    # This replaces per-layer FPSA iteration with encoder-level iteration.
    looped_encoder: bool = False
    looped_max_loops: int = 4
    looped_tol: float = 0.01
    looped_conv_exit_frac: float = 0.95
    looped_implicit_grad: bool = True

    # ---- FPSA-specific settings ----
    fpsa_tol: float = 1e-3                  # convergence tolerance (loosened from 1e-4)
    fpsa_max_iter: int = 20                 # max iterations before halt
    fpsa_implicit_grad: bool = True         # phantom gradient for O(1) memory backward
    fpsa_spectral_norm: bool = True         # SN on W_Q, W_K, W_O for contractivity
    fpsa_temperature: float = 1.0           # per-head learnable temperature init
    fpsa_selective_freeze: bool = True      # freeze converged tokens
    fpsa_adjoint_max_iter: int = 4          # Neumann refinement steps (was 32 — too many)
    fpsa_adjoint_tol: float = 1e-4
    fpsa_damping: float = 1.0               # 1.0 = undamped
    fpsa_skip_tol: float = 0.0              # 0.0 = disabled (always run FPI for now)
    fpsa_conv_exit_frac: float = 0.80       # exit when 80% of tokens converge
    
    use_rope: bool = False                  # whether to use Rotary Position Embeddings

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.attention_type not in ("vanilla", "fpsa", "iterated"):
            raise ValueError(f"attention_type must be 'vanilla', 'fpsa', or 'iterated', got {self.attention_type}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

@dataclass
class TrainConfig:
    """Training hyperparameters for experiments."""
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
    grad_clip: float = 1.0
    warmup_steps: int = 100
    log_every: int = 25
    seed: int = 42


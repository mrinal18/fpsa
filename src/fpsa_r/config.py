"""Configuration for FPSA-R (Fixed-Point Self-Attention Reasoner) and baselines."""

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class FPSARConfig:
    # ---- I/O ----
    vocab_size: int = 16
    out_vocab_size: int = 0        # 0 => same as vocab_size
    seq_len: int = 81
    num_puzzle_identifiers: int = 1
    puzzle_emb_len: int = 0          # 0 disables the sparse puzzle embedding

    # ---- width / depth ----
    hidden_size: int = 128
    num_heads: int = 4
    expansion: float = 2.0
    n_block_layers: int = 1          # transformer sub-blocks inside one recurrent block

    # ---- positional ----
    pos_encodings: Literal["rope", "learned", "none"] = "rope"
    rope_theta: float = 10000.0

    # ---- spatial conv (FPRM) ----
    conv_type: Literal["none", "conv1d", "conv2d"] = "none"
    conv_kernel_size: int = 3
    conv_bias: bool = False

    # ---- residual scaling (FPRM signal propagation) ----
    residual_scale: Literal["none", "fixed", "input-independent"] = "input-independent"
    alpha_1_init: float = 0.75
    alpha_2_init: float = 0.25

    # ---- outer fixed-point solver ----
    max_iter: int = 12               # training forward budget
    max_iter_eval: int = 48          # eval budget (test-time compute scaling)
    fp_thresh: float = 1e-3          # relative residual halting threshold
    stepsize: float = 1.0            # damping alpha for the outer Picard step
    stepsize_decay: float = 0.9
    decay_patience: int = 5
    init_std: float = 0.0            # 0.0 => z0 = 0 (deterministic); >0 => random init

    # ---- inner FPSA attention fixed point ----
    fpsa: bool = True                # False => plain single-pass attention in the block
    solver_mode: Literal["joint", "nested"] = "joint"
    inner_max_iter: int = 6          # only used by solver_mode="nested"
    inner_tol: float = 1e-3
    inner_damping: float = 0.8       # alpha in Eq. (4) of the FPSA paper
    spectral_norm: bool = True       # cap ||W_Q||,||W_K||,||W_O|| for contractivity
    mlp_sigma: float = 0.5           # spectral cap on the SwiGLU projections
    attn_temperature: float = 1.0    # learnable per-head tau, initialised here

    # ---- forward solver ----
    # "picard" needs the map to contract. "anderson"/"broyden" do not: they are
    # root-finders, so the contraction requirement belongs to the solver rather
    # than to implicit differentiation itself.
    forward_solver: Literal["picard", "anderson", "broyden"] = "picard"
    broyden_m: int = 8

    # ---- gradient scheme ----
    grad_mode: Literal["implicit", "bptt", "trunc_bptt", "onestep"] = "implicit"
    n_backwards: int = 6             # trunc_bptt: number of with-grad steps (FPRM's n_backwards_L)
    # implicit-diff backward solve
    # "neumann"/"anderson" are stationary iterations and converge only for
    # rho < 1. "gmres" is a Krylov method: it converges whenever (I - J^T) is
    # invertible, at the same cost of one VJP per iteration.
    backward_solver: Literal["anderson", "neumann", "gmres"] = "anderson"
    backward_max_iter: int = 12
    backward_tol: float = 1e-4
    anderson_m: int = 5
    anderson_beta: float = 1.0
    anderson_lam: float = 1e-4
    masked_adjoint: bool = True      # zero the adjoint on tokens that failed to converge
    adjoint_mask_tol_mult: float = 10.0  # token counts as converged if r_tok < mult * fp_thresh
    # Masking is for outliers. If more than this fraction of tokens missed
    # tolerance, the restricted equilibrium problem is meaningless and
    # masking would gut the gradient, so we stop masking instead.
    adjoint_mask_max_frac: float = 0.25

    # ---- halting / ACT ----
    halting: Literal["fixed_point", "act", "none"] = "fixed_point"
    halt_max_steps: int = 4          # ACT supervision segments
    halt_exploration_prob: float = 0.1
    q_loss_coeff: float = 0.5

    # ---- regularisation ----
    dropout: float = 0.0
    # Contraction control: hinge penalty on the estimated spectral radius of
    # the joint map at the solution. This is what keeps rho < 1 as training
    # proceeds -- without it the loop stops being a fixed point at all.
    contraction_lambda: float = 10.0
    contraction_target: float = 0.9
    n_power_iterations: int = 1
    jacobian_eps: float = 1e-3

    # ---- misc ----
    rms_norm_eps: float = 1e-5
    causal: bool = False
    forward_dtype: str = "float32"

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Named presets: one config object drives every architecture in the study, so
# comparisons are parameter- and width-matched by construction.
# ---------------------------------------------------------------------------

ARCH_PRESETS = {
    # --- ours ---
    "fpsa_r": dict(fpsa=True, solver_mode="joint", grad_mode="implicit"),
    "fpsa_r_nested": dict(fpsa=True, solver_mode="nested", grad_mode="implicit"),
    # --- ablations of ours ---
    "fpsa_r_bptt": dict(fpsa=True, solver_mode="joint", grad_mode="bptt"),
    "fpsa_r_onestep": dict(fpsa=True, solver_mode="joint", grad_mode="onestep"),
    "fpsa_r_nomask": dict(fpsa=True, solver_mode="joint", grad_mode="implicit",
                          masked_adjoint=False),
    "fpsa_r_neumann": dict(fpsa=True, solver_mode="joint", grad_mode="implicit",
                           backward_solver="neumann"),
    "fpsa_r_nospec": dict(fpsa=True, solver_mode="joint", grad_mode="implicit",
                          spectral_norm=False),
    # --- relaxing the contraction requirement (see docs/contraction.md) ---
    # Krylov backward: removes rho < 1 from the adjoint solve.
    "deq_gmres": dict(fpsa=False, grad_mode="implicit", backward_solver="gmres"),
    # Root-finding forward + Krylov backward: removes it from both directions.
    "deq_broyden": dict(fpsa=False, grad_mode="implicit", forward_solver="broyden",
                        backward_solver="gmres"),
    "deq_anderson_fwd": dict(fpsa=False, grad_mode="implicit",
                             forward_solver="anderson", backward_solver="gmres"),
    # Same, with the per-layer spectral caps removed: the capacity the caps cost
    # is only worth paying if the solvers actually need it.
    "deq_free": dict(fpsa=False, grad_mode="implicit", forward_solver="broyden",
                     backward_solver="gmres", spectral_norm=False,
                     contraction_target=1.0),
    "deq_free_anderson": dict(fpsa=False, grad_mode="implicit",
                              forward_solver="anderson", backward_solver="gmres",
                              spectral_norm=False, contraction_target=1.0),

    # Does the in-layer attention loop help once the caps are gone?
    "fpsa_free_anderson": dict(fpsa=True, solver_mode="joint", grad_mode="implicit",
                               forward_solver="anderson", backward_solver="gmres",
                               spectral_norm=False, contraction_target=1.0),

    # --- baselines ---
    "deq_block": dict(fpsa=False, grad_mode="implicit"),      # implicit diff, no in-layer FPSA
    "fprm": dict(fpsa=False, grad_mode="trunc_bptt"),         # FPRM: damped FP + truncated BPTT
    "looped_bptt": dict(fpsa=False, grad_mode="bptt"),        # looped transformer, full BPTT
    "ut_act": dict(fpsa=False, grad_mode="bptt", halting="act"),  # Universal Transformer + ACT
    "transformer": dict(fpsa=False, grad_mode="bptt", max_iter=1, max_iter_eval=1,
                        halting="none"),                       # non-recursive, depth-matched
}


def build_config(arch: str, **overrides) -> FPSARConfig:
    if arch not in ARCH_PRESETS:
        raise KeyError(f"unknown arch {arch!r}; choose from {sorted(ARCH_PRESETS)}")
    kw = dict(ARCH_PRESETS[arch])
    kw.update(overrides)
    return FPSARConfig(**kw)

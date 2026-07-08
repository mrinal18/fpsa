"""Implicit TRM: TRM whose latent recursion is an equilibrium solve trained
with O(1)-memory implicit differentiation.

Where TRM runs a fixed number of latent updates and backpropagates through the
last cycle (7 network applications in the graph), the Implicit TRM treats the
latent update as a fixed-point problem

    z* = f(z*),     f(z) = L_level(z, y + x_emb),

solves it by damped iteration UNDER no_grad (any depth, no activation
storage), and attaches gradients at the solution via one differentiable
application of f refined by a truncated Neumann adjoint solve — the FPSA
paper's implicit-differentiation scheme (Appendices B-D), generalized from
attention-only maps to the full TRM block.

Key correctness detail: the solver damps the update, z <- (1-s) z + s f(z).
The gradient closure passed to the adjoint is the SAME damped map f_s. Since
(I - J_{f_s}) = S (I - J_f) with S = diag(s), the Neumann solve on f_s times
the phantom step through f_s yields exactly g^T (I - J_f)^{-1} df/dtheta — the
true implicit gradient — while converging whenever the damped forward
iteration does (spectral radius of J_{f_s} < 1), which is strictly weaker than
requiring rho(J_f) < 1.

Per deep-supervision step:

    for h in range(H_cycles):
        z  = implicit_solve(f_h, z)     # equilibrium, implicit grads
        y  = L_level(y, z)              # one explicit step, exact grads

All H_cycles stay in the graph (each contributes only ~2 applications), unlike
TRM which must discard all but the last cycle.

Halting: inner loop halts on the per-sample fixed-point residual (adaptive
compute per puzzle); the outer deep-supervision loop keeps TRM's ACT Q-head.

Contractivity levers (config): pre-norm blocks (default), solver damping with
patience-based step-size decay (FPRM's FPOPT), optional hard spectral norm on
all linear maps, optional finite-difference Jacobian regularization at z*.
"""

import math
from typing import Dict, Optional, Tuple

import os
import sys

import torch
from torch import nn

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.implicit import attach_implicit_grad, fixed_point_solve

from .trm import ACTWrapper, InnerCarry, ReasoningModule, TRMConfig, TRMInnerBase


class ImplicitTRMInner(TRMInnerBase):
    def __init__(self, config: TRMConfig):
        super().__init__(config)
        self.L_level = ReasoningModule(config)
        if not (config.residual_scale or config.spectral_norm):
            import warnings
            warnings.warn(
                "ImplicitTRM without residual_scale or spectral_norm: the "
                "residual identity path makes the update map non-contractive, "
                "so the fixed-point solve is unlikely to converge and the "
                "Neumann adjoint may diverge. Enable residual_scale=True "
                "(default in the itrm configs) unless this is a deliberate "
                "ablation.",
                stacklevel=2,
            )

    # ----- the latent update map -----
    def _make_f(self, y: torch.Tensor, input_embeddings: torch.Tensor, seq_info: Dict):
        def f(z: torch.Tensor) -> torch.Tensor:
            return self.L_level(z, y + input_embeddings, **seq_info)
        return f

    @staticmethod
    def _damped(f, stepsize: torch.Tensor):
        """Damped map f_s(z) = (1-s) z + s f(z) with per-sample stepsize."""
        def f_s(z: torch.Tensor) -> torch.Tensor:
            s = stepsize.view(-1, *([1] * (z.dim() - 1))).to(z.dtype)
            return (1.0 - s) * z + s * f(z)
        return f_s

    def _jacobian_reg(self, f, z_star: torch.Tensor) -> torch.Tensor:
        """FDA estimate of ||J_f(z*) v||^2 (FPRM-style contractivity penalty).

        Computed in float32: bf16 resolution (~2^-8 relative) is coarser than
        the jacobian_eps=1e-3 probe, so a bf16 finite difference would measure
        quantization noise amplified by 1/(2*eps) instead of the Jacobian.
        The blocks cast weights to the input dtype (CastedLinear), so feeding
        fp32 inputs runs the probe in fp32 end to end.
        """
        cfg = self.config
        if not self.training or cfg.jacobian_reg_lambda == 0.0 or cfg.n_jacobian_samples == 0:
            return z_star.new_zeros((), dtype=torch.float32)
        z0 = z_star.detach().to(torch.float32)
        estimate = z0.new_zeros(())
        for _ in range(cfg.n_jacobian_samples):
            v = (torch.randn_like(z0) / math.sqrt(z0.shape[-1])).detach()
            z_p = f(z0 + cfg.jacobian_eps * v).to(torch.float32)
            z_m = f(z0 - cfg.jacobian_eps * v).to(torch.float32)
            jvp = (z_p - z_m) / (2 * cfg.jacobian_eps)
            estimate = estimate + jvp.pow(2).mean() / cfg.n_jacobian_samples
        return estimate

    def _solve_latent(self, f, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        cfg = self.config
        max_iter = cfg.inner_max_iter
        if (not self.training) and cfg.inner_max_iter_eval is not None:
            max_iter = cfg.inner_max_iter_eval

        if self.training and cfg.grad_mode == "bptt":
            # Truncated-BPTT ablation (FPRM-style): no-grad prefix, then
            # bptt_steps differentiable iterations.
            with torch.no_grad():
                z_pre, info = fixed_point_solve(
                    f, z, tol=cfg.inner_tol,
                    max_iter=max(0, max_iter - cfg.bptt_steps),
                    stepsize=cfg.damping, stepsize_decay=cfg.stepsize_decay,
                    decay_patience=cfg.decay_patience,
                ) if max_iter > cfg.bptt_steps else (z, None)
            z_k = z_pre.detach()
            s = cfg.damping
            for _ in range(min(cfg.bptt_steps, max_iter)):
                z_k = (1.0 - s) * z_k + s * f(z_k)
            stats = self._loop_stats(info, z_k)
            return z_k, stats

        with torch.no_grad():
            z_star, info = fixed_point_solve(
                f, z, tol=cfg.inner_tol, max_iter=max_iter,
                stepsize=cfg.damping, stepsize_decay=cfg.stepsize_decay,
                decay_patience=cfg.decay_patience,
            )
        z_star = z_star.detach()

        if not (self.training and torch.is_grad_enabled()):
            return z_star, self._loop_stats(info, z_star)

        # Masked implicit differentiation: exclude non-converged tokens from
        # the adjoint solve (attach_implicit_grad drops the mask itself when
        # convergence is still rare, to avoid gradient starvation).
        conv_mask = info["converged"] if self.config.mask_nonconverged else None

        f_s = self._damped(f, info["stepsize"])
        adjoint_steps = 0 if cfg.grad_mode == "phantom" else cfg.adjoint_steps
        z_out = attach_implicit_grad(
            z_star, f_s,
            adjoint_steps=adjoint_steps,
            adjoint_tol=cfg.adjoint_tol,
            conv_mask=conv_mask,
        )
        return z_out, self._loop_stats(info, z_out)

    @staticmethod
    def _loop_stats(info: Optional[Dict], z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if info is None:
            return {"inner_iters": z.new_zeros(()), "converged_frac": z.new_zeros(())}
        return {
            "inner_iters": info["steps"].float(),
            "converged_frac": info["converged"].float().mean(),
        }

    def forward(self, carry: InnerCarry, batch: Dict[str, torch.Tensor]):
        seq_info = self._seq_info()
        input_embeddings = self._input_embeddings(batch["inputs"])

        z, y = carry.z, carry.y
        iters_total = None
        conv_last = None
        jac_loss = input_embeddings.new_zeros(())

        for _h in range(self.config.H_cycles):
            f = self._make_f(y, input_embeddings, seq_info)
            z, stats = self._solve_latent(f, z)
            iters_total = stats["inner_iters"] if iters_total is None else iters_total + stats["inner_iters"]
            conv_last = stats["converged_frac"]
            y = self.L_level(y, z, **seq_info)   # explicit answer step, exact grad

            if self.config.jacobian_reg_lambda > 0:
                jac_loss = jac_loss + self._jacobian_reg(f, z)

        new_carry = InnerCarry(z=z.detach(), y=y.detach())
        output, q_halt, q_continue = self._outputs(y)
        stats = {
            "inner_iters": iters_total.detach(),
            "converged_frac": conv_last.detach(),
        }
        if self.config.jacobian_reg_lambda > 0:
            # Pre-weighted and differentiable; the loss head adds any
            # "jacobian_loss" stat it receives, so the lambda lives here only.
            stats["jacobian_loss"] = self.config.jacobian_reg_lambda * jac_loss
        return new_carry, output, (q_halt, q_continue), stats


def build_implicit_trm(config: TRMConfig) -> ACTWrapper:
    return ACTWrapper(config, ImplicitTRMInner(config))

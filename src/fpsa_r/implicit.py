"""Implicit differentiation of the joint (outer x in-layer) equilibrium.

Given the joint map ``G`` of ``block.py`` and its fixed point ``s*``, the
implicit function theorem gives

    dL/dtheta = (dL/ds*) (I - J_G)^{-1} dG/dtheta ,     J_G = dG/ds |_{s*}

so the backward pass needs only the *adjoint* ``lambda`` solving
``(I - J_G^T) lambda = dL/ds*``, which is itself a linear fixed point driven by
one VJP through a single application of ``G``. Nothing from the forward loop has
to be stored: the activation memory is that of one step, independent of how many
iterations the forward solver took. That is the difference from FPRM, which
backpropagates through ``n_backwards_L`` unrolled steps and therefore pays
memory linear in the truncation depth.

Two refinements over the textbook DEQ backward:

* **Anderson-accelerated adjoint.** The linear solve is run with Anderson mixing
  rather than a Neumann series, which matters because a useful reasoner sits at
  a contraction factor close to 1, exactly where the Neumann series crawls.
* **Masked adjoint.** Tokens whose forward residual never fell below tolerance
  are removed from the linear solve. By the lemma in Appendix B.2 this returns
  the exact gradient of the equilibrium problem restricted to the converged
  coordinates, instead of an arbitrarily wrong gradient at a point that is not
  a fixed point.
"""

from typing import Callable, Optional

import torch

from .solvers import anderson_solve, neumann_solve, picard_solve

# Populated on every backward pass; the trainer logs it.
BACKWARD_STATS = {"iters": 0, "rel": 0.0, "calls": 0, "masked_frac": 0.0}


class _JointAdjoint(torch.autograd.Function):
    """Identity in the forward direction; replaces the incoming gradient with
    the solution of ``(I - J_G^T) lambda = g`` in the backward direction."""

    @staticmethod
    def forward(ctx, s_next, s_star, mask, step_fn, solver, max_iter, tol,
                anderson_m, anderson_beta, anderson_lam):
        ctx.save_for_backward(s_star, mask if mask is not None else torch.empty(0))
        ctx.step_fn = step_fn
        ctx.solver = solver
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.anderson = (anderson_m, anderson_beta, anderson_lam)
        ctx.has_mask = mask is not None
        return s_next

    @staticmethod
    def backward(ctx, grad):
        s_star, mask = ctx.saved_tensors
        step_fn = ctx.step_fn

        # Rebuild a one-step graph at s* (the graph of the caller's own single
        # step is being consumed by this very backward pass, so we cannot reuse
        # it). This costs one extra forward and keeps memory at one step.
        with torch.enable_grad():
            s_var = s_star.detach().requires_grad_(True)
            s_out = step_fn(s_var)

        def vjp(lam):
            return torch.autograd.grad(s_out, s_var, lam, retain_graph=True)[0]

        m = None
        if ctx.has_mask and mask.numel() > 0:
            m = mask.to(grad.dtype).view(1, mask.shape[0], mask.shape[1], 1)
            grad = grad * m

        if ctx.solver == "neumann":
            lam, n, rel = neumann_solve(vjp, grad, ctx.max_iter, ctx.tol, mask=m)
        else:
            am, ab, al = ctx.anderson
            lam, n, rel = anderson_solve(vjp, grad, ctx.max_iter, ctx.tol,
                                         m=am, beta=ab, lam_reg=al, mask=m)

        BACKWARD_STATS["iters"] = n
        BACKWARD_STATS["rel"] = rel
        BACKWARD_STATS["calls"] += 1
        BACKWARD_STATS["masked_frac"] = 0.0 if m is None else float(1.0 - m.mean())

        if not torch.isfinite(lam).all():      # never let a bad solve poison training
            lam = grad

        return lam, None, None, None, None, None, None, None, None, None


def solve_equilibrium(step_fn: Callable[[torch.Tensor], torch.Tensor],
                      s0: torch.Tensor,
                      cfg,
                      training: bool,
                      max_iter: Optional[int] = None,
                      record_trace: bool = False):
    """Run the forward solver and attach the requested gradient scheme.

    Returns ``(s_star, info)``. ``s_star`` carries the correct autograd graph for
    ``cfg.grad_mode``:

    ``implicit``     O(1) memory, exact-at-equilibrium gradient (FPSA-R)
    ``onestep``      O(1) memory, one-term phantom gradient (FPSA paper default)
    ``trunc_bptt``   O(K) memory, K unrolled steps (FPRM)
    ``bptt``         O(T) memory, full unroll (looped transformer / UT)
    """
    max_iter = cfg.max_iter if max_iter is None else max_iter
    mode = cfg.grad_mode if training else "eval"

    if mode in ("bptt", "trunc_bptt"):
        n_grad = max_iter if mode == "bptt" else min(cfg.n_backwards, max_iter)
        n_nograd = max_iter - n_grad
        s = s0
        info = None
        if n_nograd > 0:
            s, info = picard_solve(step_fn, s, n_nograd, cfg.fp_thresh,
                                   cfg.stepsize, cfg.stepsize_decay,
                                   cfg.decay_patience, record_trace,
                                   token_tol=cfg.fp_thresh * cfg.adjoint_mask_tol_mult)
            s = s.detach()
        trace = list(info.residual_trace) if info is not None else []
        r = info.rel_residual if info is not None else float("inf")
        for _ in range(n_grad):
            s_new = step_fn(s)
            with torch.no_grad():
                r = float(((s_new - s).norm(dim=-1)
                           / s.norm(dim=-1).clamp_min(1e-8)).amax())
                if record_trace:
                    trace.append(r)
            s = s + cfg.stepsize * (s_new - s)
        from .solvers import SolverInfo
        out_info = SolverInfo(n_iters=max_iter, rel_residual=r,
                              residual_trace=trace, converged_frac=1.0)
        return s, out_info

    # --- equilibrium modes: solve without a graph, then attach one -----------
    with torch.no_grad():
        s_star, info = picard_solve(
            step_fn, s0, max_iter, cfg.fp_thresh, cfg.stepsize,
            cfg.stepsize_decay, cfg.decay_patience, record_trace,
            token_tol=cfg.fp_thresh * cfg.adjoint_mask_tol_mult)
    s_star = s_star.detach()

    if not (training and torch.is_grad_enabled()):
        return s_star, info

    s_next = step_fn(s_star)      # single differentiable application of G

    if mode == "onestep":
        return s_next, info

    mask = None
    if cfg.masked_adjoint and info.token_converged is not None:
        unconverged = 1.0 - float(info.token_converged.float().mean())
        # Masking is an *outlier* mechanism: dropping the handful of coordinates
        # that missed tolerance gives the exact gradient of the equilibrium
        # problem restricted to the rest, and the bias is negligible precisely
        # because the omitted set is tiny (<0.05% in the FPSA paper). When the
        # omitted set is the majority -- which is what happens whenever the
        # forward budget is short relative to the contraction factor -- the
        # restricted problem is not a useful object, and masking silently
        # attenuates the block gradient toward zero. Past the threshold we stop
        # masking rather than train on almost no gradient.
        if unconverged > cfg.adjoint_mask_max_frac or not bool(
                info.token_converged.any()):
            mask = None
        elif bool(info.token_converged.all()):
            mask = None                      # nothing to mask; skip the multiply
        else:
            mask = info.token_converged

    s_out = _JointAdjoint.apply(
        s_next, s_star, mask, step_fn, cfg.backward_solver,
        cfg.backward_max_iter, cfg.backward_tol,
        cfg.anderson_m, cfg.anderson_beta, cfg.anderson_lam)
    return s_out, info

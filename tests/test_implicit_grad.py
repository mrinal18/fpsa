"""Correctness tests for Neumann-series implicit differentiation.

Ground truth on a small model: the exact implicit-function-theorem gradient,
computed with a dense Jacobian solve  lambda = (I - J^T)^{-1} g.  We check

  1. exact-IFT vs fully unrolled backprop agree (both differentiate the same
     converged fixed point),
  2. the Neumann adjoint converges to the exact-IFT gradient as the number of
     adjoint steps grows, and beats the 1-step phantom gradient,
  3. masked implicit differentiation with an all-ones mask equals unmasked.

Run:  python -m pytest tests/test_implicit_grad.py -q
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "sudoku_ablation"))

from src.implicit import attach_implicit_grad, fixed_point_solve, _NeumannImplicitGrad
from fpsa_spatial import SpatialFPSAAttention


B, N, D, H = 2, 16, 32, 4


def _make_attention(seed: int = 0) -> SpatialFPSAAttention:
    torch.manual_seed(seed)
    attn = SpatialFPSAAttention(
        d_model=D,
        num_heads=H,
        value_mode="fixed",
        damping=0.5,
        epsilon=1e-3,
        k_max=16,
        use_spectral_norm=True,
        dropout=0.0,
        max_grid_size=4,
    )
    attn.eval()
    return attn


def _make_inputs(seed: int = 1):
    torch.manual_seed(seed)
    x = torch.randn(B, N, D, dtype=torch.float64)
    row_ids = torch.arange(4).repeat_interleave(4).unsqueeze(0).expand(B, -1)
    col_ids = torch.arange(4).repeat(4).unsqueeze(0).expand(B, -1)
    return x, row_ids, col_ids


def _setup(seed: int = 0):
    """Return (attn, f, x, z_star, G): converged state + a fixed loss weight."""
    attn = _make_attention(seed).double()
    x, row_ids, col_ids = _make_inputs(seed + 1)

    # Values are recomputed inside the closure (V doesn't depend on z, so the
    # map is identical) so every call builds a fresh graph — the tests run
    # several independent backward passes through f.
    def f(z):
        v = attn._get_values(x, x)
        return attn._one_step(z, x, v, row_ids, col_ids)

    with torch.no_grad():
        z = x.clone()
        for _ in range(400):
            z_new = f(z)
            if (z_new - z).norm() / (z.norm() + 1e-12) < 1e-13:
                z = z_new
                break
            z = z_new
        z_star = z.detach()

    # Fixed-point residual must be tiny for gradient comparisons to be valid.
    res = (f(z_star) - z_star).norm() / z_star.norm()
    assert res < 1e-10, f"forward iteration did not converge: rel residual {res:.2e}"

    torch.manual_seed(seed + 2)
    G = torch.randn_like(z_star)
    return attn, f, x, z_star, G


def _param_grads(attn, loss):
    grads = torch.autograd.grad(
        loss, [p for p in attn.parameters() if p.requires_grad], allow_unused=True
    )
    return [g if g is not None else torch.zeros(1, dtype=torch.float64) for g in grads]


def _flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def _exact_ift_grads(attn, f, z_star, G):
    """dL/dtheta via dense IFT solve: lambda = (I - J^T)^{-1} G."""
    z_var = z_star.detach().requires_grad_(True)

    def f_flat(zf):
        return f(zf.view_as(z_star)).reshape(-1)

    J = torch.autograd.functional.jacobian(f_flat, z_var.reshape(-1), vectorize=True)
    n = J.shape[0]
    lam = torch.linalg.solve(torch.eye(n, dtype=J.dtype) - J.T, G.reshape(-1))
    lam = lam.view_as(z_star)

    z_var2 = z_star.detach().requires_grad_(True)
    out = f(z_var2)
    loss_surrogate = (out * lam.detach()).sum()
    return _param_grads(attn, loss_surrogate), lam


def _unrolled_grads(attn, f, x, G, iters=400):
    z = x.clone().detach()
    for _ in range(iters):
        z = f(z)
    loss = (z * G).sum()
    return _param_grads(attn, loss)


def _neumann_grads(attn, f, z_star, G, steps, mask=None):
    z_out = attach_implicit_grad(
        z_star, f, adjoint_steps=steps, adjoint_tol=0.0, conv_mask=mask
    )
    loss = (z_out * G).sum()
    return _param_grads(attn, loss)


def _rel_err(a, b):
    fa, fb = _flat(a), _flat(b)
    return ((fa - fb).norm() / (fb.norm() + 1e-12)).item()


class TestImplicitGradient:
    def test_exact_ift_matches_unrolled(self):
        attn, f, x, z_star, G = _setup()
        g_exact, _ = _exact_ift_grads(attn, f, z_star, G)
        g_unrolled = _unrolled_grads(attn, f, x, G)
        err = _rel_err(g_unrolled, g_exact)
        assert err < 1e-6, f"unrolled vs exact IFT rel err {err:.2e}"

    def test_neumann_converges_to_exact(self):
        attn, f, x, z_star, G = _setup()
        g_exact, _ = _exact_ift_grads(attn, f, z_star, G)

        errs = {}
        for steps in (0, 2, 5, 15, 50):
            g_neu = _neumann_grads(attn, f, z_star, G, steps)
            errs[steps] = _rel_err(g_neu, g_exact)

        # Monotone improvement and near-exact at 50 terms.
        assert errs[50] < 1e-6, f"Neumann(50) rel err {errs[50]:.2e}: {errs}"
        assert errs[15] < 5e-3, f"Neumann(15) rel err {errs[15]:.2e}: {errs}"
        assert errs[0] > errs[5] > errs[15], f"no monotone improvement: {errs}"

    def test_phantom_is_biased_but_directionally_correct(self):
        attn, f, x, z_star, G = _setup()
        g_exact, _ = _exact_ift_grads(attn, f, z_star, G)
        g_phantom = _neumann_grads(attn, f, z_star, G, steps=0)
        cos = torch.nn.functional.cosine_similarity(
            _flat(g_phantom), _flat(g_exact), dim=0
        ).item()
        assert cos > 0.7, f"phantom gradient not aligned with exact: cos {cos:.3f}"

    def test_masked_all_ones_equals_unmasked(self):
        attn, f, x, z_star, G = _setup()
        mask = torch.ones(B, N, dtype=torch.float64)
        g_masked = _neumann_grads(attn, f, z_star, G, steps=10, mask=mask)
        g_plain = _neumann_grads(attn, f, z_star, G, steps=10)
        err = _rel_err(g_masked, g_plain)
        assert err < 1e-12, f"all-ones mask changed the gradient: {err:.2e}"

    def test_input_gradient_matches_exact(self):
        """dL/dx must also follow the IFT formula through the f closure."""
        attn, f0, x, z_star, G = _setup()

        x_var = x.detach().requires_grad_(True)
        _, row_ids, col_ids = _make_inputs(1)

        def f(z):
            v = attn._get_values(x_var, x_var)
            return attn._one_step(z, x_var, v, row_ids, col_ids)

        # Exact lambda from the dense solve (f w.r.t. z only).
        _, lam = _exact_ift_grads(attn, f0, z_star, G)
        out = f(z_star)
        (gx_exact,) = torch.autograd.grad((out * lam.detach()).sum(), x_var, retain_graph=True)

        z_out = attach_implicit_grad(z_star, f, adjoint_steps=50, adjoint_tol=0.0)
        (gx_neu,) = torch.autograd.grad((z_out * G).sum(), x_var)

        err = ((gx_neu - gx_exact).norm() / gx_exact.norm()).item()
        assert err < 1e-6, f"input gradient rel err {err:.2e}"


class TestFixedPointSolve:
    def test_solver_converges_and_reports(self):
        attn, f, x, z_star, _ = _setup()
        z, info = fixed_point_solve(f, x, tol=1e-6, max_iter=200)
        assert bool(info["converged"].all())
        assert (z - z_star).norm() / z_star.norm() < 1e-4

    def test_damping_preserves_fixed_point(self):
        attn, f, x, z_star, _ = _setup()
        # One damped step from the fixed point must stay at the fixed point.
        s = 0.3
        z_next = s * f(z_star) + (1 - s) * z_star
        assert (z_next - z_star).norm() / z_star.norm() < 1e-9

    def test_patience_decay_reduces_stepsize_on_stall(self):
        # A map with a period-2 oscillation: f(z) = -z + c. Plain iteration
        # oscillates forever; patience-based damping must converge it.
        c = torch.ones(1, 4, 8, dtype=torch.float64)
        f = lambda z: -z + c
        z0 = torch.zeros(1, 4, 8, dtype=torch.float64)
        z, info = fixed_point_solve(
            f, z0, tol=1e-6, max_iter=500,
            stepsize=1.0, stepsize_decay=0.7, decay_patience=3,
        )
        target = c / 2
        assert (z - target).norm() < 1e-4, f"oscillating map not converged: {info['sample_residual']}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

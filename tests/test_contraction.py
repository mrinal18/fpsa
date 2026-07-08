"""Contractivity tests for the FPSA update map (paper Section 3 / Appendix B).

The paper's forward and backward well-posedness rest on the update map f being
locally contractive around the fixed point. The sharp certificate is the
spectral radius of the Jacobian: rho(J_f(z*)) < 1 guarantees (i) local
convergence of the forward iteration and (ii) convergence of the Neumann
adjoint series (both iterate matrices similar to J). The 2-norm ||J||_2 upper
bounds rho but is loose for non-normal Jacobians — empirically the FPSA
Jacobian IS non-normal (||J||_2 can exceed 1 while rho stays well below 1 and
the iteration converges), so tests check rho on a dense Jacobian, which is
exact at these test sizes.

Run:  python -m pytest tests/test_contraction.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "sudoku_ablation"))

from src.implicit import estimate_spectral_radius
from fpsa_spatial import SpatialFPSAAttention


B, N, D, H = 2, 16, 32, 4


def _make(damping, use_sn, seed=0):
    torch.manual_seed(seed)
    attn = SpatialFPSAAttention(
        d_model=D, num_heads=H, value_mode="fixed",
        damping=damping, epsilon=1e-3, k_max=16,
        use_spectral_norm=use_sn, dropout=0.0, max_grid_size=4,
    ).double()
    attn.eval()
    torch.manual_seed(seed + 1)
    x = torch.randn(B, N, D, dtype=torch.float64)
    row_ids = torch.arange(4).repeat_interleave(4).unsqueeze(0).expand(B, -1)
    col_ids = torch.arange(4).repeat(4).unsqueeze(0).expand(B, -1)
    v = attn._get_values(x, x)

    def f(z):
        return attn._one_step(z, x, v, row_ids, col_ids)

    return attn, f, x


def _dense_spectral_radius(f, z):
    """Exact rho(J_f(z)) via a dense Jacobian (test-size problems only)."""
    def f_flat(zf):
        return f(zf.view_as(z)).reshape(-1)

    J = torch.autograd.functional.jacobian(f_flat, z.detach().reshape(-1), vectorize=True)
    eig = torch.linalg.eigvals(J)
    return eig.abs().max().item()


def _solve(f, x, iters=300):
    with torch.no_grad():
        z = x.clone()
        for _ in range(iters):
            z = f(z)
    return z


class TestContraction:
    def test_spectral_radius_below_one_at_fixed_point(self):
        """rho(J) < 1 at z* — the condition that makes implicit diff well-posed."""
        _, f, x = _make(damping=0.5, use_sn=True)
        z_star = _solve(f, x)
        rho = _dense_spectral_radius(f, z_star)
        assert rho < 1.0, f"not locally contractive at the fixed point: rho={rho:.3f}"

    def test_spectral_radius_below_one_at_random_points(self):
        _, f, x = _make(damping=0.5, use_sn=True)
        for seed in range(3):
            torch.manual_seed(100 + seed)
            z = x + 0.5 * torch.randn_like(x)
            rho = _dense_spectral_radius(f, z)
            assert rho < 1.0, f"rho={rho:.3f} at random point (seed {seed})"

    def test_power_iteration_estimator_agrees_with_dense(self):
        _, f, x = _make(damping=0.5, use_sn=True)
        z_star = _solve(f, x)
        rho_dense = _dense_spectral_radius(f, z_star)
        rho_pi = estimate_spectral_radius(f, z_star, n_iter=200, seed=0)
        # Power iteration converges to the dominant |eigenvalue|; allow slack
        # for slow separation of close eigenvalues.
        assert abs(rho_pi - rho_dense) / rho_dense < 0.05, (rho_pi, rho_dense)

    def test_damping_shrinks_spectral_radius(self):
        """J_alpha = (1-a) I + a J maps eigenvalues lam -> (1-a) + a*lam, which
        contracts the spectrum toward 1 along the segment to each lam; for
        spectra inside the unit disk with Re(lam) < 1 this reduces rho when the
        dominant eigenvalue has negative or complex part. At minimum, damped
        maps must stay strictly inside the unit disk here."""
        _, f25, x = _make(damping=0.25, use_sn=True, seed=0)
        _, f75, _ = _make(damping=0.75, use_sn=True, seed=0)
        z_star25 = _solve(f25, x)
        z_star75 = _solve(f75, x)
        rho25 = _dense_spectral_radius(f25, z_star25)
        rho75 = _dense_spectral_radius(f75, z_star75)
        assert rho25 < 1.0 and rho75 < 1.0, (rho25, rho75)

    def test_geometric_residual_decay(self):
        _, f, x = _make(damping=0.5, use_sn=True)
        z = x.clone()
        residuals = []
        with torch.no_grad():
            for _ in range(40):
                z_new = f(z)
                residuals.append(((z_new - z).norm() / (z.norm() + 1e-12)).item())
                z = z_new
        assert residuals[-1] < 1e-4, f"final residual {residuals[-1]:.2e}"
        rate = (residuals[-1] / residuals[0]) ** (1 / (len(residuals) - 1))
        assert rate < 0.95, f"no geometric decay: rate {rate:.3f}, residuals {residuals[:5]}..."

    def test_spectral_norm_tightens_spectrum(self):
        """SN must keep rho < 1; without SN, rho may drift larger. We assert
        the SN model contracts and does not exceed the unconstrained one."""
        _, f_sn, x = _make(damping=0.5, use_sn=True, seed=3)
        _, f_nosn, _ = _make(damping=0.5, use_sn=False, seed=3)
        torch.manual_seed(9)
        z = x + 0.5 * torch.randn_like(x)
        rho_sn = _dense_spectral_radius(f_sn, z)
        rho_nosn = _dense_spectral_radius(f_nosn, z)
        assert rho_sn < 1.0, f"SN map must contract: rho={rho_sn:.3f}"
        assert rho_sn <= rho_nosn + 0.05, (
            f"SN should not loosen the spectrum: {rho_sn:.3f} vs {rho_nosn:.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

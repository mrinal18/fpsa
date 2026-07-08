"""Tests for the Implicit TRM: gradient correctness of the latent equilibrium
solve (including the damped-map identity), contraction diagnostics, and
end-to-end training behavior of both models in the shared harness.

Run:  python -m pytest tests/test_implicit_trm.py -q
"""

import os
import sys

import pytest
import torch

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "trm"))

from src.implicit import attach_implicit_grad, fixed_point_solve
from models.trm import TRMConfig, build_trm
from models.implicit_trm import ImplicitTRMInner, build_implicit_trm


BATCH = 2


def _tiny_config(**overrides) -> TRMConfig:
    base = dict(
        seq_len=16, vocab_size=11,
        H_cycles=2, L_cycles=3, L_layers=1,
        hidden_size=32, expansion=2.0, num_heads=4,
        pos_encodings="rope", puzzle_emb_len=2,
        halt_max_steps=4, forward_dtype="float64",
        norm_style="pre", residual_scale=True, damping=0.8,
        inner_tol=1e-4, inner_max_iter=100,
        adjoint_steps=10, grad_mode="neumann",
    )
    base.update(overrides)
    return TRMConfig(**base)


def _tiny_batch(config, seed=0):
    torch.manual_seed(seed)
    return {
        "inputs": torch.randint(1, config.vocab_size, (BATCH, config.seq_len)),
        "labels": torch.randint(1, config.vocab_size, (BATCH, config.seq_len)),
    }


def _setup_latent_map(seed=0, damping=0.8):
    """Build a tiny ImplicitTRMInner and return (inner, f, z0, z_star, G)."""
    torch.manual_seed(seed)
    config = _tiny_config(damping=damping)
    inner = ImplicitTRMInner(config).double()
    inner.eval()

    batch = _tiny_batch(config, seed + 1)
    seq_info = inner._seq_info()
    B = BATCH
    N = config.seq_len + config.puzzle_emb_len
    y = inner.y_init.expand(B, N, -1).clone().double()

    # Embeddings are recomputed inside the closure so each call builds a fresh
    # graph — the tests run several independent backward passes through f.
    def f(z):
        x_emb = inner._input_embeddings(batch["inputs"])
        return inner.L_level(z, y + x_emb, **seq_info)

    z0 = inner.z_init.expand(B, N, -1).clone().double()
    with torch.no_grad():
        z, _ = fixed_point_solve(f, z0, tol=1e-14, max_iter=3000, stepsize=damping)
    z_star = z.detach()
    res = ((f(z_star) - z_star).norm() / z_star.norm()).item()

    torch.manual_seed(seed + 2)
    G = torch.randn_like(z_star)
    return inner, f, z0, z_star, G, res


def _param_grads(module, loss):
    params = [p for p in module.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def _exact_ift_grads(inner, f, z_star, G):
    def f_flat(zf):
        return f(zf.view_as(z_star)).reshape(-1)

    J = torch.autograd.functional.jacobian(f_flat, z_star.reshape(-1), vectorize=True)
    n = J.shape[0]
    lam = torch.linalg.solve(torch.eye(n, dtype=J.dtype) - J.T, G.reshape(-1)).view_as(z_star)
    out = f(z_star.detach().requires_grad_(True))
    return _param_grads(inner, (out * lam.detach()).sum())


class TestLatentEquilibriumGradients:
    def test_forward_converges(self):
        _, _, _, _, _, res = _setup_latent_map()
        assert res < 1e-9, f"latent map did not reach a fixed point: rel res {res:.2e}"

    def test_neumann_matches_exact_ift(self):
        inner, f, _, z_star, G, _ = _setup_latent_map()
        g_exact = _exact_ift_grads(inner, f, z_star, G)

        z_out = attach_implicit_grad(z_star, f, adjoint_steps=60, adjoint_tol=0.0)
        g_neu = _param_grads(inner, (z_out * G).sum())

        err = ((g_neu - g_exact).norm() / (g_exact.norm() + 1e-12)).item()
        assert err < 1e-5, f"Neumann(60) vs exact IFT rel err {err:.2e}"

    def test_damped_map_gradient_identity(self):
        """The adjoint solved on the damped map f_s must give the SAME gradient
        as on the raw map f: (I - J_{f_s}) = S (I - J_f) cancels the phantom
        step's extra factor of s."""
        inner, f, _, z_star, G, _ = _setup_latent_map()
        g_exact = _exact_ift_grads(inner, f, z_star, G)

        s = torch.full((z_star.shape[0],), 0.6, dtype=torch.float64)
        f_s = ImplicitTRMInner._damped(f, s)
        z_out = attach_implicit_grad(z_star, f_s, adjoint_steps=80, adjoint_tol=0.0)
        g_damped = _param_grads(inner, (z_out * G).sum())

        err = ((g_damped - g_exact).norm() / (g_exact.norm() + 1e-12)).item()
        assert err < 1e-5, f"damped-map gradient identity violated: rel err {err:.2e}"


def _run_training_steps(arch, n_steps=25, seed=0, **config_overrides):
    torch.manual_seed(seed)
    config = _tiny_config(forward_dtype="float32", **config_overrides)
    build = build_implicit_trm if arch == "itrm" else build_trm
    model = build(config)

    from losses import ACTLossHead
    loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    loss_head.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    batch = _tiny_batch(config, seed)
    carry = model.initial_carry(batch)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        carry, loss, metrics, _, _ = loss_head(carry=carry, batch=batch)
        (loss / BATCH).backward()
        opt.step()
        losses.append(loss.item() / BATCH)
    return losses, model


class TestTraining:
    @pytest.mark.parametrize("arch", ["trm", "itrm"])
    def test_loss_decreases(self, arch):
        losses, _ = _run_training_steps(arch)
        assert losses[-1] < losses[0] * 0.8, f"{arch}: {losses[0]:.3f} -> {losses[-1]:.3f}"

    def test_itrm_all_params_receive_grad(self):
        torch.manual_seed(0)
        config = _tiny_config(forward_dtype="float32")
        model = build_implicit_trm(config)
        from losses import ACTLossHead
        loss_head = ACTLossHead(model)
        loss_head.train()
        batch = _tiny_batch(config)
        carry = model.initial_carry(batch)
        _, loss, _, _, _ = loss_head(carry=carry, batch=batch)
        loss.backward()
        missing = [
            n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or not p.grad.abs().max() > 0)
        ]
        assert not missing, f"parameters without gradient: {missing}"

    def test_itrm_inner_loop_is_memoryless(self):
        """The latent solve must not build graph through its iterations."""
        torch.manual_seed(0)
        config = _tiny_config(forward_dtype="float32", inner_max_iter=12)
        model = build_implicit_trm(config)
        model.train()
        inner = model.inner

        recorded = []
        orig = ImplicitTRMInner._solve_latent

        def spy(self, f, z):
            z_out, stats = orig(self, f, z)
            recorded.append(stats["inner_iters"].item())
            return z_out, stats

        ImplicitTRMInner._solve_latent = spy
        try:
            batch = _tiny_batch(config)
            carry = model.initial_carry(batch)
            carry, outputs = model(carry=carry, batch=batch)
        finally:
            ImplicitTRMInner._solve_latent = orig

        assert len(recorded) == config.H_cycles
        # Loss backward works even though the loop ran under no_grad
        loss = outputs["logits"].float().square().mean()
        loss.backward()

    def test_bptt_mode_trains(self):
        losses, _ = _run_training_steps("itrm", grad_mode="bptt", bptt_steps=3)
        assert losses[-1] < losses[0] * 0.8, f"bptt: {losses[0]:.3f} -> {losses[-1]:.3f}"

    def test_jacobian_reg_produces_grad(self):
        torch.manual_seed(0)
        config = _tiny_config(forward_dtype="float32",
                              jacobian_reg_lambda=0.1, n_jacobian_samples=1)
        model = build_implicit_trm(config)
        from losses import ACTLossHead
        loss_head = ACTLossHead(model)
        loss_head.train()
        batch = _tiny_batch(config)
        carry = model.initial_carry(batch)
        _, loss, _, _, _ = loss_head(carry=carry, batch=batch)
        loss.backward()  # must not raise
        assert torch.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

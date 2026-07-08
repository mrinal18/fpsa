"""Integration test: SpatialFPSAModel trains under all three gradient modes.

Checks that unroll / phantom / neumann paths all optimize a tiny synthetic
Sudoku batch, produce gradients for every trainable parameter, and that the
implicit paths never build a graph through the forward loop (O(1) memory).

Run:  python -m pytest tests/test_spatial_grad_modes.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "sudoku_ablation"))

from fpsa_spatial import SpatialFPSAModel


def _tiny_batch(seed=0, B=8):
    torch.manual_seed(seed)
    grid = torch.randint(0, 10, (B, 81))
    target = torch.randint(1, 10, (B, 81))
    target[grid != 0] = -100
    row_ids = torch.arange(9).repeat_interleave(9).unsqueeze(0).expand(B, -1)
    col_ids = torch.arange(9).repeat(9).unsqueeze(0).expand(B, -1)
    return grid, row_ids, col_ids, target


def _make_model(grad_mode, seed=0):
    torch.manual_seed(seed)
    return SpatialFPSAModel(
        vocab_size=10, hidden_dim=32, num_layers=2, num_heads=4,
        value_mode="fixed", k_max=8, epsilon=1e-3, damping=0.5,
        ffn_mult=2.0, dropout=0.1, use_spectral_norm=True, max_grid_size=9,
        grad_mode=grad_mode, adjoint_steps=5,
    )


@pytest.mark.parametrize("grad_mode", ["unroll", "phantom", "neumann"])
def test_training_step_decreases_loss(grad_mode):
    model = _make_model(grad_mode)
    model.train()
    grid, row_ids, col_ids, target = _tiny_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        out = model(grid, row_ids, col_ids, target=target)
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())

    assert losses[-1] < losses[0] * 0.9, (
        f"{grad_mode}: loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


@pytest.mark.parametrize("grad_mode", ["phantom", "neumann"])
def test_all_params_receive_grad(grad_mode):
    model = _make_model(grad_mode)
    model.train()
    grid, row_ids, col_ids, target = _tiny_batch()
    out = model(grid, row_ids, col_ids, target=target)
    out["loss"].backward()

    missing = [
        n for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().max() == 0)
    ]
    assert not missing, f"{grad_mode}: parameters without gradient: {missing}"


def test_grad_modes_agree_directionally():
    """Phantom/Neumann gradients must point in a similar direction to the
    unrolled gradient on identically initialized models."""
    grid, row_ids, col_ids, target = _tiny_batch()

    grads = {}
    for mode in ["unroll", "phantom", "neumann"]:
        model = _make_model(mode, seed=7)
        model.train()
        # disable dropout for exact comparability
        for layer in model.layers:
            layer.attn.dropout_p = 0.0
            layer.dropout_attn.p = 0.0
            for m in layer.ffn:
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.0
        torch.manual_seed(0)
        out = model(grid, row_ids, col_ids, target=target)
        out["loss"].backward()
        grads[mode] = torch.cat([
            p.grad.reshape(-1) for _, p in sorted(model.named_parameters())
            if p.grad is not None
        ])

    cos_pn = torch.nn.functional.cosine_similarity(grads["phantom"], grads["unroll"], dim=0)
    cos_nu = torch.nn.functional.cosine_similarity(grads["neumann"], grads["unroll"], dim=0)
    assert cos_nu > 0.6, f"neumann vs unroll cosine too low: {cos_nu:.3f}"
    assert cos_pn > 0.3, f"phantom vs unroll cosine too low: {cos_pn:.3f}"
    # The Neumann refinement should be at least as aligned as raw phantom.
    assert cos_nu >= cos_pn - 0.05, (cos_nu.item(), cos_pn.item())


def test_implicit_path_detaches_forward_loop():
    """In implicit modes the loop must run under no_grad: the returned state's
    graph depth is that of ONE step, not k_max steps."""
    model = _make_model("neumann")
    model.train()
    grid, row_ids, col_ids, target = _tiny_batch()

    seen_steps = {}
    orig = type(model.layers[0].attn)._run_loop

    def spy(self, x, row_ids, col_ids, drop_mask=None):
        z, steps, norms, tok = orig(self, x, row_ids, col_ids, drop_mask)
        seen_steps["grad_in_loop"] = z.requires_grad and z.grad_fn is not None
        return z, steps, norms, tok

    type(model.layers[0].attn)._run_loop = spy
    try:
        out = model(grid, row_ids, col_ids, target=target)
        out["loss"].backward()
    finally:
        type(model.layers[0].attn)._run_loop = orig

    assert seen_steps["grad_in_loop"] is False, (
        "forward loop built an autograd graph in implicit mode"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

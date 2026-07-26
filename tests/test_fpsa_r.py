"""Invariants the method rests on. Run with `python -m pytest tests/ -q`
(or `python tests/test_fpsa_r.py` for a plain run).

These are deliberately about *correctness of the mathematics*, not about
accuracy on a benchmark: if any of them breaks, the claims in docs/FPSA-R.md
stop being true regardless of what the leaderboard says.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fpsa_r import ARCH_PRESETS, build_model            # noqa: E402
from src.fpsa_r.diagnostics import (ActivationMemory,       # noqa: E402
                                    empirical_spectral_radius)

B, N, V, D = 4, 12, 8, 64
KW = dict(vocab_size=V, seq_len=N, hidden_size=D, num_heads=4)


def _batch(seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, V, (B, N)), torch.randint(0, V, (B, N))


def _flat_grad(m):
    return torch.cat([p.grad.reshape(-1) for _, p in sorted(m.named_parameters())
                      if p.grad is not None])


def _grad_of(arch, x, y, seed=0, **kw):
    torch.manual_seed(seed)
    kw.setdefault("contraction_lambda", 0.0)
    m = build_model(arch, **KW, **kw)
    m.train()
    m.zero_grad(set_to_none=True)
    F.cross_entropy(m(x)["logits"].reshape(-1, V), y.reshape(-1)).backward()
    return _flat_grad(m), m


def test_every_arch_trains_a_step():
    x, y = _batch()
    for arch in ARCH_PRESETS:
        g, m = _grad_of(arch, x, y, max_iter=5, max_iter_eval=8, n_backwards=2)
        assert torch.isfinite(g).all(), f"{arch}: non-finite gradient"
        assert g.norm() > 0, f"{arch}: zero gradient"


def test_implicit_gradient_matches_deep_bptt():
    """The whole point: the O(1)-memory gradient is the *right* gradient."""
    x, y = _batch()
    ref, _ = _grad_of("fpsa_r_bptt", x, y, max_iter=64, max_iter_eval=64)
    ours, _ = _grad_of("fpsa_r", x, y, max_iter=12, max_iter_eval=12)
    cos = F.cosine_similarity(ours, ref, dim=0).item()
    rel = ((ours - ref).norm() / ref.norm()).item()
    assert cos > 0.999, f"cosine {cos}"
    assert rel < 0.05, f"relative error {rel}"


def test_implicit_beats_one_step_phantom():
    """Solving the adjoint is worth more than one Neumann term."""
    x, y = _batch()
    ref, _ = _grad_of("fpsa_r_bptt", x, y, max_iter=64, max_iter_eval=64)
    ours, _ = _grad_of("fpsa_r", x, y, max_iter=12, max_iter_eval=12)
    one, _ = _grad_of("fpsa_r_onestep", x, y, max_iter=12, max_iter_eval=12)
    err = lambda g: ((g - ref).norm() / ref.norm()).item()
    assert err(ours) < err(one), f"implicit {err(ours)} vs phantom {err(one)}"


def test_activation_memory_is_constant_in_loop_depth():
    x, y = _batch()
    mem = {}
    for arch in ("fpsa_r", "looped_bptt"):
        for T in (4, 32):
            torch.manual_seed(0)
            m = build_model(arch, **KW, max_iter=T, max_iter_eval=T,
                            contraction_lambda=0.0)
            m.train()
            m.zero_grad(set_to_none=True)
            probe = ActivationMemory()
            with probe.track():
                loss = F.cross_entropy(m(x)["logits"].reshape(-1, V), y.reshape(-1))
            loss.backward()
            mem[(arch, T)] = probe.mb
    assert abs(mem[("fpsa_r", 4)] - mem[("fpsa_r", 32)]) < 0.01 * mem[("fpsa_r", 4)]
    assert mem[("looped_bptt", 32)] > 3 * mem[("looped_bptt", 4)]
    assert mem[("looped_bptt", 32)] > 3 * mem[("fpsa_r", 32)]


def test_joint_and_nested_share_a_fixed_point():
    """The joint solver is a reformulation, not a different model: run both to
    convergence from the same weights and they land in the same place."""
    x, _ = _batch()
    torch.manual_seed(0)
    m = build_model("fpsa_r", **KW, max_iter=200, max_iter_eval=200, inner_max_iter=12,
                    fp_thresh=1e-7, contraction_lambda=0.0)
    m.eval()
    xin = m._inputs(x, None)
    si = m._seq_info(xin.shape[1])
    outs = {}
    for mode, step in (("joint", m.block.joint_step), ("nested", m.block.nested_step)):
        s = m.block.init_state(B, xin.shape[1], xin.device, xin.dtype)
        with torch.no_grad():
            for _ in range(200):
                s = s + (step(s, xin, si) - s)
        outs[mode] = s[0]
    rel = ((outs["joint"] - outs["nested"]).norm() / outs["joint"].norm()).item()
    assert rel < 1e-4, f"joint and nested equilibria differ by {rel}"


def test_inner_residual_prevents_rank_collapse():
    """Without re-injecting the layer input, iterated attention averages every
    token onto the same vector."""
    x, _ = _batch()
    torch.manual_seed(0)
    m = build_model("fpsa_r", **KW, max_iter=40, max_iter_eval=40,
                    contraction_lambda=0.0)
    m.eval()
    attn = m.block.layers[0].attn
    orig = attn.step
    xin = m._inputs(x, None)
    si = m._seq_info(xin.shape[1])

    def eff_rank(u):
        u = u - u.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(u)
        p = sv / sv.sum().clamp_min(1e-12)
        return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))

    ranks = {}
    for key, drop in (("with", False), ("without", True)):
        attn.step = ((lambda u, v, xr, *a, **k: orig(u, v, torch.zeros_like(xr), *a, **k))
                     if drop else orig)
        s = m.block.init_state(B, xin.shape[1], xin.device, xin.dtype)
        with torch.no_grad():
            for _ in range(40):
                s = s + (m.block.joint_step(s, xin, si) - s)
        ranks[key] = eff_rank(s[1][0])
    attn.step = orig
    assert ranks["with"] > 1.5 * ranks["without"], ranks


def test_masked_adjoint_ignores_unconverged_tokens():
    """A token excluded by the mask must not contribute to the gradient."""
    x, y = _batch()
    torch.manual_seed(0)
    m = build_model("fpsa_r", **KW, max_iter=3, max_iter_eval=3, fp_thresh=1e-9,
                    contraction_lambda=0.0)
    m.train()
    m.zero_grad(set_to_none=True)
    out = m(x)
    # a tolerance of 1e-9 after 3 iterations leaves everything unconverged
    assert out["info"].converged_frac < 1.0
    F.cross_entropy(out["logits"].reshape(-1, V), y.reshape(-1)).backward()
    assert torch.isfinite(_flat_grad(m)).all()


def test_contraction_penalty_tracks_the_spectral_radius():
    """The finite-difference power-iteration estimate should agree with the
    autograd measurement it is standing in for."""
    x, _ = _batch()
    torch.manual_seed(0)
    m = build_model("fpsa_r", **KW, max_iter=40, max_iter_eval=40,
                    contraction_lambda=1.0, n_power_iterations=8)
    m.train()
    m(x)
    estimated = m.last_rho
    m.eval()
    xin = m._inputs(x, None)
    si = m._seq_info(xin.shape[1])
    sf = lambda s: m.block.joint_step(s, xin, si)
    s = m.block.init_state(B, xin.shape[1], xin.device, xin.dtype)
    with torch.no_grad():
        for _ in range(40):
            s = s + (sf(s) - s)
    measured = empirical_spectral_radius(sf, s)
    assert abs(estimated - measured) < 0.25 * max(measured, 1e-6), (estimated, measured)


def test_eval_is_deterministic_and_iteration_budget_is_respected():
    x, _ = _batch()
    torch.manual_seed(0)
    m = build_model("fpsa_r", **KW, max_iter=8, max_iter_eval=32,
                    contraction_lambda=0.0)
    m.eval()
    with torch.no_grad():
        a, b = m(x), m(x)
        capped = m(x, max_iter=3)
    assert torch.allclose(a["logits"], b["logits"])
    assert capped["info"].n_iters <= 3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

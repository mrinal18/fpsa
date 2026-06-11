"""Gate 2 unit tests (CPU, no network). Run: python3 tests/test_gate2.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.sudoku import (shuffle_sudoku, synthetic_sudoku, is_valid_solution,
                         is_consistent, encode, load_trm_dataset, _BASE)
from models.model import FPSASeqModel
from utils.logging_utils import set_seed


def test_base_grid_valid():
    assert is_valid_solution(_BASE), "base grid invalid"
    print("PASS base grid valid")


def test_augmentation_preserves_everything(n=300):
    rng = np.random.default_rng(0)
    puz, sol = synthetic_sudoku(n, n_givens=30, seed=1)
    for i in range(n):
        assert is_valid_solution(sol[i])
        assert is_consistent(puz[i], sol[i])
        p2, s2 = shuffle_sudoku(puz[i], sol[i], rng)
        assert is_valid_solution(s2), f"augmented solution invalid at {i}"
        assert is_consistent(p2, s2), f"augmented pair inconsistent at {i}"
        assert (p2 > 0).sum() == (puz[i] > 0).sum(), "givens count changed"
        # digit relabeling is a bijection: blank count preserved
        assert (p2 == 0).sum() == (puz[i] == 0).sum()
    print(f"PASS augmentation validity/consistency on {n} puzzles")


def test_unseeded_path_matches_upstream_semantics():
    # rng=None must use np.random (upstream behavior) and still be valid
    np.random.seed(123)
    p, s = synthetic_sudoku(1, seed=2)
    p2, s2 = shuffle_sudoku(p[0], s[0], rng=None)
    assert is_valid_solution(s2) and is_consistent(p2, s2)
    print("PASS unseeded (upstream-semantics) path")


def test_encoding_and_npy_roundtrip():
    puz, sol = synthetic_sudoku(8, seed=3)
    xi, yi = encode(puz), encode(sol)
    assert xi.shape == (8, 81) and xi.min() >= 1 and xi.max() <= 10
    assert yi.min() >= 2 and yi.max() <= 10  # solutions have no blanks
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "train")
        os.makedirs(d)
        np.save(os.path.join(d, "all__inputs.npy"), xi.numpy())
        np.save(os.path.join(d, "all__labels.npy"), yi.numpy())
        x2, y2 = load_trm_dataset(td, "train")
        assert torch.equal(x2, xi) and torch.equal(y2, yi)
    print("PASS encoding + TRM npy format roundtrip")


def _make_model(d=32, heads=4, max_iter=20, tol=0.0, neumann_steps=6):
    m = FPSASeqModel(vocab_size=11, num_classes=11, d_model=d, num_heads=heads,
                     value_mode="fixed_ffn", damping=0.5, max_iter=max_iter,
                     tol=tol, backward="neumann", neumann_steps=neumann_steps,
                     use_spectral_norm=True, use_rope=True, pos_mode="2d",
                     grid_hw=(9, 9), ffn_mult=2.0)
    return m


def test_model_2d_forward_backward():
    set_seed(0)
    m = _make_model()
    puz, sol = synthetic_sudoku(4, seed=4)
    x, y = encode(puz), encode(sol)
    logits, st = m(x)
    assert logits.shape == (4, 81, 11)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 11), y.view(-1))
    loss.backward()
    gn = sum(p.grad.norm() ** 2 for p in m.parameters() if p.grad is not None) ** 0.5
    assert torch.isfinite(loss) and torch.isfinite(gn)
    assert st.iterations > 0 and len(st.rel_residual_mean) == st.iterations
    print(f"PASS 2D forward/backward (loss {loss.item():.3f}, |g| {gn:.3f}, "
          f"iters {st.iterations}, res {st.rel_residual_mean[-1]:.2e})")


def test_fd_gradcheck_2d(n_params_checked=8, eps=1e-6):
    """Finite differences vs Neumann implicit gradient on the 2D path,
    float64, fixed 50-iter solve."""
    torch.set_default_dtype(torch.float64)
    set_seed(0)
    m = _make_model(d=16, heads=4, max_iter=50, tol=0.0, neumann_steps=25)
    m.eval()
    puz, sol = synthetic_sudoku(2, seed=5)
    x, y = encode(puz), encode(sol)

    import copy as _copy
    def loss_of(model):
        logits, _ = model(x, max_iter=50)
        return torch.nn.functional.cross_entropy(logits.view(-1, 11), y.view(-1))

    base = _copy.deepcopy(m)
    l = loss_of(m)
    l.backward()
    gflat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                       for p in m.parameters()])
    params = list(base.parameters())
    sizes = torch.tensor([p.numel() for p in params])
    cum = torch.cumsum(sizes, 0)
    idxs = torch.randperm(int(cum[-1]))[:n_params_checked]
    max_err = 0.0
    for fi in idxs.tolist():
        pi = int(torch.searchsorted(cum, fi, right=True))
        local = fi - (int(cum[pi-1]) if pi > 0 else 0)
        ls = []
        for sgn in (+1, -1):
            mm = _copy.deepcopy(base)
            with torch.no_grad():
                list(mm.parameters())[pi].view(-1)[local] += sgn * eps
            ls.append(loss_of(mm).item())
        fd = (ls[0] - ls[1]) / (2 * eps)
        an = gflat[fi].item()
        max_err = max(max_err, abs(fd - an) / max(abs(an), abs(fd), 1e-12))
    torch.set_default_dtype(torch.float32)
    assert max_err < 1e-3, f"FD mismatch on 2D path: {max_err:.2e}"
    print(f"PASS 2D-path FD vs Neumann gradient (max rel err {max_err:.2e}, "
          f"k=25 Neumann terms)")


if __name__ == "__main__":
    test_base_grid_valid()
    test_augmentation_preserves_everything()
    test_unseeded_path_matches_upstream_semantics()
    test_encoding_and_npy_roundtrip()
    test_model_2d_forward_backward()
    test_fd_gradcheck_2d()
    print("\nALL GATE 2 UNIT TESTS PASS")

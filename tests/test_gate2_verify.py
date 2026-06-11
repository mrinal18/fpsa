"""Gate 2 VERIFICATION suite — adversarial checks against upstream code.

Unlike tests/test_gate2.py (self-consistency), these tests verify against
the actual HRM/TRM source cloned at /home/claude/TinyRecursiveModels:
  V-A  augmentation byte-equivalence with their shuffle_sudoku
  V-B  their REAL builder (network call monkeypatched to a local CSV)
       end-to-end -> our loader
  V-C  2D RoPE relative-position property (scores depend only on
       (drow, dcol))
  V-D  EMA matches closed form; copy_to does not mutate the source model
  V-E  exact-match metric micro-check
(resume determinism V-F runs as a separate script via subprocess)
"""
import importlib.util
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.sudoku import shuffle_sudoku, synthetic_sudoku, load_trm_dataset, encode
from models.fpsa_block import RoPE2D
from train.train_sudoku import EMA, evaluate
from models.model import FPSASeqModel
from utils.logging_utils import set_seed

TRM_DATASET_DIR = "/home/claude/TinyRecursiveModels/dataset"


def _load_upstream():
    sys.path.insert(0, TRM_DATASET_DIR)
    spec = importlib.util.spec_from_file_location(
        "trm_builder", os.path.join(TRM_DATASET_DIR, "build_sudoku_dataset.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_VA_augmentation_byte_equivalence(n=100):
    up = _load_upstream()
    puz, sol = synthetic_sudoku(n, seed=11)
    for i in range(n):
        seed = 1000 + i
        np.random.seed(seed)
        p_up, s_up = up.shuffle_sudoku(puz[i].copy(), sol[i].copy())
        np.random.seed(seed)
        p_us, s_us = shuffle_sudoku(puz[i].copy(), sol[i].copy(), rng=None)
        assert np.array_equal(p_up, p_us) and np.array_equal(s_up, s_us), \
            f"augmentation diverges from upstream at case {i}"
    print(f"PASS V-A: augmentation byte-identical to upstream over {n} seeded cases")


def test_VB_their_builder_to_our_loader():
    up = _load_upstream()
    puz, sol = synthetic_sudoku(6, n_givens=28, seed=12)

    def grid_to_q(g):
        return "".join(str(v) if v > 0 else "." for v in g.flatten())

    def grid_to_a(g):
        return "".join(str(v) for v in g.flatten())

    with tempfile.TemporaryDirectory() as td:
        for split, sl in [("train", slice(0, 4)), ("test", slice(4, 6))]:
            with open(os.path.join(td, f"{split}.csv"), "w") as f:
                f.write("source,q,a,rating\n")
                for p, s in zip(puz[sl], sol[sl]):
                    f.write(f"syn,{grid_to_q(p)},{grid_to_a(s)},100\n")

        up.hf_hub_download = lambda repo, fn, repo_type: os.path.join(td, fn)
        out = os.path.join(td, "built")
        cfg = up.DataProcessConfig(source_repo="x/y", output_dir=out,
                                   subsample_size=None, num_aug=3)
        np.random.seed(7)
        up.convert_subset("train", cfg)
        up.convert_subset("test", cfg)

        xtr, ytr = load_trm_dataset(out, "train")
        xte, yte = load_trm_dataset(out, "test")
        assert xtr.shape == (4 * (1 + 3), 81), xtr.shape   # augments applied
        assert xte.shape == (2, 81)                        # test never augmented
        # first example of each group is unaugmented: matches raw encoding
        assert torch.equal(xtr[0], encode(puz[0:1])[0])
        assert torch.equal(ytr[0], encode(sol[0:1])[0])
        # all augmented labels decode to valid solutions consistent with inputs
        from data.sudoku import is_valid_solution, is_consistent
        for i in range(len(xtr)):
            p = (xtr[i].numpy() - 1).reshape(9, 9)
            s = (ytr[i].numpy() - 1).reshape(9, 9)
            assert is_valid_solution(s) and is_consistent(p, s)
    print("PASS V-B: upstream builder (real code, patched download) -> our loader; "
          "format, augmentation count, no-test-augmentation, validity all hold")


def test_VC_rope2d_relative_position(trials=200):
    torch.manual_seed(0)
    rope = RoPE2D(head_dim=16, grid_hw=(9, 9))
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)
    # place identical q at position i and identical k at position j
    Q = q.expand(1, 1, 81, 16).contiguous()
    K = k.expand(1, 1, 81, 16).contiguous()
    Qr, Kr = rope(Q, K)
    S = (Qr @ Kr.transpose(-2, -1))[0, 0]  # (81, 81)
    rng = np.random.default_rng(0)
    for _ in range(trials):
        r1, c1, r2, c2 = rng.integers(0, 9, 4)
        dr, dc = r2 - r1, c2 - c1
        r3 = rng.integers(max(0, -dr), min(9, 9 - dr))
        c3 = rng.integers(max(0, -dc), min(9, 9 - dc))
        i1, j1 = r1 * 9 + c1, r2 * 9 + c2
        i2, j2 = r3 * 9 + c3, (r3 + dr) * 9 + (c3 + dc)
        assert torch.allclose(S[i1, j1], S[i2, j2], atol=1e-5), \
            f"score not translation-invariant: ({r1},{c1})->({r2},{c2}) vs shifted"
    print(f"PASS V-C: RoPE2D scores depend only on (drow, dcol) over {trials} pairs")


def test_VD_ema_closed_form():
    lin = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        lin.weight.fill_(0.0)
    ema = EMA(lin, decay=0.9)
    vals = [1.0, 2.0, 3.0]
    expect = 0.0
    for v in vals:
        with torch.no_grad():
            lin.weight.fill_(v)
        ema.update(lin)
        expect = 0.9 * expect + 0.1 * v
    assert abs(ema.shadow["weight"].item() - expect) < 1e-7
    src_before = lin.weight.item()
    import copy
    tgt = copy.deepcopy(lin)
    ema.copy_to(tgt)
    assert lin.weight.item() == src_before, "copy_to mutated source"
    assert abs(tgt.weight.item() - expect) < 1e-7
    print(f"PASS V-D: EMA matches closed form ({expect:.4f}); copy_to non-mutating")


def test_VE_exact_match_metric():
    set_seed(0)
    m = FPSASeqModel(vocab_size=11, num_classes=11, d_model=16, num_heads=4,
                     value_mode="fixed", max_iter=2, tol=0.0, pos_mode="2d",
                     grid_hw=(9, 9))
    puz, sol = synthetic_sudoku(4, seed=13)
    x, y = encode(puz), encode(sol)
    cell, exact, _ = evaluate(m, x, y)
    # untrained model: exact must be 0, cell roughly chance
    assert exact == 0.0 and 0.0 <= cell <= 0.5
    cell2, exact2, _ = evaluate(m, x, x)  # predicting inputs vs inputs label
    # cannot be exact unless model is the identity; just sanity that metric
    # uses ALL 81 cells: a perfect oracle check
    class Oracle(torch.nn.Module):
        def forward(self, t, max_iter=None):
            lg = torch.nn.functional.one_hot(t, 11).float() * 10
            from models.solver import SolveStats
            st = SolveStats(); st.rel_residual_mean = [0.0]; st.iterations = 1
            return lg, st
        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])
        def eval(self): return self
        def train(self): return self
    cell3, exact3, _ = evaluate(Oracle(), y, y)
    assert cell3 == 1.0 and exact3 == 1.0
    print("PASS V-E: exact-match metric (0 for untrained, 1 for oracle)")


if __name__ == "__main__":
    test_VA_augmentation_byte_equivalence()
    test_VB_their_builder_to_our_loader()
    test_VC_rope2d_relative_position()
    test_VD_ema_closed_form()
    test_VE_exact_match_metric()
    print("\nALL VERIFICATION TESTS PASS (V-A..V-E)")

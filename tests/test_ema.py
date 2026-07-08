"""Tests for the initialization-debiased EMA.

The raw EMA shadow after N updates carries weight mu^N on the random init —
0.79 after 240 steps at mu=0.999 — which made early evals report chance-level
accuracy while the live model had learned. The debiased estimate must equal
the average of only the visited weights.

Run:  python -m pytest tests/test_ema.py -q
"""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "trm"))

from train import EMAHelper


def _model(val=None):
    m = nn.Linear(4, 4, bias=False)
    if val is not None:
        with torch.no_grad():
            m.weight.fill_(val)
    return m


class TestDebiasedEMA:
    def test_debiased_removes_init_exactly(self):
        """Hold the params constant at theta*: after any number of updates the
        debiased EMA must equal theta* exactly, while the raw shadow is still
        contaminated by the init."""
        torch.manual_seed(0)
        m = _model()          # random init theta_0
        ema = EMAHelper(m, mu=0.999)

        with torch.no_grad():
            m.weight.fill_(3.0)   # theta* from step 1 on
        for _ in range(50):
            ema.update(m)

        raw = ema.shadow["weight"]
        debiased = ema._debiased("weight")

        assert not torch.allclose(raw, m.weight.float(), atol=1e-3), \
            "raw shadow should still be biased toward the init after 50 steps"
        assert torch.allclose(debiased, m.weight.float(), atol=1e-5), \
            f"debiased EMA != theta*: max err {(debiased - m.weight).abs().max():.2e}"

    def test_debiasing_vanishes_at_long_horizon(self):
        """As mu^N -> 0 the debiased and raw EMA coincide (reference protocol
        unchanged)."""
        m = _model(1.0)
        ema = EMAHelper(m, mu=0.9)
        with torch.no_grad():
            m.weight.fill_(2.0)
        for _ in range(200):   # mu^200 ~ 7e-10
            ema.update(m)
        assert torch.allclose(ema._debiased("weight"), ema.shadow["weight"], atol=1e-6)

    def test_swap_in_out_roundtrip(self):
        torch.manual_seed(1)
        m = _model()
        ema = EMAHelper(m, mu=0.99)
        with torch.no_grad():
            m.weight.fill_(5.0)
        ema.update(m)
        # NB: after exactly one update the debiased EMA equals the single
        # visited weight (5.0) — the live value. Move the live weights past
        # the EMA before checking that swap_in changes them.
        with torch.no_grad():
            m.weight.fill_(7.0)

        before = m.weight.detach().clone()
        ema.swap_in(m)
        assert not torch.allclose(m.weight, before), "swap_in must change live weights"
        ema.swap_out(m)
        assert torch.allclose(m.weight, before), "swap_out must restore live weights"

    def test_zero_updates_swap_is_identity_safe(self):
        torch.manual_seed(2)
        m = _model()
        ema = EMAHelper(m, mu=0.999)
        ema.swap_in(m)   # n_updates == 0: falls back to raw shadow (== init)
        ema.swap_out(m)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

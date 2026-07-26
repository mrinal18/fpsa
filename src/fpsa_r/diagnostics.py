"""Measurement tools. Everything reported in the paper tables comes from here.

* ``ActivationMemory`` counts the bytes autograd actually stores for the
  backward pass, via ``saved_tensors_hooks``. This is a hardware-independent,
  exact measure of the quantity that separates implicit differentiation from
  BPTT -- far more informative than ``max_memory_allocated``, which is polluted
  by the allocator's caching behaviour.
* ``empirical_lipschitz`` power-iterates the true Jacobian of the joint map at
  the solution, giving the *measured* contraction factor rather than the
  analytic bound.
* ``gradient_fidelity`` compares a gradient scheme against the exact gradient of
  a deeply-unrolled loop -- the ground truth for "did the cheap gradient point
  the right way".
"""

from contextlib import contextmanager
from typing import Callable, Dict, List

import torch


class ActivationMemory:
    """Context manager totalling the bytes of every tensor autograd saves."""

    def __init__(self):
        self.total = 0
        self.peak = 0
        self._live = 0
        self._seen = set()

    @contextmanager
    def track(self):
        self.total = 0
        self._live = 0
        self.peak = 0
        self._seen = set()

        def pack(t):
            key = (t.data_ptr(), t.shape, t.dtype)
            if key not in self._seen and t.data_ptr() != 0:
                self._seen.add(key)
                nbytes = t.numel() * t.element_size()
                self.total += nbytes
                self._live += nbytes
                self.peak = max(self.peak, self._live)
            return t

        def unpack(t):
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield self

    @property
    def mb(self) -> float:
        return self.total / (1024 ** 2)


def measure_activation_memory(model, batch, loss_fn, **fwd_kwargs) -> float:
    """MB of activations stored for one training step."""
    probe = ActivationMemory()
    model.zero_grad(set_to_none=True)
    with probe.track():
        out = model(*batch[:-1], **fwd_kwargs) if isinstance(batch, (list, tuple)) else model(batch)
        loss = loss_fn(out, batch)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return probe.mb


def empirical_spectral_radius(step_fn: Callable[[torch.Tensor], torch.Tensor],
                              s: torch.Tensor, n_iter: int = 15) -> float:
    """Spectral radius of dG/ds at ``s``, by power iteration with VJPs.

    Repeated application of ``J^T`` makes the Rayleigh quotient converge to
    ``|lambda_max(J^T)| = |lambda_max(J)|``, which is the quantity that actually
    governs the asymptotic convergence rate of the fixed-point iteration (and the
    convergence of the adjoint solve). Values < 1 certify a locally stable
    equilibrium; values >= 1 mean the loop is not converging and the implicit
    gradient is not well posed.
    """
    v = torch.randn_like(s)
    v = v / v.norm()
    rho = 0.0
    with torch.enable_grad():
        s_var = s.detach().requires_grad_(True)
        out = step_fn(s_var)
        for _ in range(n_iter):
            (jtv,) = torch.autograd.grad(out, s_var, v, retain_graph=True)
            nrm = float(jtv.norm())
            if nrm < 1e-12:
                return 0.0
            v = jtv / nrm
            rho = nrm
    return rho


def gradient_fidelity(model, tokens, targets, loss_fn,
                      reference_iters: int = 64) -> Dict[str, float]:
    """Cosine similarity and relative error of the model's gradient scheme
    against exact BPTT through ``reference_iters`` unrolled steps."""
    import copy
    from .config import FPSARConfig

    def flat_grad(m):
        gs = [p.grad.reshape(-1) for p in m.parameters() if p.grad is not None]
        return torch.cat(gs) if gs else torch.zeros(1)

    # reference: exact gradient of a deep unroll
    ref = copy.deepcopy(model)
    ref.cfg = FPSARConfig(**{**model.cfg.to_dict(), "grad_mode": "bptt",
                             "max_iter": reference_iters})
    ref.block.cfg = ref.cfg
    for layer in ref.block.layers:
        layer.cfg = ref.cfg
    ref.train()
    ref.zero_grad(set_to_none=True)
    loss_fn(ref(tokens, max_iter=reference_iters)["logits"], targets).backward()
    g_ref = flat_grad(ref)

    model.train()
    model.zero_grad(set_to_none=True)
    loss_fn(model(tokens)["logits"], targets).backward()
    g = flat_grad(model)

    n = min(g.numel(), g_ref.numel())
    g, g_ref = g[:n], g_ref[:n]
    cos = float(torch.nn.functional.cosine_similarity(g, g_ref, dim=0))
    rel = float((g - g_ref).norm() / (g_ref.norm() + 1e-12))
    model.zero_grad(set_to_none=True)
    return {"cosine": cos, "rel_error": rel,
            "grad_norm": float(g.norm()), "ref_norm": float(g_ref.norm())}


@torch.no_grad()
def residual_trajectory(model, tokens, max_iter: int = 64) -> List[float]:
    """Mean relative residual per outer iteration (convergence curve)."""
    model.eval()
    out = model(tokens, max_iter=max_iter, record_trace=True)
    return list(out["info"].residual_trace)


@torch.no_grad()
def token_convergence_map(model, tokens, max_iter: int = 32) -> torch.Tensor:
    """(N, T) per-token distance to a high-fidelity reference fixed point.

    Reproduces Figure 2 of the FPSA paper for the reasoning setting.
    """
    model.eval()
    cfg = model.cfg
    x_inj = model._inputs(tokens, None)
    B, N, _ = x_inj.shape
    seq_info = model._seq_info(N)
    step = model.block.nested_step if cfg.solver_mode == "nested" else model.block.joint_step
    step_fn = lambda s: step(s, x_inj, seq_info)

    s = model.block.init_state(B, N, x_inj.device, x_inj.dtype)
    traj = []
    for _ in range(max_iter):
        s = s + cfg.stepsize * (step_fn(s) - s)
        traj.append(s[0].clone())
    # reference: keep going well past the cap
    ref = s
    for _ in range(4 * max_iter):
        ref = ref + cfg.stepsize * (step_fn(ref) - ref)
    z_star = ref[0]
    dist = torch.stack([(z - z_star).norm(dim=-1) / z_star.norm(dim=-1).clamp_min(1e-8)
                        for z in traj], dim=-1)      # (B, N, T)
    return dist[0]

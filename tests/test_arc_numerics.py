import pytest
import torch
from src.fpsa_arc.numerics import SolverConfig, anderson, gmres, residual, solve, ConvergenceError


def test_true_forward_residual_and_batch_independence():
    torch.manual_seed(3)
    m = torch.randn(3, 8, 8, dtype=torch.float64) * .035
    b = torch.randn(3, 2, 4, dtype=torch.float64)
    fn = lambda s: torch.einsum('bij,bj->bi', m, s.flatten(1)).reshape_as(s) + b
    cfg = SolverConfig(max_iter=80, tol=1e-9)
    actual, info = anderson(fn, torch.zeros_like(b), cfg)
    exact = torch.linalg.solve(torch.eye(8).double()-m, b.flatten(1).unsqueeze(-1)).reshape_as(b)
    assert torch.allclose(actual, exact, atol=1e-8, rtol=1e-8)
    assert torch.equal(info.residual, residual(fn(actual), actual))
    singles = []
    for j in range(3):
        f = lambda s, j=j: (s.flatten(1) @ m[j].T).reshape_as(s)+b[j:j+1]
        singles.append(anderson(f, torch.zeros_like(b[j:j+1]), cfg)[0])
    assert torch.allclose(actual, torch.cat(singles), atol=1e-10)
    assert info.nfe >= info.iterations + 2


@pytest.mark.parametrize('dtype,tol', [(torch.float32, 1e-5), (torch.float64, 1e-10)])
def test_gmres_against_direct_nonnormal_and_zero_rhs(dtype, tol):
    torch.manual_seed(4)
    m = torch.randn(4, 24, 24, dtype=dtype)*.02
    m += torch.diag(torch.full((23,), .25, dtype=dtype), diagonal=1)
    b = torch.randn(4, 3, 8, dtype=dtype)
    b[0] = 0
    vjp = lambda v: torch.einsum('bij,bj->bi', m.transpose(1,2), v.flatten(1)).reshape_as(v)
    x, calls, rel = gmres(vjp,b,max_iter=80,tol=tol,restart=16)
    exact = torch.linalg.solve(torch.eye(24, dtype=dtype)-m.transpose(1,2),b.flatten(1).unsqueeze(-1)).reshape_as(b)
    assert rel <= tol
    assert torch.allclose(x, exact, atol=5*tol, rtol=5*tol)
    assert torch.equal(x[0], b[0])
    long, _, longrel = gmres(vjp,b,max_iter=160,tol=tol,restart=16)
    assert torch.equal(x, long)
    assert longrel <= tol


def test_implicit_exact_parameter_and_input_gradients():
    a = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(2,3,4, dtype=torch.float64, requires_grad=True)
    cfg = SolverConfig(max_iter=100,tol=1e-11,backward_tol=1e-11)
    s, info = solve(lambda s: a*s+b,torch.zeros_like(b),cfg)
    s.square().sum().backward()
    exact = b.detach()/(1-a.detach())
    assert torch.allclose(b.grad,2*exact/(1-a),atol=1e-8,rtol=1e-8)
    assert torch.allclose(a.grad,2*exact.square().sum()/(1-a),atol=1e-8,rtol=1e-8)
    assert info.backward_residual < cfg.backward_tol
    assert info.backward_vjps > 0


def test_strict_forward_fails_and_no_invalid_gradient_fallback():
    cfg = SolverConfig(max_iter=1,tol=1e-12)
    b = torch.ones(1,2,3,requires_grad=True)
    with pytest.raises(ConvergenceError):
        solve(lambda s:.9*s+b,torch.zeros_like(b),cfg)
    with pytest.raises(ConvergenceError, match='Cannot attach'):
        solve(lambda s:.9*s+b,torch.zeros_like(b),cfg,strict=False)
    with torch.no_grad():
        _, info = solve(lambda s:.9*s+b,torch.zeros_like(b),cfg,strict=False)
    assert not info.converged.all()


def test_strict_backward_detects_bad_linear_solve():
    b = torch.randn(1,2,3,dtype=torch.float64,requires_grad=True)
    m = torch.diag(torch.linspace(.1,.6,6,dtype=torch.float64))
    fn=lambda s:(s.flatten(1)@m).reshape_as(s)+b
    cfg=SolverConfig(max_iter=120,tol=1e-10,backward_tol=1e-12,backward_max_iter=1)
    y,_=solve(fn,torch.zeros_like(b),cfg)
    with pytest.raises(ConvergenceError,match='Adjoint'):
        y.square().sum().backward()

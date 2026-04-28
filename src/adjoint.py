import torch

class _AdjointRefine(torch.autograd.Function):
    """Refine the phantom gradient by N Neumann terms.

    Forward: pass z_graph through unchanged.
    Backward: replace the incoming grad (dL/dz_graph) with a refined version
    from  λ ← J_f^T λ + g  for N steps starting from λ = g. This truncates
    (I − J_f^T)^{-1} g ≈ (I + J_f^T + J_f^{2T} + …) g.
    """

    @staticmethod
    def forward(ctx, z_graph, z_star, x, attn_mask, module, adjoint_steps, tol):
        ctx.save_for_backward(z_star, x)
        ctx.attn_mask = attn_mask
        ctx.module = module
        ctx.adjoint_steps = adjoint_steps
        ctx.tol = tol
        return z_graph

    @staticmethod
    def backward(ctx, grad_output):
        z_star, x = ctx.saved_tensors
        module = ctx.module
        attn_mask = ctx.attn_mask

        z_var = z_star.detach().requires_grad_(True)
        with torch.enable_grad():
            z_next = module._f(z_var, x.detach(), attn_mask)

        lam = grad_output.clone()
        for _ in range(ctx.adjoint_steps):
            vjp = torch.autograd.grad(
                outputs=z_next, inputs=z_var,
                grad_outputs=lam, retain_graph=True, create_graph=False,
            )[0]
            lam_new = vjp + grad_output
            rel = (lam_new - lam).norm() / (lam.norm() + 1e-12)
            lam = lam_new
            if rel.item() < ctx.tol:
                break

        # forward inputs were: (z_graph, z_star, x, attn_mask, module, adjoint_steps, tol)
        return lam, None, None, None, None, None, None


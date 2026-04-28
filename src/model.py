import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from torch.nn.utils.parametrizations import spectral_norm as sn_param
from .config import BertConfig
from .adjoint import _AdjointRefine

class VanillaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        dropout: float = 0.1,
        **_unused,   # swallow FPSA-only kwargs for interface compat
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_O = nn.Linear(hidden_size, hidden_size, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Stub stats (matches FPSA interface for logging)
        self._last_iterations = 1
        self._last_converged_frac = 1.0

    def _split_heads(self, x):
        B, N, d = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, H, N, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * dh)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        q = self._split_heads(self.W_Q(x))
        k = self._split_heads(self.W_K(x))
        v = self._split_heads(self.W_V(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        out = self.W_O(out)
        return self.out_dropout(out)

class FPSAAttention(nn.Module):
    """Multi-head self-attention with an inner fixed-point loop.

    Shape conventions (B=batch, N=seq, d=hidden, H=heads, dh=d/H):
        x : (B, N, d)  ->  z*: (B, N, d)

    The caller is responsible for the residual connection and layer norm that
    usually wrap attention: this module returns only z* (with output dropout).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        tol: float = 1e-4,
        max_iter: int = 16,
        implicit_grad: bool = True,
        adjoint_steps: int = 0,
        spectral_norm: bool = True,
        temperature: float = 1.0,
        selective_freeze: bool = True,
        damping: float = 1.0,
        skip_tol: float = 0.01,
        conv_exit_frac: float = 0.95,
        use_rope: bool = False,
        max_seq_len: int = 512,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.tol = tol
        self.max_iter = max_iter
        self.implicit_grad = implicit_grad
        self.adjoint_steps = adjoint_steps
        self.selective_freeze = selective_freeze
        self.skip_tol = skip_tol
        self.conv_exit_frac = conv_exit_frac
        if not (0.0 < damping <= 1.0):
            raise ValueError("damping must be in (0, 1]")
        self.damping = damping

        # Learnable per-head log-temperature (keeps tau > 0).
        self.log_tau = nn.Parameter(torch.full((num_heads,), math.log(temperature)))

        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_O = nn.Linear(hidden_size, hidden_size, bias=True)

        if spectral_norm:
            # Cap ||W_Q||, ||W_K||, ||W_O|| ≤ 1. W_V is intentionally not
            # spectrally constrained because it's outside the inner loop.
            self.W_Q = _maybe_spectral_norm(self.W_Q)
            self.W_K = _maybe_spectral_norm(self.W_K)
            self.W_O = _maybe_spectral_norm(self.W_O)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout_p = attn_dropout  # used for variational dropout inside the loop

        # Variational attention-probs dropout mask — sampled ONCE per forward,
        # held constant across ALL inner iterations. This provides the same
        # regularization as vanilla BERT's attention dropout while preserving
        # the fixed-point property (f stays deterministic during the loop).
        # See torchdeq's VariationalDropout convention: same mask across iters.
        self._attn_drop_mask: Optional[torch.Tensor] = None
        
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE(self.head_dim, max_seq_len)

        # Runtime statistics for analysis / logging
        self._last_iterations: Optional[int] = None
        self._last_converged_frac: Optional[float] = None

    # -------- shape plumbing --------
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * dh)

    # -------- the core map f(z; x) --------
    # The fixed point equation is z* = x + AttentionBlock(z*, x).
    # Inside _f we iterate on z (the full state with residual). The *output*
    # of forward() is the attention update (z* - x), so the caller can use
    # FPSA as a drop-in replacement for standard MHA: it returns the delta
    # that gets added to the residual stream.
    def _f(self, z: torch.Tensor, x: torch.Tensor,
           attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """One iteration of f(z; x) = x + AttentionBlock(z, x).

        The in-loop residual 'x +' prevents rank collapse: without it, pure
        iterated attention converges to a state where all tokens are the
        same vector, which zeroes the gradient to W_Q, W_K. With it, token
        identity is preserved through the loop and the fixed point is

            z* = x + AttentionBlock(z*, x)
        """
        q = self._split_heads(self.W_Q(z))                   # (B, H, N, dh)
        k = self._split_heads(self.W_K(z))                   # (B, H, N, dh)
        v = self._split_heads(self.W_V(x))                   # (B, H, N, dh)  from x

        if getattr(self, "use_rope", False):
            q, k = self.rope(q, k)

        tau = self.log_tau.exp().view(1, self.num_heads, 1, 1)
        scale = 1.0 / (math.sqrt(self.head_dim) * tau)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)

        # Variational attention-probs dropout: use the same mask every iteration.
        # self._attn_drop_mask is set by forward() once per forward call.
        if self.training and self._attn_drop_mask is not None:
            attn = attn * self._attn_drop_mask

        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        attn_update = self.W_O(out)                        # (B, N, d)

        if self.damping < 1.0:
            return (1.0 - self.damping) * z + self.damping * attn_update
        return attn_update

    # -------- convergence helper --------
    @staticmethod
    def _rel_residual(z_next: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """(B, N) per-token relative residual ||Δ|| / ||z||."""
        num = (z_next - z).norm(dim=-1)
        den = z.norm(dim=-1).clamp(min=1e-8)
        return num / den

    def _run_loop(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor],
                  record_grad: bool) -> torch.Tensor:
        """Run the fixed-point loop. If record_grad=False, runs under no_grad."""
        B, N, _ = x.shape
        ctx = torch.enable_grad() if record_grad else torch.no_grad()

        converged = None
        if self.selective_freeze:
            converged = torch.zeros(B, N, dtype=torch.bool, device=x.device)

        z = x
        final_iter = 0
        with ctx:
            for k in range(self.max_iter):
                z_next = self._f(z, x, attn_mask)
                rel = self._rel_residual(z_next, z)
                if converged is not None:
                    km = converged.to(z.dtype).unsqueeze(-1)        # (B, N, 1)
                    z_next = km * z + (1.0 - km) * z_next
                    converged = converged | (rel < self.tol)
                z = z_next
                final_iter = k + 1
                if converged is not None and converged.float().mean() >= self.conv_exit_frac:
                    break
                if converged is None and rel.mean().item() < self.tol:
                    break

        self._last_iterations = final_iter
        self._last_converged_frac = (
            converged.float().mean().item() if converged is not None else 1.0
        )
        return z

    # -------- forward --------
    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Adaptive-depth forward pass.

        1. Apply f once (one attention step from x). This is essentially
           vanilla attention — exact gradient, no iteration.
        2. Check per-token: did z change significantly from x?
           - Small change → token is already near its fixed point after 1 step.
             Use the single-step result. Exact gradient flows through this path.
           - Large change → this token benefits from iterative refinement.
             Run the full FPI loop. Implicit gradient flows through this path.
        3. Blend the two: easy tokens get vanilla-like exact gradients,
           hard tokens get the full FPSA treatment.

        Effect: the model behaves like a mixture of architectures per token.
        Easy tokens see a vanilla transformer. Hard tokens see an FPSA
        transformer with iterative refinement. The routing is determined
        by the attention operator itself, not a separate network.

        Returns:
            delta: (B, N, d) — the attention update to add to the residual stream.
        """
        B, N, _ = x.shape

        # Sample variational attention-dropout mask (same mask for all iterations)
        if self.training and self.attn_dropout_p > 0:
            keep_p = 1.0 - self.attn_dropout_p
            self._attn_drop_mask = (
                torch.empty(B, self.num_heads, N, N, device=x.device, dtype=x.dtype)
                .bernoulli_(keep_p) / keep_p
            )
        else:
            self._attn_drop_mask = None

        try:
            # ---- Step 1: one vanilla-like attention step (always with autograd) ----
            z_one = self._f(x, x, attn_mask)

            # ---- Step 2: per-token routing decision ----
            with torch.no_grad():
                change = self._rel_residual(z_one, x)       # (B, N)
                needs_fpi = change >= self.skip_tol          # (B, N) bool

            n_hard = needs_fpi.sum().item()
            n_total = B * N
            frac_hard = n_hard / n_total

            # Track stats
            self._last_fpi_frac = frac_hard

            if n_hard == 0:
                # All tokens converged in 1 step — pure vanilla behavior
                self._last_iterations = 1
                self._last_converged_frac = 1.0
                delta = z_one
                return self.dropout(delta)

            # ---- Step 3: run FPI for hard tokens ----
            use_implicit = self.implicit_grad and self.training and x.requires_grad

            if not use_implicit:
                # Eval or no-grad: just run the loop, full autograd or no grad
                z_star = self._run_loop(
                    x, attn_mask,
                    record_grad=x.requires_grad and self.training,
                )
            else:
                # Training: phantom-gradient path for FPI tokens
                with torch.no_grad():
                    z_star_detached = self._run_loop(
                        x, attn_mask, record_grad=False
                    ).detach()
                # One differentiable f application at z* for the backward graph
                z_star = self._f(z_star_detached, x, attn_mask)
                if self.adjoint_steps > 0:
                    z_star = _AdjointRefine.apply(
                        z_star, z_star_detached, x, attn_mask,
                        self, self.adjoint_steps, self.tol,
                    )

            # ---- Step 4: blend easy (exact grad) and hard (implicit grad) ----
            mask = needs_fpi.unsqueeze(-1).float()           # (B, N, 1)
            z_final = (1.0 - mask) * z_one + mask * z_star
            # For easy tokens: gradient flows through z_one → single f application → exact
            # For hard tokens: gradient flows through z_star → implicit diff → approximate

            delta = z_final
            return self.dropout(delta)

        finally:
            self._attn_drop_mask = None

class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE) for iterating attention architectures."""
    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
        
    def forward(self, q, k):
        N = q.shape[2]
        cos = self.cos_cached[:N].view(1, 1, N, -1)
        sin = self.sin_cached[:N].view(1, 1, N, -1)
        
        q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
        k1, k2 = k[..., :k.shape[-1]//2], k[..., k.shape[-1]//2:]
        
        q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q_rot, k_rot

class IteratedAttention(nn.Module):
    """Multi-head attention run T times. V static from input, Q/K from state."""

    def __init__(self, hidden_size: int, num_heads: int, T: int = 3,
                 dropout: float = 0.1, use_rope: bool = False, max_seq_len: int = 512):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.T = T

        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.W_O = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE(self.head_dim, max_seq_len)

    def _split_heads(self, t):
        B, N, _ = t.shape
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t):
        B, H, N, dh = t.shape
        return t.transpose(1, 2).contiguous().view(B, N, H * dh)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (B, N, d) — input to the attention sublayer (after LN)
            attn_mask: optional (B, 1, 1, N) additive mask, 0 or -inf
        Returns:
            delta: (B, N, d) — the attention output. Caller adds residual.
        """
        # V is computed once from input and held fixed
        v = self._split_heads(self.W_V(x))          # (B, H, N, dh)

        z = x                                        # z_0 = x
        for t in range(self.T):
            q = self._split_heads(self.W_Q(z))       # (B, H, N, dh) from current state
            k = self._split_heads(self.W_K(z))       # (B, H, N, dh) from current state

            if getattr(self, "use_rope", False):
                q, k = self.rope(q, k)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores + attn_mask

            attn = self.attn_drop(F.softmax(scores, dim=-1))
            out = self._merge_heads(torch.matmul(attn, v))
            z = self.W_O(out)                        # no inner residual (paper's formulation)

        return self.out_drop(z)

class LoopedEncoder(nn.Module):
    """Wrap a standard BertEncoder and iterate it to a fixed point."""

    def __init__(self, encoder, max_loops: int = 4, tol: float = 1e-2,
                 conv_exit_frac: float = 0.95, implicit_grad: bool = True):
        super().__init__()
        self.encoder = encoder           # a BertEncoder (list of BertLayer's)
        self.max_loops = max_loops
        self.tol = tol
        self.conv_exit_frac = conv_exit_frac
        self.implicit_grad = implicit_grad
        self._last_loops = 0
        self._last_converged_frac = 0.0

    def _one_pass(self, z, attn_mask):
        """One full pass through all encoder layers."""
        for layer in self.encoder.layers:
            z = layer(z, attn_mask)
        return z

    def _run_loop(self, x, attn_mask):
        """Run the encoder loop until convergence or max_loops."""
        B, N, _ = x.shape
        z = x
        converged = torch.zeros(B, N, dtype=torch.bool, device=x.device)

        for k in range(self.max_loops):
            z_next = self._one_pass(z, attn_mask)

            # Per-token convergence check
            rel = (z_next - z).norm(dim=-1) / z.norm(dim=-1).clamp(min=1e-8)
            newly_converged = rel < self.tol
            converged = converged | newly_converged

            # Selective freeze: keep converged tokens
            mask = converged.unsqueeze(-1).float()
            z_next = mask * z + (1.0 - mask) * z_next

            z = z_next
            self._last_loops = k + 1

            if converged.float().mean() >= self.conv_exit_frac:
                break

        self._last_converged_frac = converged.float().mean().item()
        return z

    def forward(self, x, attn_mask=None):
        use_implicit = self.implicit_grad and self.training and x.requires_grad

        if not use_implicit:
            z = self._run_loop(x, attn_mask)
            return self.encoder.final_ln(z)

        # Phantom gradient: run loop under no_grad, attach graph with one pass
        with torch.no_grad():
            z_star = self._run_loop(x, attn_mask).detach()

        # One differentiable pass at z* (analogous to FPSA's phantom gradient)
        z_graph = self._one_pass(z_star, attn_mask)

        return self.encoder.final_ln(z_graph)

    def attention_stats(self):
        return {
            "loops": self._last_loops,
            "converged_frac": self._last_converged_frac,
            # Delegate per-layer stats
            "iters_per_layer": [getattr(l.attention, '_last_iterations', None)
                                for l in self.encoder.layers],
            "converged_per_layer": [getattr(l.attention, '_last_converged_frac', None)
                                   for l in self.encoder.layers],
        }

class BertEmbeddings(nn.Module):
    """Token + position + (optional) token-type embeddings, LN, dropout."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id
        )
        self.use_rope = getattr(cfg, "use_rope", False)
        if not self.use_rope:
            self.position_embeddings = nn.Embedding(
                cfg.max_position_embeddings, cfg.hidden_size
            )
        self.token_type_embeddings = nn.Embedding(
            cfg.type_vocab_size, cfg.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(cfg.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N = input_ids.shape
        pos_ids = self.position_ids[:, :N]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        emb = (
            self.word_embeddings(input_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        if not self.use_rope:
            emb = emb + self.position_embeddings(pos_ids)
        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)
        return emb

class BertFeedForward(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.dense1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.dense2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense2(F.gelu(self.dense1(x))))

class BertLayer(nn.Module):
    """One transformer encoder layer with pre-LN topology.

    pre-attn LN  →  attention  →  + residual  →  pre-ffn LN  →  FFN  →  + residual
    """

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.attn_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attention = _build_attention(cfg)
        self.ffn_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn = BertFeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention sublayer
        h = x + self.attention(self.attn_ln(x), attn_mask)
        # FFN sublayer
        y = h + self.ffn(self.ffn_ln(h))
        return y

class BertEncoder(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.final_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.final_ln(x)

    def attention_stats(self) -> Dict[str, Any]:
        """Return per-layer FPSA statistics (iterations, converged fraction, FPI fraction)."""
        stats = {"iters_per_layer": [], "converged_per_layer": [], "fpi_frac_per_layer": []}
        for i, layer in enumerate(self.layers):
            stats["iters_per_layer"].append(
                getattr(layer.attention, "_last_iterations", None)
            )
            stats["converged_per_layer"].append(
                getattr(layer.attention, "_last_converged_frac", None)
            )
            stats["fpi_frac_per_layer"].append(
                getattr(layer.attention, "_last_fpi_frac", None)
            )
        return stats

class BertModel(nn.Module):
    """BERT base model: embeddings + encoder. Returns final hidden states.

    When cfg.looped_encoder is True, the encoder is wrapped in a fixed-point
    loop that iterates the entire layer stack until convergence. Each individual
    layer uses vanilla attention (exact gradients). The loop provides adaptive
    depth — easy tokens converge after 1 pass, hard tokens get more.
    """

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = BertEncoder(cfg)
        self._use_looped = cfg.looped_encoder
        if self._use_looped:
            self.looped = LoopedEncoder(
                self.encoder,
                max_loops=cfg.looped_max_loops,
                tol=cfg.looped_tol,
                conv_exit_frac=cfg.looped_conv_exit_frac,
                implicit_grad=cfg.looped_implicit_grad,
            )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.embeddings(input_ids, token_type_ids)
        mask = _make_attn_mask(attention_mask)
        if self._use_looped:
            h = self.looped(emb, mask)
        else:
            h = self.encoder(emb, mask)
        return h

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

class BertForMaskedLM(nn.Module):
    """MLM head tied to word embeddings (standard for BERT pre-training)."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.cfg = cfg
        self.bert = BertModel(cfg)
        self.transform = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.transform_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.decoder_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.bert(input_ids, attention_mask)
        h = F.gelu(self.transform(h))
        h = self.transform_ln(h)
        # Tie decoder to word embeddings
        logits = h @ self.bert.embeddings.word_embeddings.weight.T + self.decoder_bias

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

class BertForSequenceClassification(nn.Module):
    """Classification head that pools the first token (a la BERT's [CLS])."""

    def __init__(self, cfg: BertConfig, num_labels: int):
        super().__init__()
        self.cfg = cfg
        self.num_labels = num_labels
        self.bert = BertModel(cfg)
        self.pooler = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.classifier = nn.Linear(cfg.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.bert(input_ids, attention_mask)
        pooled = torch.tanh(self.pooler(h[:, 0]))
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        out: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out


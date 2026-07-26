# FPSA-BERT (Fixed-Point Self-Attention)

[![arXiv](https://img.shields.io/badge/arXiv-2507.13569-b31b1b.svg)](https://arxiv.org/abs/2507.13569)

**Official PyTorch implementation for the paper on arXiv: [2507.13569](https://arxiv.org/abs/2507.13569)**

This repository contains the PyTorch implementation of **Fixed-Point Self-Attention (FPSA)** for BERT.

FPSA is an iterative attention mechanism that mathematically mirrors the Self-Transformer framework. Instead of stacking $L$ separate self-attention layers with $O(L)$ parameters, FPSA drives a single attention block to convergence via a Fixed-Point Iteration (FPI) loop. It utilizes **Rotary Position Embeddings (RoPE)** to maintain spatial awareness across iterations and relies on an $O(1)$ memory **Phantom Gradient** (Neumann-RBP implicit differentiation) solver to backpropagate through the iterative convergence.


1. **Pretrained Compatibility:** `W_Q`, `W_K`, `W_O` are kept *inside* the FPI loop to correctly preserve the residual stream subspace, allowing seamless reuse of pretrained BERT weights.
2. **Fixed V:** Value projections `V = W_V(x)` are computed once statically outside the loop to prevent feature collapse.
3. **Implicit Gradients:** The model isolates hard-to-converge tokens and computes gradients analytically using the Adjoint method, drastically slashing VRAM usage compared to standard unrolling.

## FPSA-R: implicit differentiation for looped reasoning transformers

`src/fpsa_r/` extends this work into a **reasoning architecture**, combining it
with the looped-transformer recursion of
[FPRM](https://github.com/nilskiKonjIzDunava/fprm) and replacing FPRM's
truncated backpropagation-through-time with exact, constant-memory implicit
differentiation.

**Result.** On 7x7 maze planning at a forward budget of 32, dropping the
per-layer spectral caps and targeting the spectral radius directly (rho ~= 1
rather than safely below it), with Anderson acceleration for the forward solve
and GMRES for the adjoint:

| Configuration | Exact match | -> 9x9 | -> 11x11 | Act. mem | s/step |
| --- | --- | --- | --- | --- | --- |
| **no caps + Anderson fwd + GMRES adj.** | **95.6 ± 1.2** | **77.3** | **49.5** | **55 MB** | 0.52 |
| no caps + Anderson + GMRES + in-layer FPSA | 96.2 ± 0.7 | 71.1 | 41.4 | 59 MB | 0.57 |
| spectral caps + Picard + Anderson adj. | 81.4 ± 2.1 | 28.5 | 4.1 | 57 MB | 0.34 |
| FPRM (truncated BPTT) | 80.6 ± 1.5 | 26.8 | 2.5 | 108 MB | 0.36 |
| Looped Transformer (full BPTT) | 79.0 ± 1.0 | 29.5 | 4.9 | 595 MB | 3.10 |

Every row trains at rho 0.60-0.69, so this is a comparison at matched
contractivity. Against a fully-unrolled looped transformer that is **+16 points
of exact match and an order of magnitude better extrapolation, at 1/10th the
activation memory and 1/6th the time per step**.

**The contractivity question.** The implicit gradient is only the gradient at an
equilibrium, so the loop must contract -- and that requirement, not the gradient,
is what costs accuracy. Two findings separate the concerns:

- **rho < 1 is genuinely required.** Against exact BPTT the implicit gradient is
  essentially exact at rho = 0.93 (cosine 0.999999) and *uninformative* at
  rho = 1.13 (cosine -0.02). It does not degrade gracefully. Stronger solvers
  will find fixed points past rho = 1, but the gradient there is still worthless.
- **Per-layer spectral caps are not required.** At matched rho the gradient is
  equally faithful with and without them. They are a conservative sufficient
  condition for what the spectral-radius penalty already enforces directly, and
  removing them is worth the 14 points above.

**The in-layer FPSA loop.** Under the capped recipe it costs about ten points --
but that turns out to be the constraint interacting badly with the second loop,
not the loop itself. With the caps removed the two are within noise in
distribution (96.2 ± 0.7 with it, 95.6 ± 1.2 without), though extrapolation is
still somewhat worse with it. Neutral to slightly negative on this task, not the
ten-point penalty the capped comparison suggested.

**What did not work.** An early no-caps run *without* a spectral-radius target
reached rho = 1.85 with a diverging forward pass. It scored 92.7, but was a
weight-tied deep network with an arbitrary gradient rather than an equilibrium
model -- the rho target is what makes the difference between that and the result
above. Reported in full rather than omitted.

Mechanism results, which are properties of the differentiation scheme rather
than the task:

- the O(1)-memory gradient tracks exact deep BPTT (cosine 0.99998 in the
  contractive regime) and beats truncated BPTT at every forward budget tested
- activation memory is flat in loop depth: **83x** below full BPTT and **4.1x**
  below FPRM's truncated BPTT at 128 iterations
- the joint two-level solve reaches the same equilibrium as nesting the inner
  loop with **2.0x fewer attention calls**
- Anderson mixing solves the adjoint in **1.5x-3.1x fewer VJPs** than a Neumann
  series, with the gap widening as the spectral radius approaches 1

See **[docs/FPSA-R.md](docs/FPSA-R.md)** for the full method, figures, tables
and scope limits.

```bash
python experiments/reasoning/mechanism.py   # mechanism experiments (minutes, CPU)
./scripts/run_all_fpsa_r.sh                 # full comparison grid (~3h, 4 CPU cores)
python experiments/reasoning/analyze.py     # tables + figures
python experiments/reasoning/make_report.py # rebuild docs/FPSA-R.md
python -m pytest tests/ -q                  # invariants the method rests on
```

## Directory Structure

- `src/config.py`: Architecture and training configurations.
- `src/model.py`: Core components (`FPSAAttention`, `IteratedAttention`, `RoPE`, `BertEncoder`).
- `src/adjoint.py`: Custom PyTorch autograd function for Neumann-RBP implicit gradients.
- `src/fpsa_r/`: FPSA-R — joint two-level equilibrium, implicit differentiation, contraction control.
- `experiments/reasoning/`: reasoning benchmarks, unified trainer, mechanism suite, analysis.
- `src/data_utils.py`: Datasets for WikiText-2 (MLM) and GLUE classification tasks.
- `scripts/run_all.py`: Full end-to-end execution script for pretraining and fine-tuning sweeps.

## Stage 1 Results (WikiText-2 -> GLUE-4)

At ~10M parameters (4L x 256), the FPSA architecture achieved:
- **Pretraining**: 71.7 Perplexity (Vanilla: 221.6) — a **0.32x** gap.
- **Fine-Tuning**: +0.81 percentage points ahead of Vanilla BERT on average across 4 downstream GLUE tasks.

| task | metric | vanilla | fpsa | gap(pp) |
| :--- | :--- | :--- | :--- | :--- |
| sst2 | accuracy | 79.24 | **80.35** | +1.11 |
| mrpc | accuracy | **70.75** | 70.59 | -0.16 |
| rte | accuracy | **56.32** | 53.79 | -2.53 |
| cola | matthews | 10.16 | **14.99** | +4.83 |
| **avg** | | 54.12 | **54.93** | **+0.81** |

## Next Steps
- Stage 2: WikiText-103 scaling (30M parameters).
- Stage 3: Full BookCorpus + Wikipedia pretraining (110M parameters).

## Citation
If you use this code or find our work helpful, please cite:
```bibtex
@article{mathur2025change,
  title={Change of Thought: Adaptive Test-Time Computation},
  author={Mathur, Mrinal and Doan, Mike and Pearlmutter, Barak and Plis, Sergey},
  journal={arXiv preprint arXiv:2507.13569},
  year={2025}
}
```

# FPSA-BERT (Fixed-Point Self-Attention)

This repository contains the PyTorch implementation of **Fixed-Point Self-Attention (FPSA)** for BERT.

FPSA is an iterative attention mechanism that mathematically mirrors the Self-Transformer framework. Instead of stacking $L$ separate self-attention layers with $O(L)$ parameters, FPSA drives a single attention block to convergence via a Fixed-Point Iteration (FPI) loop. It utilizes **Rotary Position Embeddings (RoPE)** to maintain spatial awareness across iterations and relies on an $O(1)$ memory **Phantom Gradient** (Neumann-RBP implicit differentiation) solver to backpropagate through the iterative convergence.

## Architecture

![BERT-FPSA Architecture](./bert_fpsa_architecture_v2.svg)

1. **Pretrained Compatibility:** `W_Q`, `W_K`, `W_O` are kept *inside* the FPI loop to correctly preserve the residual stream subspace, allowing seamless reuse of pretrained BERT weights.
2. **Fixed V:** Value projections `V = W_V(x)` are computed once statically outside the loop to prevent feature collapse.
3. **Implicit Gradients:** The model isolates hard-to-converge tokens and computes gradients analytically using the Adjoint method, drastically slashing VRAM usage compared to standard unrolling.

## Directory Structure

- `src/config.py`: Architecture and training configurations.
- `src/model.py`: Core components (`FPSAAttention`, `IteratedAttention`, `RoPE`, `BertEncoder`).
- `src/adjoint.py`: Custom PyTorch autograd function for Neumann-RBP implicit gradients.
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

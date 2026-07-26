**maze: mean ± sd over seeds. Activation memory is bytes autograd stores for one training step, and here *includes* the contraction regulariser's two extra single-step graphs for the fixed-point architectures -- see the mechanism table for the differentiation scheme in isolation. rho is the measured spectral radius of the update map at the solution.**

| Model | Params | Token acc (%) | Exact match (%) | Act. mem (MB) | s / step | Eval iters | rho | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FPSA-R (+ in-layer FPSA) | 135558 | 97.41 ± 0.33 | 62.11 ± 0.55 | 60.5 | 0.487 | 32.0 | 0.88 | 2 |
| FPSA-R, nested solver | 135558 | 98.17 ± 0.17 | 74.12 ± 2.35 | 75.5 | 0.552 | 30.1 | 0.87 | 2 |
| FPSA-R, BPTT | 135558 | 97.99 ± 0.37 | 68.26 ± 4.01 | 191.3 | 0.567 | 32.0 | 0.87 | 2 |
| FPSA-R, 1-step phantom | 135558 | 95.33 ± 0.12 | 31.64 ± 1.66 | 58.9 | 0.256 | 31.0 | 0.81 | 2 |
| FPSA-R, unmasked adjoint | 135558 | 97.62 ± 0.31 | 63.48 ± 0.28 | 60.5 | 0.488 | 32.0 | 0.93 | 2 |
| FPSA-R, Neumann adjoint | 135558 | 96.79 ± 0.66 | 59.96 ± 6.63 | 60.5 | 0.443 | 32.0 | 0.95 | 2 |
| **FPSA-R, no spectral norm** | 135558 | 99.21 ± 0.29 | 92.68 ± 4.56 | 59.1 | 0.470 | 32.0 | 1.85 | 2 |

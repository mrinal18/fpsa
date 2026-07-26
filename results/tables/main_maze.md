**maze: mean ± sd over seeds. Activation memory is bytes autograd stores for one training step, and here *includes* the contraction regulariser's two extra single-step graphs for the fixed-point architectures -- see the mechanism table for the differentiation scheme in isolation. rho is the measured spectral radius of the update map at the solution.**

| Model | Params | Token acc (%) | Exact match (%) | Act. mem (MB) | s / step | Eval iters | rho | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FPSA-R (+ in-layer FPSA) | 135558 | 97.41 ± 0.23 | 63.54 ± 2.51 | 60.5 | 0.437 | 32.0 | 0.92 | 3 |
| spectral caps + Picard + Anderson adj. | 135558 | 97.80 ± 0.18 | 75.85 ± 1.18 | 56.6 | 0.324 | 32.0 | 0.97 | 3 |
| FPRM (truncated BPTT) | 135558 | 94.13 ± 6.56 | 53.26 ± 46.12 | 108.0 | 0.756 | 31.9 | 0.88 | 3 |
| **Looped Transformer (BPTT)** | 135558 | 98.05 ± 0.31 | 78.91 ± 0.34 | 177.6 | 0.502 | 31.7 | 0.87 | 3 |
| Universal Transformer + ACT | 135558 | 95.57 ± 0.78 | 38.22 ± 9.72 | 147.6 | 0.380 | 32.0 | 0.40 | 3 |
| Transformer (non-recursive) | 1067426 | 94.60 ± 6.97 | 52.15 ± 45.23 | 134.8 | 0.361 | 1.0 | 0.07 | 3 |

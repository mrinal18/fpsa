**maze7 at a forward budget of 32. 'rho (train)' is the spectral radius the contraction regulariser holds the map at during training; every row here sits below 1, where the implicit gradient is faithful (see the faithfulness table).**

| Configuration | Exact match (%) | -> 9x9 | -> 11x11 | Act. mem (MB) | s / step | rho (train) | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Looped Transformer (BPTT) | 79.00 ± 0.97 | 29.5 | 4.9 | 595 | 3.10 | 0.63 | 2 |
| spectral caps + Picard + Anderson adj. | 81.35 ± 2.07 | 28.5 | 4.1 | 57 | 0.34 | 0.67 | 2 |
| spectral caps + Picard + GMRES adj. | 81.25 ± 2.76 | 30.9 | 4.1 | 57 | 0.52 | 0.67 | 2 |
| spectral caps + Anderson fwd + GMRES | 78.03 ± 1.52 | 25.8 | 2.3 | 57 | 0.54 | 0.64 | 2 |
| **no caps + Anderson + GMRES + in-layer FPSA** | 96.19 ± 0.69 | 71.1 | 41.4 | 59 | 0.57 | 0.62 | 2 |
| no caps + Anderson fwd + GMRES (ours) | 95.61 ± 1.20 | 77.3 | 49.5 | 55 | 0.52 | 0.60 | 4 |

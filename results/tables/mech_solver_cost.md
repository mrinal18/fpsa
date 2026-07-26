**Cost of reaching residual < 0.001 on Sudoku (batch 32). Both solvers reach the same equilibrium; the joint formulation gets there with 2.3x fewer attention calls and 1.53x less wall clock.**

| Two-level solver | outer iters | attention calls | ms / forward | final residual |
| --- | --- | --- | --- | --- |
| Nested (inner loop inside each outer step) | 5 | 16 | 409 | 5.1e-04 |
| **Joint (ours)** | 7 | **7** | **267** | 4.8e-04 |

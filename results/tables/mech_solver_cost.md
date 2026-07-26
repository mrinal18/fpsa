**Cost of reaching residual < 0.001 on Sudoku (batch 32). Both solvers reach the same equilibrium; the joint formulation gets there with 2.0x fewer attention calls and 1.26x less wall clock.**

| Two-level solver | outer iters | attention calls | ms / forward | final residual |
| --- | --- | --- | --- | --- |
| Nested (inner loop inside each outer step) | 5 | 14 | 261 | 3.5e-04 |
| **Joint (ours)** | 7 | **7** | **207** | 3.1e-04 |

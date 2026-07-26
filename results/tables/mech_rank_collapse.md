**Effective rank (entropy of the singular-value spectrum) of the converged inner attention state, 5 random inits, 40 iterations. Without the input re-injection the row-stochastic attention operator averages tokens together and the fixed point loses 66% of its effective rank.**

| Inner FPSA map | effective rank of the fixed point | of max |
| --- | --- | --- |
| **u <- x + W_O A(u) V**  (FPSA-R) | **25.6 ± 0.4** | 49 |
| u <- W_O A(u) V  (no in-loop residual) | 8.7 ± 0.4 | 49 |

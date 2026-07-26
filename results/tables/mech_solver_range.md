**Residual reached by each forward solver in 64 steps, and adjoint error reached by each backward solver in 30 VJPs, as the spectral radius is swept through 1. Mean over 2 inits.**

| rho | fwd: Picard | fwd: Broyden | fwd: Anderson | bwd: Neumann | bwd: Anderson | bwd: GMRES |
| --- | --- | --- | --- | --- | --- | --- |
| 0.55 | 8.1e-05 | 5.9e-05 | **6.6e-05** | 2.5e-08 | 6.2e-08 | **4.9e-08** |
| 0.75 | 8.8e-05 | 6.7e-05 | **7.0e-05** | 8.4e-06 | 1.8e-07 | **1.0e-07** |
| 0.96 | 1.1e-02 | 1.7e-01 | **3.9e-03** | 8.6e-03 | 7.7e-03 | **8.3e-03** |
| 1.09 | 1.1e-01 | 6.4e-01 | **1.5e-02** | 9.0e-02 | 2.5e-02 | **2.6e-02** |
| 1.23 | 2.4e-01 | 1.8e-01 | **5.5e-02** | 2.3e-01 | 6.3e-02 | **6.2e-02** |
| 1.30 | 3.0e-01 | 4.9e-01 | **6.0e-02** | 3.0e-01 | 9.2e-02 | **8.2e-02** |
| 1.35 | 3.6e-01 | 3.7e-01 | **2.1e-01** | 3.5e-01 | 1.0e-01 | **9.8e-02** |

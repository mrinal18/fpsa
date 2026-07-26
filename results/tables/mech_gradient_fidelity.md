**Fidelity against exact BPTT through 64 unrolled steps, forward budget T=12, 5 random inits.**

| Gradient scheme | cosine vs exact | relative error |
| --- | --- | --- |
| **FPSA-R implicit (Anderson)** | 0.999979 ± 0.000036 | 0.0055 ± 0.0043 |
| FPSA-R implicit (Neumann) | 0.999979 ± 0.000036 | 0.0055 ± 0.0043 |
| 1-step phantom gradient | 0.995795 ± 0.000400 | 0.1004 ± 0.0055 |
| truncated BPTT  K=1 | 0.995807 ± 0.000400 | 0.1003 ± 0.0055 |
| truncated BPTT  K=2 | 0.999809 ± 0.000021 | 0.0213 ± 0.0013 |
| truncated BPTT  K=4 | 0.999978 ± 0.000038 | 0.0057 ± 0.0044 |
| truncated BPTT  K=6 | 0.999975 ± 0.000039 | 0.0061 ± 0.0044 |
| full BPTT       T=12 | 0.999975 ± 0.000039 | 0.0061 ± 0.0044 |

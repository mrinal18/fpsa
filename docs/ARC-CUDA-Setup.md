# CUDA preflight for the official ARC recipe

The standalone model, solver tests, data builder and scoring code can run on CPU.
The full pinned TRM trainer uses a fused AdamATan2 CUDA optimizer. A successful
`pip install adam-atan2` does not prove its CUDA extension was compiled: the
Python-only wheel can be installed while `adam_atan2_backend` is absent.

Install the PyTorch build appropriate to the GPU runtime first. Then:

```bash
python -m pip install -r experiments/arc/requirements.txt
# Requires a working CUDA toolkit/nvcc and Python/CUDA headers.
python -m pip install --no-build-isolation --no-cache-dir --force-reinstall adam-atan2==0.0.3
python -m experiments.arc.preflight
```

`preflight` imports the actual backend and performs one small fused optimizer
update. The full launcher calls this before starting either training arm. It
does not silently substitute AdamW or another implementation. Missing CUDA or
a missing/incompatible extension produces an actionable error before expensive
training starts. GPU driver, toolkit, PyTorch and extension compatibility must be
established on the target machine.

CPU CI composes the actual official Hydra configurations without importing the
GPU-only optimizer. It exercises the real sparse embedding optimizer, EMA,
ACT carry, official ARC data-builder split boundary and numerical model code.
It does NOT execute the fused CUDA optimizer or a full GPU ARC training run.
The GPU preflight and full benchmark remain user-side execution requirements.

## Start small before allocating the reference training budget

The reference global batch is 768. The FPSA prototype uses explicit attention
and FP32 equilibrium states, which have a different memory footprint from TRM's
BF16 implementation. Do not assume that the reference batch fits a single Colab
GPU. An initial small-batch GPU run is an engineering test, not a resource-matched
ARC comparison. Record every override in the experiment manifest.

The CUDA optimizer package is the pinned `adam-atan2==0.0.3` used by the upstream
recipe. The project does not redistribute a compiled optimizer binary.

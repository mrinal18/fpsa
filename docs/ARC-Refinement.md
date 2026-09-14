# FPSA-R ARC recipe and answer-conditioned refinement

**Status: implementation and numerical tests; no new ARC benchmark score.**
The user's 19.0% / 6.34M experiment did not include its exact ARC adapter,
configuration, checkpoint, or evaluator. This branch does not claim to reproduce
that run or to outperform TRM. It starts from `looped_fpsa` commit
`430160fe1b8a6bdf1e6217a64f7de75801b66c89` and leaves `src/fpsa_r` untouched.

## What changed

New code in `src/fpsa_arc` supplies:

* An adapter for the existing FPSA-R joint operator using audited forward and
  backward solvers (`fpsa_arc_legacy`). It solves once, not sixteen identical
  equilibria, and remains a block-plus-attention model.
* Answer-conditioned refinement (`fpsa_arc_refine`). Each of two stages solves
  an attention equilibrium, then executes one MLP. The answer and latent carry
  survive to the next supervised segment, but are detached between segments.
* Explicit dual-value, finite-unroll, full-block, and plain block-DEQ controls.
* Per-example StableMax loss, a correctness-trained halt head, strict numerical
  checks, and separate recurrent-core/task-table parameter reporting.

`experiments/arc` connects these models to the **actual pinned TRM trainer**,
not an approximation of its optimizer, data sampler, sparse-embedding updates,
EMA implementation, or augmentation builder. The reference is
`SamsungSAILMontreal/TinyRecursiveModels` commit
`c01103738605ba39d1430519b1ee0c62f4c707f8` (MIT).

## Architecture and token/compute budget

For one grid, the encoded evidence is X and the current answer state is Y_s.
At stage l of refinement segment s:

```
anchor = RMSNorm(X + Y_s)                   # fixed during this solve
V_e = W_Ve RMSNorm(X)                       # evidence values, computed once
R* = Phi(R*; anchor, V_e)                    # recurrent attention ONLY
Y_s = RMSNorm(Y_s + R*)
Y_s = RMSNorm(Y_s + MLP_l(Y_s))              # once after the solve
```

Phi normalizes `anchor + R`, recomputes Q/K and RoPE, applies normalized-dot-
product attention, and projects the retrieval. The optional `dual` configuration
adds dynamic values from `anchor + R`; it is a distinct mechanism, not fixed-V
FPSA. The feedback output gain is bounded but this is NOT a contraction proof.

At the end of a segment, decode the current answer and supervise it. The next
segment uses the updated answer in a different conditional fixed-point problem.
This is not repeated solving of the same input-only equilibrium. Fixed sinusoidal
prefix anchors prevent zero-initialized/padded task slots from creating
zero-direction cosine queries. They add no learned parameters; this initialization
issue was caught by the actual upstream sparse-embedding integration test.

Inputs use the official 30x30 canvas: **900 grid positions + 16 prefix positions
= 916 positions**. Colors 0..9 are token IDs 2..11, PAD is 0, and EOS is 1.
Prefix capacity is 16 vectors, but the official default learned task embedding
is only 512 scalars and is padded into that capacity. Recurrence does NOT append
new text tokens. There is no CoT target: output-grid loss and halt supervision
train every answer-refinement segment.

Defaults are two stages, up to 16 segments, an inner train cap of 48 and eval cap
of 64, forward tolerance 1e-4 and adjoint tolerance 1e-5. A converged inner solver
stops early; count actual map calls, including final residual checks and the
extra differentiable map evaluation. At full evaluation there are 32 MLP calls
in the attention-only variant, but a variable number of attention calls. Dense
halting freezes states; it does not compact matrices or grant proportional GPU
savings. Backward VJPs and backward map rebuilds are recorded separately.

## Parameters measured from source

At width 512, eight heads, two stages, and expansion 4:

| Core | Learned core scalars (task table excluded) |
|---|---:|
| Answer-conditioned FPSA | 6,337,043 |
| Existing FPSA-R joint operator in the new adapter | 6,338,578 |
| Official pinned TRM attention model | 6,829,058 |

Task-table learned storage is `num_puzzle_identifiers * puzzle_emb_ndim` for
EVERY model and is reported separately. TRM stores the table in buffers and
updates it with sparse SignSGD; omitting it from `parameters()` does not make
that learned storage free. EMA follows upstream and covers dense parameters,
not the task table. These are parameter-matched-scale comparisons, not claims
of equal training/inference FLOPs.

## Install

From the new branch:

```bash
python -m pip install -r experiments/arc/requirements.txt
# Use the PyTorch build appropriate to your CPU/CUDA runtime; it is not replaced.
python -m experiments.arc.smoke --steps 40 --output results/arc_smoke
```

The smoke task is a synthetic 3x3 color permutation. Its loss and residuals only
validate plumbing, not ARC generalization. Run tests with:

```bash
python -m pytest tests/test_arc_*.py tests/test_fpsa_r.py -q
```

To run tests against upstream modules too, set `FPSA_TRM_REFERENCE` to the pinned
checkout. CI checks out that reference and runs the official-builder leakage
boundary, actual sparse optimizer, carry, and codec compatibility tests.

## Exact official data builder

```bash
python -m experiments.arc.prepare \
  --arc-version 1 --num-aug 1000 \
  --output data/arc1concept-aug-1000
```

This creates `.external/trm`, verifies its commit and clean tracked files, and
calls its `dataset.build_arc_dataset` unchanged. Its bundled ARC/concept files
are the default input source; `--input-prefix` selects an explicit alternative.
The generated manifest records source hashes, split names, seed, and upstream
commit. Never mix ARC-2 training into a claimed ARC-1 evaluation: the official
repository warns of overlap.

The official task-demonstration adaptation protocol trains on the available
example pairs of evaluation tasks, but routes their query outputs to scoring.
This is NOT a frozen-model/unseen-task experiment. Our builder test checks that
held-out query outputs do not enter training labels. Reusing the official
builder does not independently certify every possible transductive information
flow (its augmentation duplicate checks also access its converted examples).

## Recipe comparison before architecture claims

First print the exact command and planned optimizer updates without launching:

```bash
python -m experiments.arc.run \
  --data data/arc1concept-aug-1000 --arch fpsa_arc_legacy \
  --run-dir results/arc_legacy_s0 --seed 0 --devices 4 --dry-run
```

Then remove `--dry-run`. Every run uses a new output directory. Repeat with
`--arch trm` under the same data, batch, epochs, evaluation cadence, and voting
policy. The reference ARC recipe uses 768 global batch, 100000 schedule epochs,
2000 warmup updates, backbone LR 1e-4, weight decay .1, dense AdamATan2, task LR
.01/sparse SignSGD, and EMA .999. These settings are read from the pinned upstream
config. The runner supplies the README's TRM H_cycles=3/L_cycles=4 override.
Schedule epochs are the upstream group-sampling units, not ordinary complete
passes over all augmented arrays.

The launcher uses torchrun even for one GPU so the official ARC evaluator has
an initialized process group.

The default `--voting checkpoint` deliberately resets votes for each evaluation.
Use `--voting history` to reproduce upstream's across-evaluation accumulation;
then the candidate pool includes multiple checkpoints, so matching evaluation
cadence is essential. Neither score should be called a reproduced 44.6% until
that result has actually been obtained under the stated policy.

Smaller GPU experiments can override `--batch-size 32 --devices 1`, but they
are different training-budget runs. The official reference is a substantial GPU
experiment, not a claim that full ARC training will fit a free Colab session.

## Answer-refinement experiment

```bash
python -m experiments.arc.run \
  --data data/arc1concept-aug-1000 --arch fpsa_arc_refine \
  --run-dir results/arc_refine_s0 --seed 0 --devices 4
```

Other named architectures:

| Configuration | Meaning |
|---|---|
| fpsa_arc_legacy | Existing joint FPSA-R operator, one audited equilibrium |
| fpsa_arc_deq | Existing block operator without FPSA, one audited equilibrium |
| fpsa_arc_refine | Fixed-evidence attention equilibrium + one MLP per stage |
| fpsa_arc_dual | As above with dynamic scratch-value heads |
| fpsa_arc_bptt | Finite six-step attention trajectory; same train/eval depth |
| fpsa_arc_block | Same stage parameters, MLP inside each recurrent map |
| trm | Official TRM code, unchanged |

Additional Hydra overrides are explicit and recorded, e.g.
`--override arch.stability_weight=0.1`. The opt-in regularizer uses exact JVP
probes, NOT a certified spectral norm. A few probe directions can miss unstable
modes. It is off for the recipe baseline; do not claim that a soft penalty
ensures convergence. Conditional implicit solves and a BPTT warm-start are
separate experiments and must be labeled as such.

## Intentional differences from original TRM execution

* FPSA equilibrium state and small linear solves use FP32/FP64 rather than the
  TRM BF16 state. This favors numerical auditing and costs memory/time.
* Compilation is disabled for BOTH arms while auditing dynamic solver loops.
* The loss has the same per-example normalization, but safely evaluates the
  unused StableMax reciprocal branch and excludes blank padded rows from the
  halt loss. It is not textually identical to upstream's loss implementation.
* We use the actual upstream training-time ACT/carry convention and always run
  the configured number of segments at evaluation. The halt head is trained
  confidence, not proof that an unseen answer is correct. Residual thresholds
  govern inner numerical acceptance, not answer validity.
* The joint legacy adapter replaces only its numerical solves and compact task
  embedding plumbing; it is not a reproduction of the unseen 19% ARC adapter.

## Diagnostics, failure behavior, and checkpoints

`manifest.json` records commits, command, seed, dtype and voting policy.
`console.log` receives merged stdout/stderr live; exceptions print its tail.
`diagnostics/numerics_rank*.jsonl` records the actual forward residuals, NFEs,
adjoint VJP count and adjoint residual after optimizer updates. There are no
silent cap escalations or invalid-gradient fallbacks in this branch.

On a numerical failure, save a diagnostic checkpoint and STOP. This is useful
for investigating a failure, not evidence of robustness on the full benchmark.
The instrumentation also saves raw weights beside upstream's EMA checkpoints.
Failure snapshots may contain optimizer/carry state, but the sampler/prefetch
state is not serialized: **bit-exact resume is not implemented**.

## Candidate evaluation without target leakage

All runs save official-style `inputs`, `preds`, `puzzle_identifiers`, and
`q_halt_logits`. Export one checkpoint/rank:

```bash
python -m experiments.arc.export_predictions \
  --predictions results/arc_refine_s0/step_XXXX_all_preds.0 \
  --identifiers data/arc1concept-aug-1000/identifiers.json \
  --challenges .external/trm/kaggle/combined/arc-agi_evaluation_challenges.json \
  --source refine_s0_stepXXXX_rank0 --output results/candidates.jsonl

python -m experiments.arc.evaluate \
  --predictions results/candidates.jsonl \
  --solutions .external/trm/kaggle/combined/arc-agi_evaluation_solutions.json \
  --output results/arc_scores.json
```

Use `--canonical-only` for canonical predictions. The scorer reports top-one,
top-two, and candidate oracle coverage separately. It matches the task-macro
convention: average query accuracy inside a task, then average original tasks.
Missing predictions score zero; duplicate candidate events raise an error.
Rank by vote count, then mean learned confidence. Labels enter only the scorer,
never the ranker; oracle coverage cannot select submitted candidates. No
unvalidated residual penalty is added to candidate ranking.

## Acceptance gates

1. Numerical, carry, codec, loss and evaluator regressions pass.
2. Reproduce a TRM reference through the pinned recipe; audit candidate budgets.
3. Evaluate the existing FPSA checkpoint once its adapter/config is available.
4. Separate recipe gain from answer-refinement gain and from candidate voting.
5. Run multi-seed ARC evaluation and actual compute/latency measurements before
   claiming to beat TRM or calling the method state of the art.

Sources: upstream README, config/cfg_pretrain.yaml, config/arch/trm.yaml,
dataset/build_arc_dataset.py, puzzle_dataset.py, evaluators/arc.py,
models/recursive_reasoning/trm.py, models/sparse_embedding.py, models/losses.py,
and models/ema.py at the pinned commit. See `third_party/TRM_LICENSE.txt`.

### Generate a new bounded candidate pool

`predict.py` supports checkpoints produced by this branch and the official TRM
trainer. It is not a universal loader for the unknown 19% notebook checkpoint.
It reads prepared test INPUTS and identifiers, never prepared query labels.

```bash
python -m experiments.arc.predict \
  --data data/arc1concept-aug-1000 \
  --checkpoint results/arc_refine_s0/step_XXXX \
  --config results/arc_refine_s0/all_config.yaml \
  --challenges .external/trm/kaggle/combined/arc-agi_evaluation_challenges.json \
  --augmentations 8 --batch-size 8 --device cuda \
  --source refine_s0_stepXXXX_k8 --output results/refine_k8.jsonl
```

Use `--augmentations 1` for canonical single-trajectory predictions, then 8, 32,
or 100 for explicit breadth studies. Variants are selected before inference.
Do not concatenate overlapping candidate budgets or duplicate the same run to
inflate votes. A budget sidecar records candidate events, segments, elapsed time
and known forward map counts. Missing TRM map counts are null, not invented.

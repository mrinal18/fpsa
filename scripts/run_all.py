import torch
import time
import math
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.config import BertConfig, TrainConfig
from src.model import BertForMaskedLM, BertForSequenceClassification
from src.data_utils import MLMDataset, GlueDataset



MLM_BLOCK = 128
mlm_train = MLMDataset(train_tokens, block_size=MLM_BLOCK, seed=0)
mlm_val   = MLMDataset(val_tokens,   block_size=MLM_BLOCK, seed=999)
print(f"MLM train chunks: {len(mlm_train)}  (block_size={MLM_BLOCK})")
print(f"MLM val chunks:   {len(mlm_val)}")

# Pretraining loop
import time
import math


def pretrain_mlm(model_type="vanilla", epochs=30, batch_size=64, lr=5e-4,
                 num_layers=4, hidden_size=256, num_heads=4,
                 fpsa_max_iter=20, fpsa_adjoint=4,
                 device=None, log_every=200):
    """Pretrain a BERT model with MLM.

    model_type: "vanilla" or "fpsa"
      - vanilla: standard BERT, single-pass attention
      - fpsa: Fixed-Point Self-Attention (iterate Q/K from evolving state,
              V from input, converge to fixed point, phantom gradient backward)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=MLM_BLOCK + 8,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        attention_type=model_type,          # "vanilla" or "fpsa"
        fpsa_max_iter=8,
        fpsa_adjoint_max_iter=fpsa_adjoint,
        fpsa_tol=1e-3,
        fpsa_spectral_norm=False,
        fpsa_selective_freeze=True,
        fpsa_implicit_grad=True,
        fpsa_damping=0.5,
        fpsa_skip_tol=0.0,                 # disabled: always run FPI
        fpsa_conv_exit_frac=0.80,           # exit when 80% converge
        use_rope=(model_type != "vanilla"),
    )
    model = BertForMaskedLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label = f"{model_type}-{num_layers}L"
    print(f"=== Pretraining {label} ({n_params:,} params) ===")

    train_loader = DataLoader(mlm_train, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(mlm_val, batch_size=batch_size*2,
                            shuffle=False, num_workers=2, pin_memory=True)

    total_steps = len(train_loader) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                             betas=(0.9, 0.98), eps=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=0.06,
        anneal_strategy="linear", cycle_momentum=False,
    )

    # Diagnostics: track loop counts throughout training
    loop_history = []  # list of (step, loops, conv_frac)

    model.train()
    step = 0
    running_loss = 0.0
    t0 = time.time()
    for ep in range(epochs):
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(ids, mask, labels=labels)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            running_loss += out["loss"].item()
            step += 1

            if step % log_every == 0:
                avg = running_loss / log_every
                ppl = math.exp(min(avg, 20))
                elapsed = time.time() - t0
                diag = ""
                if model_type == "fpsa":
                    stats = model.bert.encoder.attention_stats()
                    iters = stats.get("iters_per_layer", [])
                    conv = stats.get("converged_per_layer", [])
                    if iters and iters[0] is not None:
                        iters_str = "/".join(str(x) for x in iters)
                        conv_str = "/".join(f"{c:.2f}" for c in conv if c is not None)
                        diag = f"  iters={iters_str}  conv={conv_str}"
                print(f"  ep{ep+1}/{epochs} step {step:5d}/{total_steps} "
                      f"loss={avg:.3f} ppl={ppl:.1f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"({elapsed:.0f}s){diag}")
                running_loss = 0.0

    # Final validation
    model.eval()
    val_loss = 0.0; val_n = 0
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(ids, mask, labels=labels)
            val_loss += out["loss"].item() * ids.shape[0]
            val_n += ids.shape[0]
    val_loss /= max(val_n, 1)
    val_ppl = math.exp(min(val_loss, 20))
    print(f"  FINAL val_loss={val_loss:.3f}  val_ppl={val_ppl:.1f}  "
          f"total_time={time.time()-t0:.0f}s")

    # Summarize
    return model, cfg, {"val_loss": val_loss, "val_ppl": val_ppl}


# Stage 1 pretraining config
STAGE = "stage1"

if STAGE == "stage1":
    PRE_EPOCHS = 30
    PRE_BATCH = 64
    PRE_LR = 5e-4
    NUM_LAYERS = 4
    HIDDEN_SIZE = 256
    NUM_HEADS = 4
    FPSA_MAX_ITER = 20
    FPSA_ADJOINT = 4
elif STAGE == "stage2":
    PRE_EPOCHS = 30
    PRE_BATCH = 128
    PRE_LR = 3e-4
    NUM_LAYERS = 6
    HIDDEN_SIZE = 384
    NUM_HEADS = 6
    FPSA_MAX_ITER = 20
    FPSA_ADJOINT = 4
else:
    raise ValueError(f"Unknown STAGE: {STAGE}")

torch.manual_seed(0)
vanilla_pretrained, vanilla_cfg, vanilla_pre_stats = pretrain_mlm(
    "vanilla", epochs=PRE_EPOCHS, batch_size=PRE_BATCH, lr=PRE_LR,
    num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS)

torch.manual_seed(0)
fpsa_pretrained, fpsa_cfg, fpsa_pre_stats = pretrain_mlm(
    "fpsa", epochs=PRE_EPOCHS, batch_size=PRE_BATCH, lr=PRE_LR,
    num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS,
    fpsa_max_iter=FPSA_MAX_ITER, fpsa_adjoint=FPSA_ADJOINT)

print("\nPretraining complete. Stashing encoder state dicts for fine-tuning.")
vanilla_encoder_sd = vanilla_pretrained.bert.state_dict()
fpsa_encoder_sd = fpsa_pretrained.bert.state_dict()

del vanilla_pretrained, fpsa_pretrained
if torch.cuda.is_available():
    torch.cuda.empty_cache()

"""### Pretraining gate: does the iterated attention pretrain OK?

Before fine-tuning, check that the iterated attention pretrained sensibly. If not, debug pretraining first.

"""

# Pretraining gate check
v_ppl = vanilla_pre_stats["val_ppl"]
f_ppl = fpsa_pre_stats["val_ppl"]
ratio = f_ppl / v_ppl
print(f"vanilla val PPL: {v_ppl:.1f}")
print(f"fpsa    val PPL: {f_ppl:.1f}")
print(f"ratio (fpsa/vanilla): {ratio:.2f}")
print()

# Loop-specific diagnostics
if "iters_per_layer" in fpsa_pre_stats:
    med = fpsa_pre_stats["median_loops"]
    conv = fpsa_pre_stats["mean_conv"]
    print(f"fpsa attention median loops: {med:.1f}  (healthy: 1-4)")
    print(f"fpsa attention mean conv:    {conv:.3f}  (healthy: >0.5)")
    print()

# Gate decision
gate_pass = True
warnings = []
if ratio > 1.5:
    gate_pass = False
    warnings.append(f"fpsa val PPL is {ratio:.2f}x vanilla's — should be <1.5x")
if ratio > 1.0:
    warnings.append(f"fpsa val PPL is {ratio:.2f}x vanilla's (slightly worse but acceptable)")

if gate_pass:
    print("✅ PRETRAINING GATE: PASS — fpsa pretrained healthily. Proceed to fine-tuning.")
else:
    print("⚠️  PRETRAINING GATE: WARNING — see issues below")
    for w in warnings:
        print(f"   - {w}")
    print()
    print("   You can still proceed to fine-tuning for diagnostic info, but")
    print("   if fine-tuning also shows fpsa losing, come back and debug pretraining first.")

"""## 5. Fine-tuning loop (with pretrained weights)"""

import time
import torch
from torch.utils.data import DataLoader

VOCAB_SIZE = tokenizer.vocab_size


def build_model(model_type, num_labels, num_layers=4, hidden_size=256,
                num_heads=4, fpsa_max_iter=20, fpsa_adjoint=4,
                load_pretrained=True):
    cfg = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=MAX_LEN + 8,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        attention_type=model_type,           # "vanilla" or "fpsa"
        fpsa_max_iter=8,
        fpsa_adjoint_max_iter=fpsa_adjoint,
        fpsa_tol=1e-3,
        fpsa_spectral_norm=False,
        fpsa_selective_freeze=True,
        fpsa_implicit_grad=True,
        fpsa_damping=0.5,
        fpsa_skip_tol=0.0,
        fpsa_conv_exit_frac=0.80,
        use_rope=(model_type != "vanilla"),
    )
    model = BertForSequenceClassification(cfg, num_labels=num_labels)

    # Load pretrained encoder weights (if available)
    if load_pretrained:
        src_sd = vanilla_encoder_sd if model_type == "vanilla" else fpsa_encoder_sd
        # The pretrained encoder has max_position_embeddings=MLM_BLOCK+8; the
        # fine-tune model has MAX_LEN+8. Position embeddings are (Pos, d). If
        # they match size, load; otherwise skip position embeddings.
        tgt_sd = model.bert.state_dict()
        loaded, skipped = 0, []
        for k, v in src_sd.items():
            if k in tgt_sd and tgt_sd[k].shape == v.shape:
                tgt_sd[k].copy_(v)
                loaded += 1
            else:
                skipped.append((k, tuple(v.shape), tuple(tgt_sd[k].shape) if k in tgt_sd else "MISSING"))
        model.bert.load_state_dict(tgt_sd)
        if skipped:
            print(f"  [pretrained] loaded {loaded} tensors, skipped {len(skipped)} (shape mismatch)")
            for k, src, tgt in skipped[:3]:
                print(f"    skipped {k}: pretrained{src} vs finetune{tgt}")

    return model, cfg


def evaluate(model, loader, device, metric="accuracy"):
    """Evaluate on a GLUE-style loader.

    Supported metrics:
      - "accuracy": standard accuracy (SST-2, MRPC, RTE, QNLI, MNLI)
      - "matthews": Matthews correlation (CoLA)
      - "f1": F1 score, positive class (MRPC, QQP)
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab = batch["label"].to(device)
            out = model(ids, mask)
            pred = out["logits"].argmax(dim=-1)
            all_preds.append(pred.cpu())
            all_labels.append(lab.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    if metric == "accuracy":
        return float((preds == labels).mean())
    elif metric == "matthews":
        # Matthews correlation: (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        tp = float(((preds == 1) & (labels == 1)).sum())
        tn = float(((preds == 0) & (labels == 0)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        if denom == 0:
            return 0.0
        return (tp * tn - fp * fn) / denom
    elif metric == "f1":
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def train_model(model, train_loader, val_loader, epochs, lr, device,
                label, metric="accuracy", log_every=50, warmup_frac=0.1):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=warmup_frac,
        anneal_strategy="linear", cycle_momentum=False,
    )
    model.to(device); model.train()
    step = 0
    best_score = -1.0
    t0 = time.time()
    # Track per-layer fpsa iterations across the whole training run
    fpsa_iter_log = []  # list of [iters_layer_0, iters_layer_1, ...] at each log step
    for ep in range(epochs):
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab = batch["label"].to(device)
            out = model(ids, mask, labels=lab)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            if step % log_every == 0:
                score = evaluate(model, val_loader, device, metric)
                best_score = max(best_score, score)
                # Get iterated per-layer iteration counts if this is an iterated model
                iterated_info = ""
                if hasattr(model.bert.encoder, 'attention_stats'):
                    stats = model.bert.encoder.attention_stats()
                    iters = stats.get("iters_per_layer", [])
                    conv = stats.get("converged_per_layer", [])
                    if iters and iters[0] is not None:
                        iterated_iter_log.append(iters)
                        iters_str = "/".join(str(x) for x in iters)
                        # Min convergence fraction across layers (most concerning)
                        min_conv = min(c for c in conv if c is not None) if conv else 1.0
                        iterated_info = f"  iters=[{iters_str}] conv_min={min_conv:.2f}"
                print(f"  [{label:>20}] ep{ep+1}/{epochs} step{step:5d} "
                      f"loss={out['loss'].item():.3f} "
                      f"{metric}={score:.4f} best={best_score:.4f}"
                      f"{iterated_info} "
                      f"({time.time()-t0:.0f}s)")
                model.train()
    # Final eval
    final = evaluate(model, val_loader, device, metric)
    best_score = max(best_score, final)
    # Compute fpsa per-layer iteration summary over the whole run
    fpsa_summary = ""
    if fpsa_iter_log:
        import numpy as np
        arr = np.array(fpsa_iter_log)  # (n_log_steps, n_layers)
        med_per_layer = np.median(arr, axis=0).astype(int).tolist()
        p99_per_layer = np.percentile(arr, 99, axis=0).astype(int).tolist()
        fpsa_summary = f"  fpsa iters: median={med_per_layer}  p99={p99_per_layer}"
    print(f"  [{label:>20}] FINAL {metric}={best_score:.4f}  "
          f"(total time {time.time()-t0:.0f}s){fpsa_summary}")
    return best_score


def run_one_task(task_name, model_type, seed=0,
                 train_max=None, val_max=None, epochs=3,
                 batch_size=32, lr=3e-4, num_layers=4, hidden_size=256,
                 num_heads=4, fpsa_max_iter=20, fpsa_adjoint=4,
                 load_pretrained=True, metric="accuracy"):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, num_labels = load_task(task_name, train_max, val_max)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2,
                            shuffle=False, num_workers=2)
    model, cfg = build_model(model_type, num_labels,
                              num_layers=num_layers, hidden_size=hidden_size,
                              num_heads=num_heads,
                              fpsa_max_iter=fpsa_max_iter, fpsa_adjoint=fpsa_adjoint,
                              load_pretrained=load_pretrained)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label = f"{model_type[:4]}-{num_layers}L/{task_name}/s{seed}"
    best = train_model(model, train_loader, val_loader, epochs, lr, device,
                       label, metric=metric)
    return {"task": task_name, "model_type": model_type, "score": best,
            "metric": metric, "params": n_params, "seed": seed}


print("Training utilities ready")

"""## 6. Fine-tuning: 4 GLUE tasks × 3 seeds

Per-task config chosen to balance signal quality and T4 runtime:

| task | train size | epochs | batch | lr | metric | expected T4 time/seed |
|---|---|---|---|---|---|---|
| SST-2 | 20k subsample | 3 | 32 | 3e-4 | accuracy | ~3 min |
| MRPC | 3.7k full | 5 | 32 | 2e-4 | accuracy | ~1 min |
| RTE | 2.5k full | 5 | 16 | 2e-4 | accuracy | ~1 min |
| CoLA | 8.5k full | 5 | 32 | 3e-4 | Matthews corr | ~2 min |

Per model × 3 seeds × 4 tasks ≈ 30 min. Both models → ~60 min total.

**Why these per-task choices:**
- SST-2 subsampled to 20k so one seed fits in a few minutes; full 67k takes ~10 min/seed
- MRPC/RTE are small, more epochs help
- RTE batch=16 because the train set is tiny — small batches give better gradient signal
- CoLA uses Matthews correlation (the canonical metric); accuracy is misleading due to class imbalance

If you want to match the paper's protocol exactly, set `train_max=None` for all tasks and bump epochs to the per-task optima from the BERT paper (SST-2: 3, others: 3-4). That triples runtime.

"""

# Stage 1 sweep: 4 tasks × 3 seeds × 2 model types (vanilla vs looped)

SEEDS = [0, 1, 2]

# Per-task configuration
TASK_CONFIGS = {
    "sst2": {"train_max": 20000, "val_max": None, "epochs": 3,
             "batch_size": 32, "lr": 3e-4, "metric": "accuracy"},
    "mrpc": {"train_max": None,  "val_max": None, "epochs": 5,
             "batch_size": 32, "lr": 2e-4, "metric": "accuracy"},
    "rte":  {"train_max": None,  "val_max": None, "epochs": 5,
             "batch_size": 16, "lr": 2e-4, "metric": "accuracy"},
    "cola": {"train_max": None,  "val_max": None, "epochs": 5,
             "batch_size": 32, "lr": 3e-4, "metric": "matthews"},
}

results = []
total_runs = len(TASK_CONFIGS) * len(SEEDS) * 2
run_idx = 0
sweep_start = time.time()

for task_name, tc in TASK_CONFIGS.items():
    for seed in SEEDS:
        for model_type in ["vanilla", "fpsa"]:
            run_idx += 1
            print(f"\n--- Run {run_idx}/{total_runs}: {model_type} / {task_name} / seed {seed} ---")
            elapsed_min = (time.time() - sweep_start) / 60
            print(f"    (sweep elapsed: {elapsed_min:.1f} min)")
            r = run_one_task(
                task_name, model_type, seed=seed,
                train_max=tc["train_max"], val_max=tc["val_max"],
                epochs=tc["epochs"], batch_size=tc["batch_size"], lr=tc["lr"],
                num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE,
                num_heads=NUM_HEADS,
                fpsa_max_iter=FPSA_MAX_ITER, fpsa_adjoint=FPSA_ADJOINT,
                metric=tc["metric"],
            )
            results.append(r)

total_elapsed = (time.time() - sweep_start) / 60
print(f"\nSweep complete in {total_elapsed:.1f} min ({total_elapsed/60:.2f} hours).")

"""## 7. Results — per-task, with seed statistics

Aggregate across seeds. Report mean ± std per task per model.

"""

# Aggregate across seeds
from collections import defaultdict
import numpy as np

grouped = defaultdict(list)
for r in results:
    grouped[(r["task"], r["model_type"])].append(r["score"])

tasks = list(TASK_CONFIGS.keys())
metrics = {t: TASK_CONFIGS[t]["metric"] for t in tasks}

print("=" * 85)
print(f"Stage 1 — {len(SEEDS)} seeds, {NUM_LAYERS}L x {HIDDEN_SIZE} model, pretrained on WikiText-2")
print("=" * 85)
print(f"{'task':<8} {'metric':<12} {'vanilla':>18} {'fpsa':>18} {'gap(pp)':>10}")
print("-" * 85)

per_task_gaps = []
van_means = {}; loop_means = {}
for task in tasks:
    v = grouped[(task, "vanilla")]
    l = grouped[(task, "fpsa")]
    vm, vs = np.mean(v), np.std(v)
    lm, ls = np.mean(l), np.std(l)
    van_means[task] = vm; loop_means[task] = lm
    gap = (lm - vm) * 100
    per_task_gaps.append(gap)
    print(f"{task:<8} {metrics[task]:<12} "
          f"{vm*100:>7.2f} +/- {vs*100:>4.2f}   "
          f"{lm*100:>7.2f} +/- {ls*100:>4.2f}   "
          f"{gap:>+9.2f}")

print("-" * 85)
va = np.mean(list(van_means.values())) * 100
la = np.mean(list(loop_means.values())) * 100
print(f"{'avg':<8} {'':<12} {va:>18.2f} {la:>18.2f} {la - va:>+9.2f}")
print("=" * 85)

print()
gap_abs = la - va
if gap_abs >= -1.0:
    if gap_abs > 1.0:
        print(f"PASS: FPSA attention wins by {gap_abs:.2f}pp avg. Scale up to Stage 2.")
    else:
        print(f"PASS: Gap is {gap_abs:+.2f}pp. Architecture works. Scale up for stronger signal.")
else:
    print(f"FAIL: FPSA attention lags by {abs(gap_abs):.2f}pp avg. Debug before scaling.")

vp = [r for r in results if r['model_type']=='vanilla' and r['seed']==0][0]['params']
lp = [r for r in results if r['model_type']=='fpsa' and r['seed']==0][0]['params']
print(f"\nParams: vanilla = {vp:,}, fpsa = {lp:,} (should be identical)")

"""## 8. Interpreting Stage 1 results + Stage 2/3 plan

### If Stage 1 passed (FPSA matches or beats vanilla)

Move to Stage 2. The code changes are minimal — see below.

### If Stage 1 failed (FPSA behind by > 1pp avg)

**Do not scale up yet.** Common causes and fixes:

- **`FPSA_ITER` too low**: FPSA might need more refinement steps after pretraining. Try `FPSA_ITER=16` (change the constant near the bottom of the pretraining config cell).
- **Fine-tune LR too aggressive**: FPSA's loss surface has more curvature because the attention map composes with itself. Try lowering `lr` in the `TASK_CONFIGS` dict to 1e-4.
- **Not enough pretraining**: bump `PRE_EPOCHS` to 50 or 60. FPSA's pretraining may need more steps to catch up to vanilla in perplexity.
- **Check the FPSA diagnostics**: if `p99_iters` is hitting `FPSA_ITER`, the fixed-point isn't converging at high LR mid-training. Lower `fpsa_damping=0.5` in the model config — this halves the per-step update and usually restores convergence.

### FPSA runtime diagnostics

To see per-layer iteration counts during training or eval:

```python
stats = model.bert.encoder.attention_stats()
# stats["iters_per_layer"]     -> e.g. [3, 4, 5, 4] median iterations
# stats["converged_per_layer"] -> e.g. [1.0, 1.0, 0.98, 1.0] convergence fraction
```

Typical healthy values: 3–6 iterations per layer, 98%+ convergence.

---

## Stage 2: WikiText-103 at ~30M params (~8–12h on L4)

If Stage 1 gate passed, Stage 2 code changes are minimal. In the MLM data cell, swap:

```python
# Stage 2: load WikiText-103 instead of WikiText-2
WT = load_dataset("wikitext", "wikitext-103-raw-v1")
# ... (rest of tokenization and MLMDataset construction is identical)
```

Then in the pretraining config cell, flip:

```python
STAGE = "stage2"   # triggers the 6L × 384 config path
```

Stage 2 takes ~4–6 hours pretraining per model on an L4 GPU (~$0.50–0.75/hr on Colab Pro, ~$0.40/hr on Lambda Cloud). Fine-tuning time is similar to Stage 1. Total Stage 2 budget: **$30–50**.

**Stage 2 pass criteria** (to advance to Stage 3):
- FPSA val PPL within 1.2× vanilla's
- FPSA mean GLUE-4 score within 0.5pp of vanilla OR FPSA ahead
- FPSA iteration diagnostics healthy as in Stage 1

## Stage 3: Full BERT-Base (110M, BookCorpus+Wiki)

Stage 3 is not runnable in a Colab notebook — needs real infrastructure:

- Multi-GPU setup (4× A100 recommended) or single A100 for ~2 weeks
- BookCorpus + Wikipedia dataset (~50GB, streamed)
- Proper checkpoint saving / resuming
- Full 8 GLUE tasks with per-task hyperparameter sweep (BERT paper protocol)

Budget: **$700–1500** on cloud GPU. Don't run Stage 3 until Stage 2 shows FPSA winning with clear statistical significance (3+ seeds, gap > std deviation).

A Stage 3 training script stub is provided separately in `scripts/stage3_pretrain.py`. The architecture code (FPSA module itself) doesn't change — only the training harness, data pipeline, and compute infrastructure.

"""
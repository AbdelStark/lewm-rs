# PushT Training Report

**Date:** 2026-05-15
**Job:** `abdelstark/6a06f0c43308d79117b90276`
**Hardware:** A10G-large (HuggingFace Jobs)
**Wall time:** 318 min (~5h18m)
**Artifacts:** `abdelstark/lewm-rs-pusht/train/pusht-full-lewm-20260515T100908Z/`

**Naming note:** this historical Hub path predates the bounded-artifact
correction. Bounded PushT smoke/short jobs now use `pusht-bounded-module-lewm`
labels, while the approval-gated production PushT job now uploads future full
Burn/Jepa runs under `train/pusht-full-burn-jepa-*`.

## Dataset

| Property | Value |
|----------|-------|
| Source | `quentinll/lewm-pusht` |
| Format | HDF5 with Blosc compression |
| Training windows | 2,092,476 |
| Image resolution | 224×224 |
| Action space | 2-DOF (x, y) |
| Smoothed action dim | 10 (Conv1d smoother) |

## Training Configuration

| Property | Value |
|----------|-------|
| Mode | `pusht-minimal-lewm` |
| Model | `PushtFullLewmCore` bounded host core (dim=192, 14 tensor groups) |
| Max steps | 50,000 |
| Batch size | 64 |
| Device | `cuda:0` (A10G-large) |
| Precision | bf16 mixed |
| Optimizer | AdamW |
| LR schedule | Cosine + linear warmup (warmup=1000) |
| Peak LR | 3.00e-04 |
| Final LR | 1.00e-05 |
| Weight decay | 0.05 |
| β | [0.9, 0.95] |
| Grad clip | 1.0 |
| Seed | 0 |
| Config hash | `438eb30f4bb0` |

## Training Curve

| Step | Total loss | SIGReg | Pred loss | LR | Grad norm (post) |
|-----:|-----------|--------|-----------|-----|-----------------|
| 1 | 4.912e-01 | 4.905e-01 | 6.82e-04 | 3.00e-07 | 1.72e-01 |
| 100 | 4.899e-01 | 4.893e-01 | 6.14e-04 | 3.00e-05 | 1.84e-01 |
| 500 | 4.382e-01 | 4.380e-01 | 2.27e-04 | 1.50e-04 | 4.64e-01 |
| 1,000 | 8.69e-02 | 8.69e-02 | 8.43e-07 | 3.00e-04 | 7.05e-01 |
| 5,000 | 6.09e-06 | 4.96e-06 | 1.13e-06 | 2.95e-04 | 4.97e-03 |
| 10,000 | 8.35e-06 | 8.03e-06 | 3.12e-07 | 2.77e-04 | 2.90e-03 |
| 20,000 | 4.17e-06 | 4.13e-06 | 4.29e-08 | 2.05e-04 | 5.53e-03 |
| 30,000 | 2.28e-06 | 1.91e-06 | 3.73e-07 | 1.14e-04 | 1.35e-03 |
| 40,000 | 7.42e-06 | 6.94e-06 | 4.78e-07 | 3.88e-05 | 1.92e-03 |
| 50,000 | 3.17e-06 | 3.00e-06 | 1.69e-07 | 1.00e-05 | 3.44e-03 |

## Summary

| Metric | Value |
|--------|-------|
| Initial loss | 0.4912 |
| Final loss | 3.17e-06 |
| Loss ratio | 154,900× reduction |
| Gradient explosions | 0 |
| Steps completed | 50,000 / 50,000 |

The loss is dominated by the SIGReg term during the first ~1,000 steps (encoder
representation learning). After step ~5,000 both SIGReg and pred_loss have
converged to near-zero and remain there through the end of training, with minor
oscillation from the cosine schedule tail.

## Artifacts

| File | Description |
|------|-------------|
| `step_0050000.safetensors` | Bounded host-core parameter mirror (14 tensors, 1.2 KB) |
| `step_0050000.mpk` | Bounded host-core checkpoint (model + optimizer + RNG, 1.2 KB) |
| `step_0050000.json` | Checkpoint sidecar (config hash, step, seed) |
| `step_0050000.parity.json` | Parity probe output |
| `train_losses.jsonl` | Per-step loss log (50,000 rows) |
| `train_report.json` | Training summary (schema v1.0.0) |

## ONNX Export

The existing root and `tract-compat/` ONNX artifacts in
`abdelstark/lewm-rs-pusht` come from the parity-verified reference checkpoint,
not from the 50k-step bounded host-core checkpoint above.

The F1 export attempt on 2026-05-18 showed that
`step_0050000.safetensors` contains 14 bounded-core tensors, not the 255-tensor
full Burn/Jepa mirror expected by `python/export_onnx.py`. See
`reports/pusht_onnx_export.md` for the command log and blocker evidence.

The exporter is ready for a compatible full checkpoint:

```bash
uv run --extra parity python python/export_onnx.py \
  --safetensors train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors \
  --meta tests/fixtures/reference_model.meta.json \
  --output-dir /tmp/pusht-onnx \
  --variant both \
  --action-dim 10
```

## CEM Planning Evaluation

Pending. Target: ≥ 87% success rate on 50 test episodes (matching the reference
paper). This is blocked until a trained full Burn/Jepa PushT checkpoint is
available or the release acceptance criteria are changed to evaluate the bounded
host-core checkpoint.

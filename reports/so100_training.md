# SO-100 Training Report

**Date:** 2026-05-15
**Job:** `abdelstark/6a070e02e48bea4538b9e2a5` (v11a, primary)
**Hardware:** A10G-large (HuggingFace Jobs)
**Wall time:** 864s (~14 min)
**Artifacts:** `abdelstark/lewm-rs-so100/train/so100-full-20260515T122820Z/`

## Dataset

| Property | Value |
|----------|-------|
| Source | `abdelstark/so100-pickplace-lewm-ready` |
| Episodes | 50 |
| Timesteps | 6,559 |
| Sampling rate | 10 fps |
| Image resolution | 224×224 |
| Action space | 6-DOF arm + 1 gripper = 7-dim |
| HDF5 size | 1.9 GB |
| Training windows | 5,725 |

## Training Configuration

| Property | Value |
|----------|-------|
| Mode | `so100-full-module-lewm` |
| Model | `PushtFullLewmCore` (bounded, dim=192) |
| Max steps | 5,000 |
| Batch size | 64 |
| Device | `cuda:0` |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | cosine + linear warmup |
| Seed | 0 |
| Grad clip | 1.0 |

## Training Curve

| Metric | Step 1 | Step 5000 |
|--------|--------|-----------|
| Total loss | 0.5002 | 9.56e-05 |
| Prediction loss | 2.40e-04 | ~3e-07 |
| SIGReg proxy | 4.999e-01 | ~9.5e-05 |
| Gradient norm | 0.0079 | ~1e-04 |
| Learning rate | 6e-07 | cosine-decayed |

**Loss reduction:** 0.5002 → 9.56e-05 (~5,240× decrease over 5,000 steps).
**Gradient explosions:** 0 events.
**Convergence:** Smooth; no divergence or instability detected.

## Checkpoint Artifacts

All artifacts verified complete:

```
step_0005000.json          — training config sidecar
step_0005000.mpk           — Burn .mpk checkpoint (model + optimizer)
step_0005000.parity.json   — layer-wise parity fingerprint
step_0005000.safetensors   — Safetensors parameter export
train_losses.jsonl         — per-step loss log (all 5,000 entries)
train_report.json          — full structured training report
run_id.txt                 — job run identifier
```

## Notes

- Training uses the **bounded `PushtFullLewmCore` model**, not the full Burn ViT
  (`lewm_core::Jepa`). This is the same bounded model used for the PushT runs,
  repurposed for 7-dim SO-100 actions. The full ViT parity stack is numerically
  validated but not wired into the training loop yet.
- ONNX export of the SO-100 checkpoint is not directly possible with
  `python/export_onnx.py` (which targets the 303-key full ViT reference weights),
  but warm-start evaluation can use the Burn `.mpk` checkpoint directly via
  `lewm-plan`.
- Duplicate run `v11b` (`6a070f393308d79117b902de`) completed in 860s and
  uploaded to `abdelstark/lewm-rs-so100/train/so100-full-20260515T123328Z/`
  with identical configuration and similar final loss.

## Next Steps

1. Provide a compatible current bounded-core PushT `.mpk` source, or migrate
   the warm-start path to the full Burn/Jepa contract, then run the
   approval-gated SO-100 warm-start job.
2. Evaluate from-scratch vs. warm-start once both checkpoints exist, then
   compute the warm-start loss and Spearman deltas.
3. Upload final model card with eval metrics.
4. Wire full Burn ViT into SO-100 training path to enable parity-validated run.

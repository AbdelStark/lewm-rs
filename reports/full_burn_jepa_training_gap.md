# Full Burn/Jepa Training Gap

**Date:** 2026-05-18  
**Related issues:** F1 / #243, F3 / #245, F13 / #255  
**Status:** Blocks production release

## Summary

The ONNX exporter and verifier are ready for a valid full-layout PushT
checkpoint, but the checked-in PushT training job still cannot produce one.
Rerunning `jobs/train_pusht.yaml` would produce another bounded
`PushtFullLewmCore` artifact, not a trained `lewm_core::Jepa` checkpoint.
The checked-in bounded PushT jobs now publish under `pusht-bounded-module-lewm`
labels so new artifacts cannot be mistaken for full Burn/Jepa checkpoints.

This is the root implementation gap behind the current F1 blocker.

## Evidence

`configs/pusht.toml` defines the full JEPA architecture dimensions, but it does
not select a full Burn/Jepa training mode.

`jobs/train_pusht.yaml` runs:

```text
lewm-train train --config configs/pusht.toml ... --max-steps ${LEWM_MAX_STEPS:-1000}
```

That job is intentionally labeled as a bounded-module run until the full
Burn/Jepa path exists.

`crates/lewm-train/src/trainer.rs` dispatches PushT training through
`write_pusht_train_artifacts`, which always initializes
`PushtFullLewmTrainingStart::fresh`. That start state owns
`PushtFullLewmCore`, the bounded host-side model.

The checkpoint writer for that path is `write_pusht_full_lewm_checkpoint`,
which serializes `PushtFullLewmRecord` as JSON bytes and exports parameter
tensors from `PushtFullLewmCore::parameter_specs()`. The current published
50k-step PushT `.safetensors` contains 14 bounded-core tensors, while
`python/export_onnx.py` requires the full 303-key PyTorch/Burn mapping.

`docs/src/crates/lewm-train.md` already names the missing mode:

```text
pusht-full-burn-jepa | Full lewm_core::Jepa (303 params); wire-up pending.
```

The exporter itself is not the blocker:

- It rejects the current published bounded artifact with an explicit checkpoint
  contract diagnostic.
- It successfully exports and verifies both ONNX variants from the local
  converted full-layout reference checkpoint at
  `/tmp/lewm-parity-dumps/reference_model.safetensors`.

## Required Implementation

F1 needs a PushT training path that produces a real trained
`lewm_core::Jepa` checkpoint. At minimum:

1. Add an explicit training mode boundary so PushT bounded-core training and
   full Burn/Jepa training cannot be confused in reports, job paths, or Hub
   uploads.
2. Collate PushT samples into Burn tensors shaped `(B, horizon, 3, 224, 224)`
   and `(B, horizon, 10)` using the existing PushT packing contract.
3. Train `lewm_core::Jepa<B>` through `Jepa::criterion`, including SIGReg RNG
   substream accounting and the existing scheduler / grad-clip / NaN guard
   policy.
4. Persist a Burn `NamedMpkFileRecorder` record and a full-layout safetensors
   mirror compatible with `python/export_onnx.py`.
5. Add a local smoke that proves the full Burn/Jepa path writes a checkpoint
   whose safetensors recover all 303 expected PyTorch keys.
6. Update the HF Job so the production PushT run selects that full Burn/Jepa
   mode and uploads under a path that cannot be mistaken for the bounded
   artifact.
7. Only after a real full-layout 50k PushT checkpoint exists, run:

```text
uv run --project python --extra parity python python/export_onnx.py \
  --safetensors <full-pusht-step_0050000.safetensors> \
  --meta tests/fixtures/reference_model.meta.json \
  --output-dir /tmp/pusht-onnx-full \
  --variant both \
  --action-dim 10

uv run --project python --extra parity python python/verify_onnx.py \
  --dir /tmp/pusht-onnx-full
```

## Acceptance Gate

Do not mark F1 resolved until all of these are true:

- The source PushT safetensors recovers all 303 expected PyTorch keys.
- Both `onnxruntime` and `tract-compat` variants are generated under
  `onnx-full/`.
- `python/verify_onnx.py --dir <onnx-full-root>` passes.
- The verified ONNX variants are uploaded to `abdelstark/lewm-rs-pusht`.
- `conformance/release_blockers.json` marks F1 resolved only after the evidence
  above is in place.

# Full Burn/Jepa Training Gap

**Date:** 2026-05-18  
**Related issues:** F1 / #243, F3 / #245, F13 / #255  
**Status:** Blocks production release

## Summary

The ONNX exporter and verifier are ready for a valid full-layout PushT
checkpoint. `lewm-train` now has an opt-in CPU full Burn/Jepa mode
(`experimental.pusht_train_mode = "full_burn_jepa"`) that trains
`lewm_core::Jepa`, writes a Burn `NamedMpk` record, and mirrors the full
Safetensors layout. The checked-in PushT training job still does not select
that mode, so rerunning `jobs/train_pusht.yaml` would produce another bounded
`PushtFullLewmCore` artifact, not the production 50k full-JEPA checkpoint.
The checked-in bounded PushT jobs and checkpoints now use
`pusht-bounded-module-lewm` labels so new artifacts cannot be mistaken for full
Burn/Jepa checkpoints.

This is the root implementation gap behind the current F1 blocker.

## Evidence

`configs/pusht.toml` defines the full JEPA architecture dimensions, but it does
not select the opt-in full Burn/Jepa training mode.

`jobs/train_pusht.yaml` runs:

```text
lewm-train train --config configs/pusht.toml ... --max-steps ${LEWM_MAX_STEPS:-1000}
```

That job is intentionally labeled as a bounded-module run until the full
Burn/Jepa path exists.

`crates/lewm-train/src/trainer.rs` now dispatches PushT training by
`experimental.pusht_train_mode`. The default `bounded_module` path initializes
`PushtFullLewmTrainingStart::fresh`; that start state owns `PushtFullLewmCore`,
the bounded host-side model. The opt-in `full_burn_jepa` path initializes
`lewm_core::Jepa<Autodiff<NdArray<f32>>>`, tensors real PushT windows, trains
through `Jepa::criterion`, and writes full-layout checkpoint artifacts.

The checkpoint writer for the bounded path is `write_pusht_full_lewm_checkpoint`,
which serializes `PushtFullLewmRecord` as JSON bytes and exports parameter
tensors from `PushtFullLewmCore::parameter_specs()`. The current published
50k-step PushT `.safetensors` contains 14 bounded-core tensors, while
`python/export_onnx.py` requires the full 303-key PyTorch/Burn mapping.

`docs/src/crates/lewm-train.md` now names both PushT modes:

```text
pusht-bounded-module-lewm | Current checked-in PushtFullLewmCore bounded host-module train path.
pusht-full-burn-jepa | Opt-in CPU Burn autodiff path selected with experimental.pusht_train_mode = "full_burn_jepa".
```

The exporter itself is not the blocker:

- It rejects the current published bounded artifact with an explicit checkpoint
  contract diagnostic.
- It successfully exports and verifies both ONNX variants from the local
  converted full-layout reference checkpoint at
  `/tmp/lewm-parity-dumps/reference_model.safetensors`.

## Required Implementation

F1 still needs a production PushT training run that produces a real trained
50k-step `lewm_core::Jepa` checkpoint. Remaining work:

1. Run the full mode against the production PushT data/config long enough to
   produce `step_0050000.safetensors`.
2. Add a local or CI smoke that proves the full Burn/Jepa path writes a
   checkpoint whose safetensors recover all 303 expected PyTorch keys for the
   production config.
3. Update the HF Job so the production PushT run selects that full Burn/Jepa
   mode and uploads under a path that cannot be mistaken for the bounded
   artifact.
4. Only after a real full-layout 50k PushT checkpoint exists, run:

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

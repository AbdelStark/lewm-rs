# Full Burn/Jepa Training Gap

**Date:** 2026-05-18  
**Related issues:** F1 / #243, F3 / #245, F13 / #255  
**Status:** Blocks production release

## Summary

The ONNX exporter and verifier are ready for a valid full-layout PushT
checkpoint. `lewm-train` now has an opt-in CPU full Burn/Jepa mode
(`experimental.pusht_train_mode = "full_burn_jepa"`) that trains
`lewm_core::Jepa`, writes a Burn `NamedMpk` record, and mirrors the full
Safetensors layout. The approval-gated PushT production job now selects this
CPU-backed mode, reports `--device cpu`, and uploads under
`train/pusht-full-burn-jepa-*`, but no approved production run has produced or
uploaded a 50k full-JEPA checkpoint yet. Before upload, that job runs the
exporter's safetensors-only checkpoint contract check so another bounded or
otherwise non-exportable checkpoint cannot be published under the full
checkpoint path. The
checked-in bounded PushT smoke/short jobs and checkpoints use
`pusht-bounded-module-lewm` labels so new artifacts cannot be mistaken for full
Burn/Jepa checkpoints.

This is the remaining artifact gap behind the current F1 blocker.

## Evidence

`configs/pusht.toml` defines the full JEPA architecture dimensions, but it does
not select the opt-in full Burn/Jepa training mode.

`jobs/train_pusht.yaml` now runs:

```text
lewm-train train \
  --config configs/pusht.toml \
  --set 'experimental.pusht_train_mode="full_burn_jepa"' \
  --device cpu \
  ... \
  --max-steps ${LEWM_MAX_STEPS:-50000}
```

It uploads to `train/pusht-full-burn-jepa-$(date -u +%Y%m%dT%H%M%SZ)`.
Before the upload, it validates the produced checkpoint with:

```text
python python/export_onnx.py \
  --safetensors /tmp/out/step_${checkpoint_step}.safetensors \
  --check-contract-only
```

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
`python/export_onnx.py` requires the exact 255 Burn destination tensors that
recover the full 303-key PyTorch source mapping.

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
- A local one-step release-config full Burn/Jepa smoke completed on CPU and
  passed the exporter's safetensors-only contract check:

  ```text
  cargo run -p lewm-train --bin lewm-train -- \
    --config configs/pusht.toml \
    --set 'experimental.pusht_train_mode="full_burn_jepa"' \
    --device cpu \
    --output-dir /tmp/lewm-pusht-full-burn-jepa-contract-smoke-1779100637 \
    --max-steps 1 \
    train

  [pusht-burn-jepa step 1/1] loss=0.721793 pred=0.607750 sigreg=0.114043 lr=3.00e-7 elapsed=65s eta=0s
  train artifacts written to /tmp/lewm-pusht-full-burn-jepa-contract-smoke-1779100637; mode=pusht-full-burn-jepa; checkpoint_step=1; checkpoint_complete=true

  uv run --project python --with safetensors python python/export_onnx.py \
    --safetensors /tmp/lewm-pusht-full-burn-jepa-contract-smoke-1779100637/step_0000001.safetensors \
    --check-contract-only

  Checkpoint contract ok: recovered 303 of 303 expected PyTorch keys
  Burn destination tensors: 255
  Safetensors SHA-256: b9cbd30771c4f35725fe8ea8ec54660fd18df59e8aade45c06e2d111e60bb3eb
  ```

  `scripts/full_pusht_contract_smoke.py` now wraps this local operator smoke
  without any Hub upload or paid job launch. This proves the release-config
  full-mode writer can produce the F1 ONNX contract locally; it still does not
  replace the missing approved 50k PushT production checkpoint.

## Required Implementation

F1 still needs a production PushT training run that produces a real trained
50k-step `lewm_core::Jepa` checkpoint. Remaining work:

1. With human approval, run `jobs/train_pusht.yaml` against the production
   PushT data/config long enough to produce `step_0050000.safetensors`.
2. Only after a real full-layout 50k PushT checkpoint exists, run:

```text
uv run --project python --extra parity python python/export_onnx.py \
  --safetensors <full-pusht-step_0050000.safetensors> \
  --meta tests/fixtures/reference_model.meta.json \
  --output-dir /tmp/pusht-onnx-full \
  --variant both \
  --action-dim 10

uv run --project python --extra parity python python/verify_onnx.py \
  --dir /tmp/pusht-onnx-full

python3 python/upload_checkpoints.py \
  --src /tmp/pusht-onnx-full \
  --dst abdelstark/lewm-rs-pusht \
  --path-prefix onnx-full/ \
  --dry-run
```

## Acceptance Gate

Do not mark F1 resolved until all of these are true:

- The source PushT safetensors contains the exact 255 expected Burn destination
  tensors and recovers all 303 expected PyTorch keys.
- Both `onnxruntime` and `tract-compat` variants are generated under
  `onnx-full/`.
- `python/verify_onnx.py --dir <onnx-full-root>` passes.
- The verified ONNX variants are uploaded to `abdelstark/lewm-rs-pusht`; the
  local `--dry-run` upload check is only a preflight and does not resolve F1.
- `conformance/release_blockers.json` marks F1 resolved only after the evidence
  above is in place.

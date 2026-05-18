# PushT ONNX Export Report

**Date:** 2026-05-18
**Issue:** F1 / #243
**Status:** Still blocked by artifact mismatch after live Hub re-check

## Objective

Export the full PushT checkpoint at
`abdelstark/lewm-rs-pusht/train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors`
to both release ONNX layouts, then upload them under `onnx-full/`:

| Variant | Target | Expected path |
|---------|--------|---------------|
| `onnxruntime` | HF Space / general Python inference | `onnx-full/onnxruntime/` |
| `tract-compat` | `lewm-infer` / Tract 0.22.1 | `onnx-full/tract-compat/` |

## Commands Run

Downloaded the advertised full checkpoint and sidecars:

```bash
hf download abdelstark/lewm-rs-pusht \
  --include 'train/pusht-full-lewm-20260515T100908Z/*' \
  --local-dir /tmp/pusht-full
```

Attempted export with the release exporter:

```bash
uv run --project python --extra parity python python/export_onnx.py \
  --safetensors /tmp/pusht-full/train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors \
  --meta tests/fixtures/reference_model.meta.json \
  --output-dir /tmp/pusht-onnx-full \
  --variant both \
  --action-dim 10
```

## Evidence

The Hub tree contains no exportable full `lewm_core::Jepa` PushT checkpoint.
The advertised 50k-step checkpoint is only 1.2 KB:

```text
train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors 1264 bytes
train/pusht-full-lewm-20260515T100908Z/step_0050000.mpk         1266 bytes
```

Live unauthenticated Hub API re-check on 2026-05-18 confirmed the same state
at repo commit `c59a9beb6fec79717719fb541220294001d67100`:

```text
train/pusht-full-lewm-20260515T100908Z/step_0050000.json        1009 bytes
train/pusht-full-lewm-20260515T100908Z/step_0050000.mpk         1266 bytes
train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors 1264 bytes
```

Safetensors inspection shows 14 bounded-core tensors, not the 303 tensors
required by the locked PushT reference / `python/export_onnx.py` mapping:

```text
count 14
action_encoder.bias
action_encoder.x.weight
action_encoder.y.weight
encoder.bias
encoder.energy.weight
encoder.pixel.weight
encoder.time.weight
pred_proj.bias
pred_proj.weight
predictor.action.weight
predictor.bias
predictor.latent.weight
projector.bias
projector.weight
```

The downloaded `train_report.json` also identifies the run mode as
`pusht-minimal-lewm`, and `docs/src/crates/lewm-train.md` states that the
current PushT 50k-step result was produced by the 14-parameter
`PushtFullLewmCore` bounded host path. That path is not the full Burn/Jepa
model expected by the ONNX exporter.

The live `step_0050000.json` sidecar likewise reports:

```json
{
  "run_id": "pusht-minimal-lewm-v1",
  "step": 50000,
  "metrics_last_step": {
    "loss/train": 3.1696034906382155e-06
  }
}
```

The exporter now fails before graph generation with an explicit checkpoint
contract diagnostic:

```text
Loading Burn safetensors: /tmp/pusht-full-contract-check/train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors
ERROR: checkpoint does not match the full Burn/Jepa ONNX export contract
recovered 0 of 303 expected PyTorch keys
source safetensors tensor count: 14 (expected 255 Burn destination tensors)
the tensor names match the bounded PushtFullLewmCore training artifact, not the full lewm_core::Jepa checkpoint required for ONNX export
first missing Burn destination tensors:
  - action_encoder.fc1.bias
  - action_encoder.fc1.weight
  - action_encoder.fc2.bias
  - action_encoder.fc2.weight
  - action_encoder.smoother.bias
  - action_encoder.smoother.weight
  - encoder.blocks.0.attn.proj.bias
  - encoder.blocks.0.attn.proj.weight
  - encoder.blocks.0.attn.qkv.bias
  - encoder.blocks.0.attn.qkv.weight
first unexpected Burn destination tensors:
  - action_encoder.bias
  - action_encoder.x.weight
  - action_encoder.y.weight
  - encoder.bias
  - encoder.energy.weight
  - encoder.pixel.weight
  - encoder.time.weight
  - pred_proj.bias
  - pred_proj.weight
  - predictor.action.weight
first missing PyTorch source keys:
  - action_encoder.embed.0.bias
  - action_encoder.embed.0.weight
  - action_encoder.embed.2.bias
  - action_encoder.embed.2.weight
  - action_encoder.patch_embed.bias
  - action_encoder.patch_embed.weight
  - encoder.embeddings.cls_token
  - encoder.embeddings.patch_embeddings.projection.bias
  - encoder.embeddings.patch_embeddings.projection.weight
  - encoder.embeddings.position_embeddings
provide a full Burn/Jepa safetensors checkpoint or use a separate exporter for the bounded-core artifact
```

This diagnostic can now be reproduced with only `safetensors` installed; a
bad checkpoint no longer needs a full `torch` environment just to fail the
contract preflight:

```text
uv run --project python --with safetensors python python/export_onnx.py \
  --safetensors /tmp/pusht-full-live-check/train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors \
  --meta tests/fixtures/reference_model.meta.json \
  --output-dir /tmp/pusht-onnx-full-live-check \
  --variant both \
  --action-dim 10
```

No `onnx-full/` upload was performed because the source artifact does not
satisfy the F1 acceptance contract.

## Tooling Completed

The Python edge tooling is ready for a valid full PushT checkpoint:

- `python/export_onnx.py --variant both` writes explicit `onnxruntime/` and
  `tract-compat/` directories.
- `onnxruntime` exports use opset 18 with dynamic batch axes.
- `tract-compat` exports use opset 17 with fixed batch for Tract 0.22.1.
- `onnx_export.json` is written at the export root and in each variant
  directory, including `action_dim`, `step_count`, source safetensors path,
  source SHA-256, variant options, and export timestamp.
- `python/verify_onnx.py` verifies ONNX Runtime shape execution for both
  variants and checks dynamic batch behavior for the `onnxruntime` variant.
- `python/export_onnx.py` now validates the checkpoint contract up front and
  refuses bounded-core artifacts with an actionable F1 diagnostic instead of
  falling through to a raw missing-key error.
- Invalid-checkpoint preflight is safetensors-only: `torch` is required only
  after the source checkpoint has passed the full-layout contract.
- `python/export_onnx.py --check-contract-only` is available for job gates
  that need to validate the exact 255 Burn destination / 303 PyTorch source
  full-checkpoint contract before upload without exporting ONNX.

Focused validation:

```text
uv run --project python pytest python/tests/test_export_onnx.py python/tests/test_verify_conversion.py
7 passed

uv run --project python ruff check python/export_onnx.py python/verify_onnx.py python/tests/test_export_onnx.py python/pyproject.toml
All checks passed
```

Positive full-layout smoke, using the converted locked upstream reference
checkpoint at `/tmp/lewm-parity-dumps/reference_model.safetensors`:

```text
uv run --project python --extra parity python python/export_onnx.py \
  --safetensors /tmp/lewm-parity-dumps/reference_model.safetensors \
  --meta tests/fixtures/reference_model.meta.json \
  --output-dir /tmp/lewm-reference-onnx-contract-check \
  --variant both \
  --action-dim 10

Recovered 303 PyTorch keys from Burn checkpoint.
Encoder ONNX written: /tmp/lewm-reference-onnx-contract-check/onnxruntime/encoder.onnx (opset=18, dynamic_batch=True)
Predictor ONNX written: /tmp/lewm-reference-onnx-contract-check/onnxruntime/predictor.onnx (action_dim=10, opset=18, dynamic_batch=True)
Encoder ONNX written: /tmp/lewm-reference-onnx-contract-check/tract-compat/encoder.onnx (opset=17, dynamic_batch=False)
Predictor ONNX written: /tmp/lewm-reference-onnx-contract-check/tract-compat/predictor.onnx (action_dim=10, opset=17, dynamic_batch=False)

uv run --project python --extra parity python python/verify_onnx.py \
  --dir /tmp/lewm-reference-onnx-contract-check

onnx verify: variant=onnxruntime ok=true batches=1,2 encoder_shape=(2, 192) predictor_shape=(2, 3, 192) action_dim=10
onnx verify: variant=tract-compat ok=true batches=1 encoder_shape=(1, 192) predictor_shape=(1, 3, 192) action_dim=10
```

This validates the exporter against a valid full-layout checkpoint. It does not
complete F1 because the source is the locked upstream reference conversion, not
the missing lewm-rs 50k PushT training checkpoint.

## Required Resolution

F1 can be completed only after one of these is true:

1. A real full Burn/Jepa PushT checkpoint is produced and uploaded with the
   255-tensor Burn destination safetensors layout expected by
   `python/export_onnx.py`.
2. The release acceptance criteria are changed to support the bounded
   `PushtFullLewmCore` checkpoint with a separate ONNX exporter and evaluator.

Until then, F2, F4, and F7 remain blocked by the absence of a trained full
PushT ONNX artifact.

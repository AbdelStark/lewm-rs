# Inference and Export Report

**Date:** 2026-05-15
**Artifacts:** `abdelstark/lewm-rs-pusht` (ONNX files)

## ONNX Export Pipeline

The export script (`python/export_onnx.py`) converts a Burn `.safetensors`
checkpoint to two ONNX variants:

| Variant | Opset | Target | File |
|---------|-------|--------|------|
| onnxruntime | 18 (dynamo) | onnxruntime inference | `encoder.onnx`, `predictor.onnx` |
| Tract-compat | 17 (legacy) | Tract CPU runner | `tract-compat/encoder.onnx`, `tract-compat/predictor.onnx` |

The Tract-compat variant differs in:
- `torch.onnx.export` with `dynamo=False` (legacy exporter, opset 17)
- Fixed batch dimension (batch=1 baked in)
- Causal-mask buffer materialized as a named input

Both variants are uploaded to `abdelstark/lewm-rs-pusht`.

### Export Metadata (`onnx_export.json`)

```json
{
  "schema_version": "1.0.0",
  "source": "reference_checkpoint",
  "config": {
    "action_dim": 10,
    "history_size": 3,
    "latent_dim": 192,
    "image_size": 224
  }
}
```

Note: `action_dim=10` is the smoothed action dimension used by the predictor.
For the PushT reference checkpoint, this is the frameskip-packed action
($A_p = 2 \cdot 5 = 10$); the encoder's Conv1d is `(10, 10, k=1)`, a
per-timestep linear lift, not a temporal smoother.

## Tract CPU Benchmark

Benchmark run: `lewm-infer bench` on Apple M3 ARM, release build.

| Metric | Value |
|--------|-------|
| Hardware | Apple M3 (ARM, 8-core) |
| Build | Release (`cargo build --release`) |
| Episodes | 10 |
| CEM iterations | 5 |
| CEM candidates | 1,024 |
| Median (p50) | **4.08 s/episode** |
| p95 | 4.13 s/episode |
| Backend | Tract 0.22.1 (pre-compiled ONNX engine) |

**Note:** Debug and release builds produce identical latency because the hot
path is Tract's pre-compiled ONNX engine — Rust optimization level does not
affect Tract's execution. The benchmark is CPU-only; GPU inference is not
supported by Tract.

### Command

```bash
cargo build --release -p lewm-infer
./target/release/lewm-infer bench \
  --checkpoint-dir /path/to/tract-compat/ \
  --action-dim 10 \
  --episodes 10
```

## onnxruntime Verification

Inference verified with onnxruntime (Python):

```python
import onnxruntime as ort
import numpy as np

enc_sess = ort.InferenceSession("encoder.onnx")
pred_sess = ort.InferenceSession("predictor.onnx")

# Encoder: [B, T, C, H, W] → [B, T, N+1, D]
frames = np.random.randn(1, 3, 3, 224, 224).astype(np.float32)
latents = enc_sess.run(None, {"frames": frames})[0]  # [1, 3, 197, 192]

# Predictor: [B, T, N+1, D], [B, T, action_dim] → [B, T, N+1, D]
actions = np.random.randn(1, 3, 10).astype(np.float32)
pred = pred_sess.run(None, {"latents": latents, "actions": actions})[0]
```

## Demo Space

The live Gradio demo at `abdelstark/lewm-rs-demo` loads the Tract-compat ONNX
files from Hub at startup. It runs CEM planning in-browser via a Python backend
(`onnxruntime`), selecting actions by minimizing predicted latent cost.

- Space: `https://huggingface.co/spaces/abdelstark/lewm-rs-demo`
- Gradio SDK: 5.33.0 (Python 3.13 compatible)
- Action dim: auto-detected from `onnx_export.json`

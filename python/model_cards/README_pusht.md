---
license: mit
tags:
  - robotics
  - world-model
  - jepa
  - pusht
  - rust
  - burn
language:
  - en
library_name: lewm-rs
---

# lewm-rs — PushT Training Artifact

A Rust/Burn implementation of **LeWorldModel** (Le-WM) trained on the [PushT expert dataset](https://huggingface.co/datasets/quentinll/lewm-pusht).

This repository currently publishes the 50k-step PushT training logs and a
bounded-core training checkpoint. It does **not** yet publish the full
255-tensor Burn/Jepa mirror required for trained-checkpoint ONNX export,
CEM planning evaluation, or the final release model card.

Current release status:

| Item | Status |
|------|--------|
| 50k PushT training run | Complete |
| Uploaded checkpoint | Bounded `PushtFullLewmCore` artifact, 14 tensors |
| Full Burn/Jepa safetensors | Missing |
| Release ONNX under `onnx-full/` | Blocked |
| CEM planning success rate | Blocked |

## Training Results

| Metric | Value |
|--------|-------|
| Steps | 50,000 |
| Wall time | 318 min on A10G-large |
| Initial loss | 0.4912 |
| Final loss | 3.17e-06 |
| Gradient explosions | 0 |
| Job | `abdelstark/6a06f0c43308d79117b90276` |

**Loss curve:**

| Step | Total loss | SIGReg | Pred loss |
|-----:|-----------|--------|-----------|
| 1 | 4.91e-01 | 4.90e-01 | 6.82e-04 |
| 1,000 | 8.69e-02 | 8.69e-02 | 8.43e-07 |
| 5,000 | 6.09e-06 | 4.96e-06 | 1.13e-06 |
| 25,000 | 1.92e-06 | 1.72e-06 | 1.93e-07 |
| 50,000 | 3.17e-06 | 3.00e-06 | 1.69e-07 |

## Architecture Target and Current Artifact

LeWorldModel is a JEPA-based world model combining:

- **ViT-Tiny visual encoder** — 192-dim, 12 transformer layers, 3 attention heads, 14×14 patch tokens from 224×224 images
- **Action encoder** — Conv1d smoother + 2-layer SiLU MLP mapping 2-DOF action history to 192-dim embeddings (smoothed dim=10)
- **AdaLN-zero autoregressive predictor** — 6 transformer blocks, 16 heads, 2048 MLP, with AdaLN-zero conditioning
- **Projector / Pred-proj MLPs** — 192 → 2048 → 192 with BatchNorm1d

Total parameters: ~18M (303 tensors)

The uploaded `train/pusht-full-lewm-20260515T100908Z/step_0050000.*`
checkpoint is not that full 18M-parameter artifact. It is the bounded host-core
training path used by the current trainer and contains 14 exported tensors.

```
Encoder:   ViT-Tiny (192-d, 12L, 3H, patch=14, img=224)
Predictor: 6 blocks, 16 heads, dim=192, mlp=2048
Action:    2-DOF → smooth10 → emb192 (mlp_scale=4)
Training:  SIGReg + prediction MSE loss (λ=1.0, knots=17, proj=1024)
```

## Training Details

| Field | Value |
|-------|-------|
| Dataset | `quentinll/lewm-pusht` (PushT expert demonstrations) |
| Hardware | A10G-large (HuggingFace Jobs) |
| Precision | bf16 mixed |
| Steps | 50,000 |
| Batch size | 64 |
| Optimizer | AdamW (lr=3e-4 → 1e-5, warmup=1000, wd=0.05, β=[0.9, 0.95]) |
| Grad clip | 1.0 |
| History size | 3 frames |
| Prediction horizon | 4 frames |
| Frameskip | 5 |
| Seed | 0 |

## CEM Planning Evaluation

CEM (Cross-Entropy Method) planning evaluation is blocked until a full
Burn/Jepa PushT checkpoint is available and exported to `onnx-full/`. Target:
≥ 87% success rate on 50 test episodes (matching the reference paper).

## Parity Verification

The lewm-rs implementation has activation-level parity coverage against the
reference PyTorch checkpoint:

| Component | Tolerance | Status |
|-----------|-----------|--------|
| Encoder | L∞ < 1e-4 | ✅ PASS |
| Action encoder | L∞ < 1e-4 | ✅ PASS |
| Predictor | L∞ < 1e-4 | ✅ PASS |
| Pred-proj MLP | L∞ < 1e-4 | ✅ PASS |
| SIGReg loss | \|Δ\| < 1e-3 | ✅ PASS |

Reference checkpoint: [`quentinll/lewm-pusht@22b330c`](https://huggingface.co/quentinll/lewm-pusht)

These parity tests validate the implementation surfaces. They are not evidence
that the current 14-tensor PushT upload is the full release checkpoint.

## Artifacts

| File | Description |
|------|-------------|
| `train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors` | Bounded-core parameter mirror (14 tensors, ~1.2 KB) |
| `train/pusht-full-lewm-20260515T100908Z/step_0050000.mpk` | Bounded-core Burn checkpoint mirror |
| `train/pusht-full-lewm-20260515T100908Z/train_report.json` | Training summary |
| `encoder.onnx` + `predictor.onnx` | Reference ONNX export, not the final F1 trained-checkpoint export |
| `tract-compat/encoder.onnx` + `tract-compat/predictor.onnx` | Reference Tract-compatible export, not the final F1 trained-checkpoint export |
| `onnx-full/` | Missing; blocked until a full PushT checkpoint exists |

## CPU Inference (Tract)

Tract benchmark from the reference export path: **4.08 s/episode** (p50,
release build, Apple M3 ARM, 5 CEM iter × 1024 candidates).

Do not run the release exporter against the current
`step_0050000.safetensors`; it is the 14-tensor bounded-core artifact and will
not satisfy the full PyTorch/Burn key mapping. The release path is:

```bash
# Export once the full 255-tensor PushT Burn mirror is available.
uv run --extra parity python python/export_onnx.py \
  --safetensors <full-pusht-step_0050000.safetensors> \
  --meta tests/fixtures/reference_model.meta.json \
  --output-dir /tmp/lewm-onnx-full \
  --variant both \
  --action-dim 10

# Verify both ONNX layouts before upload.
uv run --extra parity python python/verify_onnx.py --dir /tmp/lewm-onnx-full
```

## Repository

- **Training code**: [AbdelStark/lewm-rs](https://github.com/AbdelStark/lewm-rs)
- **Reference model**: [quentinll/lewm-pusht](https://huggingface.co/quentinll/lewm-pusht)
- **Demo Space**: [abdelstark/lewm-rs-demo](https://huggingface.co/spaces/abdelstark/lewm-rs-demo)
- **SO-100 model**: [abdelstark/lewm-rs-so100](https://huggingface.co/abdelstark/lewm-rs-so100)

## License

MIT. See [LICENSE](https://github.com/AbdelStark/lewm-rs/blob/main/LICENSE).

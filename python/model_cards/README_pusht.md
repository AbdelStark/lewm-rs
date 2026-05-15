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

# lewm-rs — PushT Trained Checkpoint

A Rust/Burn implementation of **LeWorldModel** (Le-WM) trained on the [PushT expert dataset](https://huggingface.co/datasets/quentinll/lewm-pusht).

This checkpoint achieves **numerical parity** with the reference PyTorch implementation from [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) (L∞ < 1e-4 on all activation-level parity tests).

## Architecture

LeWorldModel is a JEPA-based world model combining:

- **ViT-Tiny visual encoder** — 192-dim, 12 transformer layers, 3 attention heads, 14×14 patch tokens from 224×224 images
- **Action encoder** — Conv1d smoother + 2-layer SiLU MLP mapping 2-DOF action history to 192-dim embeddings
- **AdaLN-zero autoregressive predictor** — 6 transformer blocks, 16 heads, 2048 MLP, with AdaLN-zero conditioning
- **Projector / Pred-proj MLPs** — 192 → 2048 → 192 with BatchNorm1d

Total parameters: ~18M

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
| Batch size | 64 (grad_accum=2 → effective 128) |
| Optimizer | AdamW (lr=3e-4 → 1e-5, warmup=1000, wd=0.05, β=[0.9, 0.95]) |
| Grad clip | 1.0 |
| History size | 3 frames |
| Prediction horizon | 4 frames |
| Frameskip | 5 |
| Seed | 0 |

## CEM Planning Evaluation

CEM (Cross-Entropy Method) planning evaluation on 50 test episodes:

| Metric | Value |
|--------|-------|
| Success rate | TBD (pending eval run) |
| Avg episode length | TBD |
| CEM candidates | 1000 |
| CEM elite | 100 |
| Planning horizon | 5 |

## Parity Verification

All 10 activation-level parity tests pass against the reference PyTorch checkpoint:

| Component | Tolerance | Status |
|-----------|-----------|--------|
| Encoder | L∞ < 1e-4 | ✅ PASS |
| Action encoder | L∞ < 1e-4 | ✅ PASS |
| Predictor | L∞ < 1e-4 | ✅ PASS |
| Pred-proj MLP | L∞ < 1e-4 | ✅ PASS |
| SIGReg loss | \|Δ\| < 1e-3 | ✅ PASS |

Reference checkpoint: [`quentinll/lewm-pusht@22b330c`](https://huggingface.co/quentinll/lewm-pusht)

## Usage

### Inference with Tract (CPU)

```bash
# Export to ONNX
uv run --extra parity python python/export_onnx.py \
  --safetensors train/RUNID/checkpoint_final.safetensors \
  --meta tests/fixtures/reference_model.meta.json \
  --output-dir /tmp/lewm-onnx

# CPU benchmark
lewm-infer bench --checkpoint-dir /tmp/lewm-onnx
```

### CEM Planning

```bash
lewm-train eval \
  --config configs/pusht.toml \
  --checkpoint-dir train/RUNID
```

## Repository

- **Training code**: [AbdelStark/lewm-rs](https://github.com/AbdelStark/lewm-rs)
- **Reference model**: [quentinll/lewm-pusht](https://huggingface.co/quentinll/lewm-pusht)
- **Paper**: [Le-WM: Learning World Models in Latent Space](https://arxiv.org/abs/2502.16560)
- **Demo Space**: [abdelstark/lewm-rs-demo](https://huggingface.co/spaces/abdelstark/lewm-rs-demo)

## License

MIT. See [LICENSE](https://github.com/AbdelStark/lewm-rs/blob/main/LICENSE).

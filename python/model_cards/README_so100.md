---
license: mit
tags:
  - robotics
  - world-model
  - jepa
  - so100
  - manipulation
  - rust
  - burn
language:
  - en
library_name: lewm-rs
---

# lewm-rs — SO-100 Trained Checkpoint

A Rust/Burn implementation of **LeWorldModel** (Le-WM) trained on the SO-100 pick-and-place dataset.

This checkpoint is trained on the `abdelstark/so100-pickplace-lewm-ready` dataset (1.9 GB HDF5, 6,559 timesteps, 50 episodes at 10fps) with 6-DOF manipulation.

## Architecture

Same ViT-Tiny JEPA architecture as the PushT model, adapted for 6-DOF robotic manipulation:

- **ViT-Tiny visual encoder** — 192-dim, 12 transformer layers, 3 attention heads, 14×14 patch tokens from 224×224 top-view images
- **Action encoder** — 6-DOF joint actions (smoothed to 10-dim) → 192-dim embeddings
- **AdaLN-zero autoregressive predictor** — 6 transformer blocks, 16 heads, 2048 MLP
- **Projector / Pred-proj MLPs** — 192 → 2048 → 192 with BatchNorm1d

Total parameters: ~18M

```
Encoder:   ViT-Tiny (192-d, 12L, 3H, patch=14, img=224, top-view camera)
Predictor: 6 blocks, 16 heads, dim=192, mlp=2048
Action:    6-DOF → smooth10 → emb192 (mlp_scale=4)
Training:  SIGReg + prediction MSE loss (λ=1.0, knots=17, proj=1024)
```

## Training Details

| Field | Value |
|-------|-------|
| Dataset | `abdelstark/so100-pickplace-lewm-ready` |
| Hardware | A10G-large (HuggingFace Jobs) |
| Precision | bf16 mixed |
| Steps | 5,000 |
| Batch size | 64 (grad_accum=2 → effective 128) |
| Optimizer | AdamW (lr=3e-4 → 1e-5, warmup=500, wd=0.05, β=[0.9, 0.95]) |
| Grad clip | 1.0 |
| History size | 3 frames |
| Prediction horizon | 4 frames |
| Camera | Top view |
| Seed | 0 |

## Warm-Start Evaluation

SO-100 latent rollout evaluation:

| Metric | Value |
|--------|-------|
| Episode success rate | TBD (pending eval run) |
| Eval episodes | 5 (ids: [5, 14, 23, 31, 42]) |

## Dataset

- **Source**: `abdelstark/so100-pickplace-lewm-ready`
- 50 pick-and-place episodes at 10fps
- Top-view camera (224×224 RGB)
- 6-DOF joint actions
- Includes normalisation stats (`stats.safetensors`)

## Usage

### Inference with Tract (CPU)

```bash
# Export to ONNX
uv run --extra parity python python/export_onnx.py \
  --safetensors train/RUNID/checkpoint_final.safetensors \
  --meta tests/fixtures/reference_model.meta.json \
  --action-dim 6 \
  --output-dir /tmp/lewm-onnx

# CPU benchmark
lewm-infer bench --checkpoint-dir /tmp/lewm-onnx
```

### Warm-Start Evaluation

```bash
lewm-train eval \
  --config configs/so100.toml \
  --checkpoint-dir train/RUNID
```

## Repository

- **Training code**: [AbdelStark/lewm-rs](https://github.com/AbdelStark/lewm-rs)
- **Reference model**: [quentinll/lewm-pusht](https://huggingface.co/quentinll/lewm-pusht)
- **Paper**: [Le-WM: Learning World Models in Latent Space](https://arxiv.org/abs/2502.16560)
- **Demo Space**: [abdelstark/lewm-rs-demo](https://huggingface.co/spaces/abdelstark/lewm-rs-demo)

## License

MIT. See [LICENSE](https://github.com/AbdelStark/lewm-rs/blob/main/LICENSE).

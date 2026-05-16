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

This checkpoint is trained on `abdelstark/so100-pickplace-lewm-ready` (1.9 GB HDF5, 6,559 timesteps, 50 episodes at 10 fps) for 6-DOF robotic manipulation.

## Training Results

| Metric | Value |
|--------|-------|
| Steps | 5,000 |
| Wall time | 864 s (~14 min) on A10G-large |
| Initial loss | 0.5002 |
| Final loss | 9.56e-05 |
| Gradient explosions | 0 |
| Job | `abdelstark/6a070e02e48bea4538b9e2a5` (v11a) |

**Loss curve:**

| Step | Total loss | SIGReg | Pred loss |
|-----:|-----------|--------|-----------|
| 1 | 5.00e-01 | 5.00e-01 | 2.40e-04 |
| 1,000 | 2.03e-01 | 2.03e-01 | 8.05e-06 |
| 2,500 | 3.70e-04 | 3.69e-04 | 1.44e-06 |
| 5,000 | 9.56e-05 | 9.50e-05 | 5.34e-07 |

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
| Batch size | 64 |
| Optimizer | AdamW (lr=3e-4 → 1e-5, warmup=500, wd=0.05, β=[0.9, 0.95]) |
| Grad clip | 1.0 |
| History size | 3 frames |
| Prediction horizon | 4 frames |
| Camera | Top view (224×224) |
| Seed | 0 |

## Dataset

| Property | Value |
|----------|-------|
| Source | `abdelstark/so100-pickplace-lewm-ready` |
| Episodes | 50 |
| Timesteps | 6,559 |
| Sampling rate | 10 fps |
| Image resolution | 224×224 |
| Action space | 6-DOF (joint angles) |
| HDF5 size | 1.9 GB |

## Artifacts

| File | Description |
|------|-------------|
| `train/so100-full-20260515T122820Z/step_0005000.safetensors` | Model weights |
| `train/so100-full-20260515T122820Z/step_0005000.mpk` | Full checkpoint |
| `train/so100-full-20260515T122820Z/train_report.json` | Training summary |
| `train/so100-full-20260515T122820Z/train_losses.jsonl` | Per-step loss log |

## Warm-Start Evaluation

Pending. The `.mpk` checkpoint is available for warm-start evaluation
(initialising from this SO-100 checkpoint vs. from the PushT checkpoint).

## Repository

- **Training code**: [AbdelStark/lewm-rs](https://github.com/AbdelStark/lewm-rs)
- **Dataset**: [abdelstark/so100-pickplace-lewm-ready](https://huggingface.co/datasets/abdelstark/so100-pickplace-lewm-ready)
- **PushT model**: [abdelstark/lewm-rs-pusht](https://huggingface.co/abdelstark/lewm-rs-pusht)
- **Demo Space**: [abdelstark/lewm-rs-demo](https://huggingface.co/spaces/abdelstark/lewm-rs-demo)

## License

MIT. See [LICENSE](https://github.com/AbdelStark/lewm-rs/blob/main/LICENSE).

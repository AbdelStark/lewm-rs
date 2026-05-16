# Architecture at a glance

> **Motivation.** Part II is a tour of the LeWM model from the outside
> in: the wrapper that owns everything, the four modules it composes,
> and the precise tensor shapes on every edge of the dataflow graph.
> This page is the map.
>
> **Position.** Top of [Part II — Architecture](../introduction.md).
>
> **What you should leave with.** A clear picture of how the four
> sub-modules of `lewm_core::Jepa` compose, what each one's input and
> output shapes are, and where to find the byte-level spec for each
> component.

## 1. The four modules

`crates/lewm-core` defines exactly four sub-modules that together
constitute the LeWM model:

| Module | Source | Parameters | Spec |
|--------|--------|-----------:|------|
| `Vit` (encoder) | `crates/lewm-core/src/vit.rs` | ~5.5 M | [RFC 0002 §4.2](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0002-core-model-architecture.md#42-vit-encoder) |
| `Embedder` (action) | `crates/lewm-core/src/embedder.rs` | ~0.2 M | [RFC 0002 §4.5](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0002-core-model-architecture.md#45-embedder) |
| `Mlp` (projector + pred-proj) | `crates/lewm-core/src/mlp.rs` | ~1.8 M | [RFC 0002 §4.4](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0002-core-model-architecture.md#44-projector-mlp) |
| `ArPredictor` (predictor) | `crates/lewm-core/src/predictor.rs` | ~10.5 M | [RFC 0002 §4.7](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0002-core-model-architecture.md#47-arpredictor) |
| **Total** | | **18 042 672** (303 tensors) | |

The top-level wrapper `Jepa` (`crates/lewm-core/src/jepa.rs`) owns all
four and provides forward, rollout, and cost entry points.

## 2. The composition

```text
                  pixels        actions        (raw)
              (B, T, 3, 224, 224)  (B, T-1, A)
                  │                    │
                  ▼                    ▼
              ┌───────┐         ┌───────────┐
              │  Vit  │         │ Embedder  │
              │       │         │           │
              │ HF    │         │ Conv1d-k1 │
              │ ViT-T │         │ + SiLU MLP│
              └───┬───┘         └─────┬─────┘
                  │                   │
              (B, T, 192)         (B, T-1, 192)
                  │                   │
                  ▼                   │
              ┌────────┐              │
              │projector│             │
              │ (Mlp)   │             │
              │ 192→1024│             │
              └────┬────┘             │
                   │                  │
                (B, T, 1024)          │
                   │                  │
                   │                  ▼
                   │           ┌──────────────┐
                   └──────────▶│ ArPredictor   │
                               │ AdaLN-zero    │
                               │ 6 blocks,     │
                               │ 16 heads      │
                               └──────┬───────┘
                                      │
                                  (B, T-1, 192)
                                      │
                                      ▼
                               ┌────────────┐
                               │ pred_proj  │
                               │   (Mlp)    │
                               │  192→1024  │
                               └─────┬──────┘
                                     │
                                 (B, T-1, 1024)
                                     │
                                     ▼  ẑ_next, the source-arm prediction
```

The output of `pred_proj` is the **source arm** of the prediction loss.
The output of `projector` (with `T-1` frames shifted to align with the
prediction target) is the **target arm**. Both arms share the same
encoder; no EMA, no stop-gradient.

## 3. The wrapper entry points

`Jepa` exposes three public methods, listed in
`crates/lewm-core/src/jepa.rs`:

```rust,ignore
/// Encode a windowed image tensor to embeddings.
///
/// # Shape
/// - input  `pixels: (B, T, C, H, W)`
/// - output `(B, T, D)`
pub fn encode(&self, pixels: Tensor<B, 5>) -> Tensor<B, 3>;

/// Predict the next-step latent given a history of latents and actions.
///
/// # Shape
/// - input  `latents : (B, T, D)`
///          `actions : (B, T, A)`
/// - output `(B, T, D)` predicted latents
pub fn predict(&self, latents: Tensor<B, 3>, actions: Tensor<B, 3>) -> Tensor<B, 3>;

/// Compute the planning cost between a candidate latent rollout and a goal latent.
///
/// # Shape
/// - input  `pred_z : (B, D)`
///          `goal_z : (B, D)`
/// - output `(B,)` per-sample MSE cost
pub fn get_cost(&self, pred_z: Tensor<B, 2>, goal_z: Tensor<B, 2>) -> Tensor<B, 1>;
```

The training loop uses `encode` to produce both source and target
latents, then `predict` for the autoregressive rollout. The planner
uses `encode` once on the current observation and once on the goal,
then `predict` many times in batch, and `get_cost` to score each
rollout.

## 4. Shape inventory at a glance

| Symbol | Meaning | LeWM PushT value |
|--------|---------|------------------|
| $B$ | Batch | 64 (effective 128 with accum 2) |
| $T$ | Window length | 3 (history frames) |
| $C$ | Channels | 3 |
| $H, W$ | Image size | 224, 224 |
| $D$ | Embedding dim | 192 |
| $A$ | Raw action dim | 2 (PushT) / 6 (SO-100) |
| $A_p$ | Packed action dim | 10 (Conv1d smoother output) |
| $E_a$ | Action embedding dim | 192 (matches $D$) |
| $\text{proj}$ | Projection dim | 1024 (SIGReg space) |
| $K$ | SIGReg projections | 1024 |
| $J$ | SIGReg knots | 17 |

These are the shapes that appear throughout the rest of Part II. The
[shape contracts](./shape-contracts.md) page is the single source of
truth for which shape is allowed at which boundary.

## 5. How to read the rest of Part II

The following pages drill into each module:

- **[The ViT-Tiny encoder](./encoder.md)** — patch embed, position
  embeddings, attention, MLP, the 12-layer stack, the CLS read-off.
- **[The autoregressive predictor](./predictor.md)** — AdaLN-zero
  blocks, causal mask, the 6-block stack.
- **[The action encoder](./action-encoder.md)** — Conv1d smoother, MLP
  lift to embedding dim.
- **[Projector and pred-proj MLPs](./projector.md)** — the two MLPs
  that bracket SIGReg.
- **[The `Jepa` wrapper and rollout](./jepa-wrapper.md)** — the
  top-level forward, autoregressive rollout, cost function.
- **[Shape contracts and tensor flow](./shape-contracts.md)** — the
  full edge-by-edge tensor shape table.
- **[Parameter inventory](./parameter-inventory.md)** — the 303-tensor
  parameter table with sizes and roles.

Every page is written assuming familiarity with [Part I — Concepts](../introduction.md).
If you have not read those yet, the [LeWM specialization](../concepts/lewm.md)
page is the cheapest path to context.

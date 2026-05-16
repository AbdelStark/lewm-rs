# The `Jepa` wrapper and rollout

> **Motivation.** The four sub-modules of LeWM are owned by a single
> top-level struct, `Jepa`, that provides three forward entry points
> (`encode`, `predict`, `get_cost`) and an autoregressive `rollout`
> helper. This page documents what `Jepa` exposes and how the training
> loop and planner use it.
>
> **Position.** Fifth page in [Part II](./overview.md).
>
> **What you should leave with.** A clear picture of `Jepa`'s public
> API, the rollout invariants, and the place each entry point shows up
> in `lewm-train` and `lewm-plan`.

## 1. The struct

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct Jepa<B: Backend> {
    config:       JepaConfig,
    vit:          Vit<B>,
    action_enc:   Embedder<B>,
    projector:    Mlp<B>,        // 192 → 2048 → 192 (BN1d + GELU between)
    predictor:    ArPredictor<B>,
    pred_proj:    Mlp<B>,        // 192 → 2048 → 192 (BN1d + GELU between)
}
```

`JepaConfig` aggregates the four sub-configs (`VitConfig`,
`PredictorConfig`, `EmbedderConfig`, `MlpConfig × 2`). Cluster defaults
live in `crates/lewm-core/src/config.rs`.

## 2. Public methods

### 2.1 `encode`

```rust,ignore
/// Encode a windowed image tensor to embeddings.
///
/// # Shape
/// - input  `pixels: (B, T, C, H, W)`
/// - output `(B, T, D)`
pub fn encode(&self, pixels: Tensor<B, 5>) -> Tensor<B, 3>
```

Internally, `encode` reshapes the input to `(B·T, C, H, W)`, runs it
through `vit`, takes the CLS token, and reshapes to `(B, T, D)`. The
output is the *unprojected* encoder embedding; the caller is
responsible for applying `projector` if the loss arm is being computed.

### 2.2 `predict`

```rust,ignore
/// Predict the next-step latent given a history of latents and actions.
///
/// # Shape
/// - input  `latents : (B, T, D)`
///          `actions : (B, T, A)` (raw actions; smoothing happens inside)
/// - output `(B, T, D)` predicted latents
pub fn predict(&self, latents: Tensor<B, 3>, actions: Tensor<B, 3>)
    -> Tensor<B, 3>
```

`predict` runs the action encoder, then the predictor with AdaLN-zero
conditioning. The output's $t$-th frame is the predictor's estimate
of $\mathbf z_{t+1}$ given $\mathbf z_{0..t}$ and the action sequence
leading to it.

### 2.3 `get_cost`

```rust,ignore
/// Compute the planning cost between a candidate latent and a goal latent.
///
/// # Shape
/// - input  `pred_z : (B, D)`
///          `goal_z : (B, D)`
/// - output `(B,)` per-sample squared L2 distance
pub fn get_cost(&self, pred_z: Tensor<B, 2>, goal_z: Tensor<B, 2>)
    -> Tensor<B, 1>
```

The cost is

$$
J(\mathbf z, \mathbf z_{\text{goal}}) = \big\lVert \mathbf z - \mathbf z_{\text{goal}}\big\rVert^2_2.
$$

This is the function the CEM planner minimises (see
[Planning with CEM](../planning/cem.md)).

## 3. The training-loop forward

In `crates/lewm-train/src/step.rs`, one optimizer step looks like
(simplified):

```rust,ignore
// Batch: pixels (B, T+1, 3, 224, 224), actions (B, T_raw, A)
let pixels  = batch.pixels;
let actions = batch.actions;

// Encode every frame in the window
let z = jepa.encode(pixels);                              // (B, T+1, D = 192)

// Apply the projector to all T+1 frames; output dim equals input dim
let z_proj = jepa.projector.forward(z);                   // (B, T+1, D = 192)

// Source arm: feed history latents and actions through the predictor
let history_z       = z.narrow(1, 0, T);                   // (B, T, D)
let pred_z          = jepa.predict(history_z, actions);    // (B, T, D)
let pred_z_proj     = jepa.pred_proj.forward(pred_z);      // (B, T, D)

// Target arm: the next-step projected embedding
let target_z_proj   = z_proj.narrow(1, 1, T);              // (B, T, D)

// Losses
let l_pred   = mse(pred_z_proj, target_z_proj);
let l_sigreg = sigreg(z_proj);                             // computed on all (T+1) frames
let l_total  = l_pred + lambda * l_sigreg;
```

Both `projector` and the encoder receive gradient from the prediction
path *and* the target path. SIGReg adds an additional gradient signal
into the projector and encoder via `z_proj`. See
[Gradient flow](../training/gradient-flow.md) for the detailed graph.

## 4. The autoregressive rollout (planning)

The planner uses a different forward pattern, called the *autoregressive
rollout*. Given:

- a current observation $\mathbf o_t$ encoded once to $\mathbf z_0
  \in \mathbb R^D$,
- a horizon $H$, and
- a candidate action sequence $\mathbf a_{1:H}$,

the rollout produces $\hat{\mathbf z}_H \in \mathbb R^D$ for cost
evaluation:

```rust,ignore
let mut history = Tensor::zeros([B, T, D], device);
history = scatter(history, dim=1, index=T-1, src=z_0);          // pad with z_0 at the end

for step in 0..H {
    // Action at this step (already smoothed by Embedder downstream)
    let a_step = actions.narrow(1, step, T);                    // (B, T, A)
    let z_next = jepa.predict(history, a_step);                 // (B, T, D)
    let z_pred = z_next.narrow(1, T-1, 1);                       // (B, 1, D)

    // Slide the window: history ← [history[1:], z_pred]
    history = concat([history.narrow(1, 1, T-1), z_pred], dim=1);
}

let z_final = history.narrow(1, T-1, 1).squeeze(1);              // (B, D)
let cost    = jepa.get_cost(z_final, z_goal);                    // (B,)
```

The sliding-window pattern keeps the predictor's input shape constant
at $(B, T, D)$ throughout the rollout. After $H$ predictions, the
window contains $T-1$ predicted frames; the cost is computed on the
last one.

Importantly, the planner runs this loop **inside Tract (CPU)** or
**inside Burn**, depending on the backend. Both paths share the same
shape contract.

## 5. The cost function for CEM

CEM scores $n_{\text{cand}}$ candidate sequences in parallel by
broadcasting `history` and `goal` across the candidate axis:

```rust,ignore
let history_b = history.unsqueeze::<4>(1).expand([B, n_cand, T, D]);
let goal_b    = goal.unsqueeze::<4>(1).expand([B, n_cand, D]);
// candidates: (B, n_cand, H, A)

for step in 0..H {
    // Run predictor in one big batched call across the candidate axis
    let cand_step = candidates.narrow(2, step, T).reshape([B*n_cand, T, A]);
    let hist_step = history_b.reshape([B*n_cand, T, D]);
    let pred = jepa.predict(hist_step, cand_step);
    // ... slide window ...
}

let z_final = history_b.narrow(...).reshape([B, n_cand, D]);
let cost    = jepa.get_cost(z_final.reshape([B*n_cand, D]),
                             goal_b.reshape([B*n_cand, D]))
                  .reshape([B, n_cand]);
```

This batched broadcast is the entire reason ONNX export keeps the
predictor and encoder as **separate graphs**: the planner needs to run
the predictor `n_cand × H` times per step but the encoder only once
per step. See [ONNX export](../inference/onnx-export.md).

## 6. Source pointers

| Topic | Source |
|-------|--------|
| `JepaConfig` and `Jepa` | `crates/lewm-core/src/jepa.rs` |
| Training step | `crates/lewm-train/src/step.rs` |
| CEM rollout (in-Rust) | `crates/lewm-plan/src/cem.rs` |
| CEM rollout (Tract CPU) | `crates/lewm-infer/src/plan.rs` |
| Reference rollout (Python) | `python/pusht_runner.py` |

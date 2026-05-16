# The action encoder

> **Motivation.** Raw robot actions are tiny (2-D for PushT, 6-D for
> SO-100) and arrive at a higher rate than the latent stream. The data
> plane packs `frameskip` consecutive raw actions into one "packed"
> action vector; the action encoder then lifts that packed vector to
> the predictor's embedding dimension so it can drive the AdaLN-zero
> modulation heads downstream.
>
> **Position.** Third module in [Part II](./overview.md).
>
> **What you should leave with.** The split of responsibilities between
> the data plane (frameskip packing) and the encoder (Conv1d k=1 lift
> + 2-layer MLP), the exact shapes at each boundary, and a pointer to
> the source.

## 1. Configuration

`EmbedderConfig` in `crates/lewm-core/src/config.rs`:

| Field          | LeWM PushT | SO-100 | Meaning |
|----------------|-----------:|-------:|---------|
| `input_dim`    | 10         | 6      | Action dim entering the encoder (after any data-plane packing). |
| `smoothed_dim` | 10         | 10     | Output channels of the Conv1d. |
| `emb_dim`      | 192        | 192    | Output embedding dim; matches the predictor's hidden dim $D$. |
| `mlp_scale`    | 4          | 4      | Inner MLP width is `emb_dim * mlp_scale = 768`. |

The two reference tasks reach `EmbedderConfig` by different paths:

- **PushT.** Raw actions are 2-D and arrive at `frameskip = 5` higher
  than the model's step rate, so the data plane packs each block of
  5 raw actions into one $A_p = 10$ vector — hence
  `input_dim = 10 = raw_action_dim * frameskip`.
- **SO-100.** The 6-DOF action stream is already at the model's rate;
  no packing is needed, so `input_dim = 6 = SO100_ACTION_DIM`.

The encoder itself does not know about frameskip — it is purely a
linear lift of whatever dim arrives, followed by an MLP. The
contracts that wire `dataset.raw_action_dim`, `dataset.frameskip`,
and `model.action_encoder.input_dim` are validated in
`crates/lewm-train/src/config.rs`.

The action embedding dimension $E_a$ is fixed at $192$ to match the
encoder's hidden dim $D$. This lets the action stream broadcast cleanly
into the AdaLN-zero modulation heads downstream without an extra linear
projection.

The action embedding dimension $E_a$ is fixed at $192$ to match the
encoder's hidden dim $D$. This lets the action stream broadcast cleanly
into the AdaLN-zero modulation heads downstream without an extra linear
projection.

## 2. Stage 1 — Conv1d (k=1) lift

Actions arrive as a stream `(B, T, input_dim)`. The first stage is a
Conv1d with **kernel size = 1**, **stride = 1**, no padding — i.e. a
learned per-timestep linear lift from `input_dim` to `smoothed_dim`
channels (10 → 10 for PushT, 6 → 10 for SO-100).

```rust,ignore
// crates/lewm-core/src/embedder.rs
let mut smoother = Conv1dConfig::new(config.input_dim, config.smoothed_dim, 1)
    .with_initializer(Initializer::Zeros)
    .init(device);
// asserted: smoother.kernel_size == 1
```

Forward:

```text
actions: (B, T, input_dim)
x = actions.permute([0, 2, 1])                 # (B, input_dim, T)
x = self.smoother.forward(x)                   # (B, smoothed_dim, T) — kernel=1, T unchanged
x = x.permute([0, 2, 1])                       # (B, T, smoothed_dim)
```

The Conv1d preserves the time axis: no temporal smoothing happens here.

### 2.1 Why a Conv1d k=1 and not a Linear?

Functionally, `Conv1d(input_dim, smoothed_dim, kernel=1)` is identical
to a per-timestep `Linear(input_dim, smoothed_dim)`. The upstream
reference (`module.py`) uses `patch_embed = nn.Conv1d(...)` and the
parity tests pin this exact parameter layout, so we keep it.

### 2.2 Where the actual smoothing happens (PushT only)

The temporal "smoothing" of the PushT raw action stream is the
**frameskip packing** done by the data loader: `frameskip = 5`
consecutive raw actions of dim $A = 2$ are concatenated into one packed
action of dim $A_p = A \cdot \text{frameskip} = 10$. This reduces the
action rate to the latent rate and aligns the action stream with the
visual encoder's frame rate. SO-100 needs no packing because its 6-DOF
stream is already at the model's step rate.

## 3. Stage 2 — MLP lift

The lifted stream is then expanded from `smoothed_dim = 10` to
$E_a = 192$ through a 2-layer SiLU MLP with hidden width
`emb_dim * mlp_scale = 768`:

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct Embedder<B: Backend> {
    smoother: burn::nn::conv::Conv1d<B>,     // stage 1: Conv1d(input_dim, 10, k=1)
    fc1: burn::nn::Linear<B>,                // Linear(10  → 768) — mlp_scale = 4
    fc2: burn::nn::Linear<B>,                // Linear(768 → 192)
}
```

Forward (full):

```text
actions: (B, T, input_dim)                     # input_dim = 10 (PushT) or 6 (SO-100)
y = self.smoother.forward(actions)             # (B, T, 10)  — kernel=1 lift
y = silu(self.fc1.forward(y))                  # (B, T, 768)
y = self.fc2.forward(y)                        # (B, T, 192)
return y                                       # action embeddings, ready for AdaLN
```

The output shape is `(B, T, 192)`, exactly the conditioning shape the
predictor's `ConditionalBlock` expects.

### 3.1 Initialisation

All three linear layers (`smoother`, `fc1`, `fc2`) use truncated normal
$\sigma = 0.02$, biases zero. There is no zero-init trick here: the
action encoder is unconditional and benefits from a normal init.

## 4. The action-time-alignment contract

A critical implementation detail: the predictor expects the action
stream to be **aligned** with the latent history so that
`actions[:, t]` is the action that *led into* the observation
`latents[:, t]`. The data pipeline (`lewm-data`) is responsible for
this alignment; see [Data plane](../training/data.md) for the exact
window-sampling rule.

When debugging an off-by-one in this alignment, the typical symptom is
that the prediction loss does not drop below the SIGReg floor — the
predictor cannot find $\mathcal F(\mathbf z_t, \mathbf a_t) \approx
\mathbf z_{t+1}$ because the actions it sees are off by one frame.
Both PushT and SO-100 training runs in [Results](../results/pusht.md)
verify alignment by sanity-checking that the prediction loss drops by
many orders of magnitude — a wrong alignment would stall it.

## 5. Parameter count

Shapes follow Burn's Conv1d convention `(out_channels, in_channels, kernel)`
and Linear convention `(in_features, out_features)`:

| Tensor | Shape (PushT) | Count (PushT) | Shape (SO-100) | Count (SO-100) |
|--------|--------------:|--------------:|---------------:|---------------:|
| `smoother.weight` | $10 \times 10 \times 1$ | 100   | $10 \times 6 \times 1$ | 60 |
| `smoother.bias`   | $10$                    |  10   | $10$                   | 10 |
| `fc1.weight`      | $10 \times 768$         | 7 680 | $10 \times 768$        | 7 680 |
| `fc1.bias`        | $768$                   |  768  | $768$                  | 768 |
| `fc2.weight`      | $768 \times 192$        | 147 456 | $768 \times 192$     | 147 456 |
| `fc2.bias`        | $192$                   |  192  | $192$                  | 192 |
| **Action encoder total** |              | **156 206** |                     | **156 166** |

The two task variants differ only at the `smoother.weight`
input-channel axis. The downstream MLP is identical because
`smoothed_dim = 10` is locked.

## 6. Parity test

`crates/lewm-core/tests/parity_action_encoder.rs` runs the action
encoder on the fixture's raw action stream and compares to the upstream
PyTorch dump.

| Test | Tolerance | Status |
|------|-----------|--------|
| `parity_action_encoder` | $L_\infty < 10^{-4}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |

## 7. Source pointers

| Topic | Source |
|-------|--------|
| `EmbedderConfig` | `crates/lewm-core/src/config.rs` |
| `Embedder`, `ActionSmoother` | `crates/lewm-core/src/embedder.rs` |
| Parity test | `crates/lewm-core/tests/parity_action_encoder.rs` |
| Action normalisation stats | `python/compute_stats.py`, `python/compute_so100_stats.py` |

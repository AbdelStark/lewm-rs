# The action encoder

> **Motivation.** Raw robot actions are tiny (2-D for PushT, 6-D for
> SO-100) and arrive at a higher rate than the latent stream. They need
> to be smoothed and lifted to the predictor's embedding dimension
> before they can be used as AdaLN-zero conditioners.
>
> **Position.** Third module in [Part II](./overview.md).
>
> **What you should leave with.** The two-stage structure (Conv1d
> smoother + MLP lift), the exact shapes at each boundary, and a
> pointer to the source.

## 1. Configuration

`EmbedderConfig` in `crates/lewm-core/src/config.rs`:

| Field | LeWM PushT | SO-100 |
|-------|-----------:|-------:|
| `action_dim` (raw $A$) | 2 | 6 |
| `frameskip` | 5 | 5 |
| `packed_dim` ($A_p$) | 10 | 10 |
| `embed_dim` ($E_a$) | 192 | 192 |
| `mlp_ratio` | 4 | 4 |
| `act` | `silu` | `silu` |

The action embedding dimension $E_a$ is fixed at $192$ to match the
encoder's hidden dim $D$. This is what allows the action stream to
broadcast cleanly into the AdaLN-zero modulation heads downstream
without an extra linear projection.

## 2. Stage 1 — Conv1d smoother

Raw actions arrive as a stream `(B, T_raw, A)`. The first stage is a
single Conv1d with **kernel size = `frameskip` = 5**, **stride = 1**,
no padding, that smooths the action sequence and packs it to dimension
$A_p = 10$.

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct ActionSmoother<B: Backend> {
    smoother: burn::nn::conv::Conv1d<B>,   // Conv1d(A, A_p, kernel=5, stride=1, padding=0)
}
```

Forward:

```text
raw_actions: (B, T_raw, A)
x = transpose(raw_actions, 1, 2)               # (B, A, T_raw)
x = self.smoother.forward(x)                   # (B, A_p, T_smoothed)
x = transpose(x, 1, 2)                         # (B, T_smoothed, A_p)
```

The output time dimension `T_smoothed = T_raw - kernel + 1`.

### 2.1 Why a Conv1d?

The frameskip Conv1d is conceptually a *learned linear interpolation
kernel* that maps `frameskip` adjacent raw actions to one "packed"
action vector. With $A = 2$ raw action dims and `frameskip = 5`, the
Conv1d has $2 \cdot 10 \cdot 5 = 100$ weights + 10 biases — a trivial
parameter footprint, but enough to capture the right temporal pooling.

The smoothing is critical because the raw action stream is *noisier* at
its native rate than the visual encoder's latent stream at frame rate.
Pre-smoothing both denoises and aligns the temporal granularity.

## 3. Stage 2 — MLP lift

The smoothed stream is then lifted from $A_p = 10$ to $E_a = 192$ by a
2-layer SiLU MLP:

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct Embedder<B: Backend> {
    smoother: ActionSmoother<B>,             // stage 1
    fc1: burn::nn::Linear<B>,                // Linear(10  → 768) with mlp_ratio=4
    fc2: burn::nn::Linear<B>,                // Linear(768 → 192)
}
```

Forward (full):

```text
raw_actions: (B, T_raw, A)
y = self.smoother.forward(raw_actions)         # (B, T_smoothed, 10)
y = silu(self.fc1.forward(y))                  # (B, T_smoothed, 768)
y = self.fc2.forward(y)                         # (B, T_smoothed, 192)
return y                                         # action embeddings, ready for AdaLN
```

The output shape is `(B, T_smoothed, 192)`, exactly the conditioning
shape the predictor's `ConditionalBlock` expects.

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

| Tensor | Shape | Count |
|--------|------:|------:|
| `smoother.weight` | $2 \times 10 \times 5$ (PushT) / $6 \times 10 \times 5$ (SO-100) | 100 / 300 |
| `smoother.bias`   | $10$ | 10 |
| `fc1.weight`      | $10 \times 768$ | 7 680 |
| `fc1.bias`        | $768$ | 768 |
| `fc2.weight`      | $768 \times 192$ | 147 456 |
| `fc2.bias`        | $192$ | 192 |
| **Action encoder total** | | **~156 K (PushT)** |

The SO-100 variant differs only in the `smoother.weight` shape (input
channels 6 instead of 2). The downstream MLP is unchanged.

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

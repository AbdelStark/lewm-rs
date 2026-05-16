# Shape contracts and tensor flow

> **Motivation.** When implementing a neural network from a spec, the
> single most common bug is a quietly-wrong shape: a permutation, an
> off-by-one, an unnecessary squeeze. This page is the single source of
> truth for every tensor shape on every edge of the dataflow graph.
>
> **Position.** Reference page in [Part II](./overview.md). Bookmark
> this; the rest of the docs cite it.
>
> **What you should leave with.** A complete shape inventory you can
> use to audit a reimplementation in any framework.

## 1. Top-level shape map

For one training step with batch size $B$, history length $T$, the
shapes are:

```text
   Stage                              Shape
   ─────                              ─────
   batch.pixels                       (B, T+1, 3, 224, 224)    f32
   batch.actions                      (B, T,   A)              f32  (A=2 PushT, 6 SO-100)
   batch.action_norm_stats            (mean: (A,), std: (A,))  f32

   ── encoder ──
   pixels → vit.embeddings            (B·(T+1), 257, 192)      f32
   blocks×12                          (B·(T+1), 257, 192)      f32
   final_norm                         (B·(T+1), 257, 192)      f32
   CLS read-off [:, 0, :]             (B·(T+1), 192)           f32
   reshape                            (B, T+1, 192)            f32   ◀ jepa.encode output

   ── projector ──
   projector(z)                       (B, T+1, 1024)           f32   ◀ z_proj, target arm

   ── action encoder ──
   actions.transpose                  (B, A, T)                f32
   Conv1d (kernel=5)                  (B, 10, T-4) on raw
   transpose                          (B, T-4, 10)
   MLP                                (B, T-4, 192)            f32   ◀ a_emb

   ── predictor ──
   input_proj(history_z)              (B, T, 1024)             f32
   + pos_emb (broadcast)              (B, T, 1024)             f32
   blocks×6 (AdaLN-zero gated)        (B, T, 1024)             f32
   final_norm                         (B, T, 1024)             f32
   output_proj                        (B, T, 192)              f32   ◀ pred_z (192-D)

   ── pred_proj ──
   pred_proj(pred_z)                  (B, T, 1024)             f32   ◀ pred_z_1024, source arm

   ── losses ──
   l_pred  = MSE(pred_z_1024,         scalar                   f32
                 z_proj[:, 1:T+1])
   l_sigreg = SIGReg(z_proj)          scalar                   f32 (computed in F32)
   l_total = l_pred + λ·l_sigreg      scalar                   f32
```

The conventions: **f32** for everything in this table except the BF16
"island" of the predictor's matmuls in mixed-precision training mode.
LayerNorm, AdaLN modulation, and SIGReg always run in F32 (see
[Mixed precision](../training/mixed-precision.md)).

## 2. Edge-by-edge contract

Each row below is one edge of the graph. The contract is normative for
parity: any reimplementation must produce these exact shapes (modulo
permutation conventions noted below).

### 2.1 Encoder edges

| From | To | Shape |
|------|----|-------|
| batch.pixels | vit.patch_embed | $(B \cdot (T+1), 3, 224, 224)$ |
| patch_embed.proj output | flatten | $(B \cdot (T+1), 192, 16, 16)$ |
| flatten + transpose | $+$ CLS + pos_emb | $(B \cdot (T+1), 256, 192)$ |
| with CLS | block_0 input | $(B \cdot (T+1), 257, 192)$ |
| block_0 → block_11 | each block | $(B \cdot (T+1), 257, 192)$ |
| block_11 output | final_norm | $(B \cdot (T+1), 257, 192)$ |
| final_norm output | CLS read-off | $(B \cdot (T+1), 192)$ |
| CLS reshape to $(B, T+1)$ | encode output | $(B, T+1, 192)$ |

Note that the encoder is run on **all $T+1$ frames** in the window, not
just the first $T$. The $T+1$-th frame is needed as the *target* of the
prediction loss.

### 2.2 Projector edge

| From | To | Shape |
|------|----|-------|
| jepa.encode output | projector(192 → 1024) | $(B, T+1, 1024)$ |

`projector` is run on every frame in the window. The first $T$ rows
serve as the predictor's input embedding; the last $T$ rows (i.e.
indices `1..T+1`) serve as the prediction loss target.

### 2.3 Action encoder edges

| From | To | Shape |
|------|----|-------|
| batch.actions | transpose for Conv1d | $(B, A, T)$ |
| Conv1d (in=A, out=10, k=5, stride=1) | output | $(B, 10, T-4)$ |
| transpose back | MLP input | $(B, T-4, 10)$ |
| fc1 + SiLU | | $(B, T-4, 768)$ |
| fc2 | a_emb | $(B, T-4, 192)$ |

This is the canonical action-encoder shape. The output's time
dimension is $T - 4$ because the Conv1d kernel size $k = 5$ consumes 4
trailing frames at the boundary. The data pipeline guarantees that the
*raw* action stream is long enough that the smoothed stream has the
required $T - 4$ frames aligned with the encoder's $T$-frame history;
see [Data plane §3](../training/data.md).

In `lewm-rs` v1 the LeWM defaults are arranged so that **the smoothed
action stream has exactly $T$ frames** ($T = 3$ history with appropriate
upstream framing). See [`crates/lewm-data/src/transform/window.rs`].

### 2.4 Predictor edges

| From | To | Shape |
|------|----|-------|
| history_z = z[:, 0:T, :] | input_proj | $(B, T, 192) \to (B, T, 1024)$ |
| + pos_emb | block_0 | $(B, T, 1024)$ |
| Each ConditionalBlock with a_emb | output | $(B, T, 1024)$ |
| final_norm | output_proj | $(B, T, 1024) \to (B, T, 192)$ |

Causal mask is $(T, T)$ upper-triangular bool, pre-registered as a
buffer. Position embedding is $(1, T, 1024)$, broadcast.

### 2.5 Pred-proj edge

| From | To | Shape |
|------|----|-------|
| predictor output | pred_proj | $(B, T, 192) \to (B, T, 1024)$ |

This is the *source arm* of the prediction loss.

### 2.6 Loss edges

| From | To | Shape |
|------|----|-------|
| pred_z_1024 (source) | mse vs. target | $(B, T, 1024)$ each |
| z_proj (target) | mse | sliced to $(B, T, 1024)$ (indices 1..T+1) |
| z_proj reshape | sigreg input | $(B \cdot (T+1), 1024)$ |
| sigreg (in F32) | scalar | $()$ |
| MSE | scalar | $()$ |

## 3. Shape invariants

The following invariants are checked at runtime by `Jepa::forward`
in `crates/lewm-core/src/jepa.rs`. Violations return
`LewmCoreError::InvalidShape`.

- `pixels.shape[0] == actions.shape[0]` (batch axis).
- `pixels.shape[1] == actions.shape[1] + 1` (one more pixel frame than action).
- `pixels.shape[2] == self.config.encoder.num_channels`.
- `pixels.shape[3] == pixels.shape[4] == self.config.encoder.image_size`.
- `actions.shape[2] == self.config.embedder.action_dim`.

## 4. Permutation conventions

A note on framework conventions: PyTorch and Burn both use
`(B, C, H, W)` for images but differ in `Conv1d` layout. The upstream
LeWM PyTorch code uses `(B, A, T)` for Conv1d input (channels first),
which is the same as Burn's convention, so no permutation is required.

For the transformer blocks, both PyTorch and Burn use `(B, T, D)` as
the canonical "sequence" layout. The attention sublayer reshapes
internally to `(B, num_heads, T, head_dim)` for the scaled dot-product;
the output is reshaped back to `(B, T, D)` before the residual.

## 5. Numerical types

In the *unmixed* (F32) training path, every tensor in the table is F32.
In the BF16-mixed path, the predictor's matmuls run in BF16 with F32
accumulators; the encoder's matmuls run in F32 for parity; the
projector's matmuls run in BF16; SIGReg always runs in F32.

The precision invariants are pinned in [RFC 0014 §4] and discussed in
[Mixed precision](../training/mixed-precision.md).

[`crates/lewm-data/src/transform/window.rs`]: https://github.com/AbdelStark/lewm-rs/blob/main/crates/lewm-data/src/transform/window.rs
[RFC 0014 §4]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0014-performance-engineering.md

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
   batch.actions                      (B, T_raw, A)            f32  (A=2 PushT, 6 SO-100; T_raw = T+4 with frameskip kernel=5)
   batch.action_norm_stats            (mean: (A,), std: (A,))  f32

   ── encoder ──
   pixels → vit.embeddings            (B·(T+1), 257, 192)      f32
   blocks×12                          (B·(T+1), 257, 192)      f32
   final_norm                         (B·(T+1), 257, 192)      f32
   CLS read-off [:, 0, :]             (B·(T+1), 192)           f32
   reshape                            (B, T+1, 192)            f32   ◀ jepa.encode output

   ── projector ──
   fc1 → BN1d → GELU → fc2            (B, T+1, 192)            f32   ◀ z_proj, target arm

   ── action encoder ──
   actions.transpose                  (B, A, T_raw)            f32
   Conv1d (kernel=5)                  (B, 10, T_raw-4) = (B, 10, T)
   transpose                          (B, T, 10)
   MLP (10 → 768 → 192)               (B, T, 192)              f32   ◀ a_emb

   ── predictor (no entry/exit projection) ──
   tokens = z_proj[:, 0:T, :]         (B, T, 192)              f32
   + pos_embed (broadcast)            (B, T, 192)              f32
   blocks×6 (AdaLN-zero gated)        (B, T, 192)              f32
   final LayerNorm                    (B, T, 192)              f32   ◀ pred_z

   ── pred_proj ──
   fc1 → BN1d → GELU → fc2            (B, T, 192)              f32   ◀ pred_z_proj, source arm

   ── losses ──
   l_pred  = MSE(pred_z_proj,         scalar                   f32
                 z_proj[:, 1:T+1])
   l_sigreg = SIGReg(z_proj)          scalar                   f32 (computed in F32)
   l_total = l_pred + λ·l_sigreg      scalar                   f32
```

The conventions: **f32** for everything in this table except the BF16
"island" of the predictor's and projector's matmuls in mixed-precision
training mode. LayerNorm, AdaLN modulation, BatchNorm1d, and SIGReg
always run in F32 (see
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
| jepa.encode output | $\text{Linear}(192 \to 2048)$ | $(B, T+1, 2048)$ |
| BatchNorm1d (feature axis = 2048) | | $(B, T+1, 2048)$ |
| GELU + $\text{Linear}(2048 \to 192)$ | projector output | $(B, T+1, 192)$ |

`projector` is run on every frame in the window. The first $T$ rows
serve as the predictor's input embedding; the last $T$ rows (i.e.
indices `1..T+1`) serve as the prediction loss target.

### 2.3 Action encoder edges

Let $T_{\text{raw}}$ denote the raw action-stream length in one
training window. With Conv1d kernel size $k = 5$ and stride $1$, the
smoothed stream has length $T_{\text{raw}} - 4$. The data pipeline
guarantees $T_{\text{raw}} = T + 4$, so the smoothed stream is
$T$-aligned with the encoder's history.

| From | To | Shape |
|------|----|-------|
| batch.actions | transpose for Conv1d | $(B, A, T_{\text{raw}})$ |
| Conv1d (in=A, out=10, k=5, stride=1) | output | $(B, 10, T_{\text{raw}} - 4) = (B, 10, T)$ |
| transpose back | MLP input | $(B, T, 10)$ |
| fc1 + SiLU | | $(B, T, 768)$ |
| fc2 | a_emb | $(B, T, 192)$ |

In `lewm-rs` v1 the LeWM defaults pin $T = 3$ and $T_{\text{raw}} = 7$,
which yields smoothed length $T_{\text{raw}} - 4 = 3 = T$. See
[`crates/lewm-data/src/transform/window.rs`].

### 2.4 Predictor edges

The predictor operates on the $D = 192$ token dim throughout; there is
no entry-side `input_proj` or exit-side `output_proj`. The attention
sublayer expands internally to $\text{inner\_dim} = \text{heads} \times
\text{dim\_head} = 16 \times 64 = 1024$ for the $Q, K, V$ projection
and contracts back to $192$ via `proj`; the MLP sublayer expands to
$\text{mlp\_dim} = 2048$ and contracts back.

| From | To | Shape |
|------|----|-------|
| `tokens = z_proj[:, 0:T, :]` | $+$ `pos_embed` | $(B, T, 192)$ |
| each `ConditionalBlock` (with `a_emb`) | $\to$ next block | $(B, T, 192)$ |
| `norm` (final affine LayerNorm) | predictor output | $(B, T, 192)$ |

Inside one `ConditionalBlock`:

| From | To | Shape |
|------|----|-------|
| input tokens | `norm1` (affine-free LN) + modulation | $(B, T, 192)$ |
| `attn.qkv` | $Q, K, V$ stack | $(B, T, 3072)$ |
| split, scaled dot-product, recombine | post-attention | $(B, T, 1024)$ |
| `attn.proj` | gated residual addition | $(B, T, 192)$ |
| `norm2` + modulation $\to$ `mlp.fc1` | hidden | $(B, T, 2048)$ |
| GELU + `mlp.fc2` | gated residual addition | $(B, T, 192)$ |
| `adaln(a_emb)` | $6 \times D$ modulation features | $(B, T, 1152)$ |

Causal mask is $(T, T)$ upper-triangular bool, pre-registered as a
buffer inside `CausalSelfAttention`. Position embedding is $(1, T, 192)$,
broadcast.

### 2.5 Pred-proj edge

| From | To | Shape |
|------|----|-------|
| predictor output | $\text{Linear}(192 \to 2048)$ | $(B, T, 2048)$ |
| BatchNorm1d + GELU + $\text{Linear}(2048 \to 192)$ | pred_proj output | $(B, T, 192)$ |

This is the *source arm* of the prediction loss.

### 2.6 Loss edges

| From | To | Shape |
|------|----|-------|
| `pred_z_proj` (source) | mse vs. target | $(B, T, 192)$ each |
| `z_proj` (target) | mse | sliced to $(B, T, 192)$ (indices `1..T+1`) |
| `z_proj` reshape | sigreg input | $(B \cdot (T+1), 192)$ |
| sigreg projection $\mathbf P \in \mathbb R^{K \times D} = \mathbb R^{1024 \times 192}$ | sketched samples | $(B \cdot (T+1), 1024)$ |
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


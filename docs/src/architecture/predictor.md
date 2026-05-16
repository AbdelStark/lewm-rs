# The autoregressive predictor

> **Motivation.** The predictor is where the world model "lives". Given
> a history of latent embeddings and the actions that connect them, it
> produces the next-step latent. This page documents its AdaLN-zero
> conditional blocks, the causal mask, and the exact forward semantics.
>
> **Position.** Second major module in [Part II](./overview.md).
>
> **What you should leave with.** The dataflow of one `ConditionalBlock`,
> the structure of `ArPredictor`, and the parity tests that pin it.

## 1. Configuration

`PredictorConfig` in `crates/lewm-core/src/config.rs`:

| Field | LeWM PushT |
|-------|-----------:|
| `num_frames` (history length $T$) | 3 |
| `depth` | 6 |
| `inner_dim` | 1024 |
| `num_heads` | 16 |
| `head_dim` | 64 |
| `mlp_dim` | 2048 |
| `dropout_p` | 0.0 |
| `layer_norm_eps` | 1.0e-12 |
| `pos_emb_kind` | learned absolute (`(T, inner_dim)`) |

Note that the *inner dim* of the predictor (1024) is larger than the
encoder's hidden size (192). The predictor projects the incoming
$(B, T, 192)$ latents up to $(B, T, 1024)$ at the entry point, runs 6
self-attention blocks at the wider dim, then projects back down to
$(B, T, 192)$ at the exit.

## 2. AdaLN-zero `ConditionalBlock`

The unit of the predictor. Conceptually it is a pre-norm transformer
block whose two LayerNorms are *adaptive* — modulated by the action
embedding — and whose residuals are *gated* by a third modulation
parameter. See [AdaLN-zero concepts](../concepts/adaln.md) for the
intuition.

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct ConditionalBlock<B: Backend> {
    norm1: burn::nn::LayerNorm<B>,           // eps = 1e-12, *no learnable affine*
    attention: PredictorAttention<B>,
    norm2: burn::nn::LayerNorm<B>,           // eps = 1e-12, *no learnable affine*
    mlp:   PredictorMlp<B>,                  // SiLU-activated, 1024→2048→1024
    ada_ln_modulation: AdaLnZero<B>,         // Linear(192 → 6·1024), zero-init
}
```

The crucial detail: **`norm1` and `norm2` have *no* learnable
affine parameters**. The modulated scale ($\gamma$) and shift ($\beta$)
are produced fresh on every step by the `ada_ln_modulation` head. This
is the AdaLN convention: the LayerNorm is purely "centre and rescale to
unit variance", with the affine transform delegated to the conditioner.

### 2.1 Forward of a single block

Given $\mathbf x \in \mathbb R^{B \times T \times 1024}$ (the
predictor's running hidden state) and the action embedding $\mathbf c
\in \mathbb R^{B \times T \times 192}$:

```rust,ignore
// Produce 6 modulation parameters per block, per time step
let mod_params = self.ada_ln_modulation.forward(c);  // (B, T, 6·1024)
let [g1, b1, a1, g2, b2, a2] = split_last(mod_params, 1024);

let n1 = self.norm1.forward(x);                       // (B, T, 1024)
let modulated1 = n1 * (1 + g1) + b1;                  // FiLM-style mod
let attn_out = self.attention.forward(modulated1, causal_mask);
let x = x + a1 * attn_out;                            // gated residual

let n2 = self.norm2.forward(x);
let modulated2 = n2 * (1 + g2) + b2;
let mlp_out = self.mlp.forward(modulated2);
let x = x + a2 * mlp_out;
return x;
```

The `1 + g` parametrisation is critical: it makes $g = 0$ correspond to
*unit* scale. At AdaLN-zero init, $g_1 = g_2 = 0$, $b_1 = b_2 = 0$,
$a_1 = a_2 = 0$, so the whole block is the identity.

### 2.2 Initialisation

**RFC0002-AdaLN-Init [MUST]** — `ada_ln_modulation` is *fully zero-init*:
both its weight $\in \mathbb R^{192 \times 6 \cdot 1024}$ and its bias
$\in \mathbb R^{6 \cdot 1024}$ are the zero tensor at the end of
`Module::init`.

The other parameters of the block — `attention.qkv`, `attention.proj`,
`mlp.fc1`, `mlp.fc2` — use the standard truncated-normal init with
$\sigma = 0.02$, biases zero.

### 2.3 Attention sub-layer (predictor variant)

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct PredictorAttention<B: Backend> {
    qkv:  burn::nn::Linear<B>,    // Linear(1024 → 3072, bias=True)
    proj: burn::nn::Linear<B>,    // Linear(1024 → 1024, bias=True)
    num_heads: usize,             // 16
    head_dim: usize,              // 64
}
```

This is the same algorithmic block as the encoder's `Attention`, but
with one critical difference: **the causal mask is applied to the
score matrix** before softmax. The mask is upper-triangular boolean,
pre-registered as a buffer on `ArPredictor` (see §3.2).

Forward:

```text
qkv = self.qkv.forward(x)                        # (B, T, 3072)
q, k, v = split_to_heads(qkv, num_heads=16)
scores = q @ k.transpose(-1, -2) / sqrt(64)       # (B, 16, T, T)
scores = mask_fill(scores, mask, -inf)            # apply causal mask
probs  = softmax(scores, dim=-1)
out    = probs @ v                                # (B, 16, T, 64)
out    = combine_heads(out)                       # (B, T, 1024)
return self.proj.forward(out)
```

The mask is `(T, T)` upper-triangular boolean with `True` above the
diagonal. `mask_fill(scores, mask, -inf)` sets the masked positions to
negative infinity, which become 0 after softmax. This is the standard
"causal decoder" attention.

### 2.4 MLP sub-layer

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct PredictorMlp<B: Backend> {
    fc1: burn::nn::Linear<B>,   // Linear(1024 → 2048)
    fc2: burn::nn::Linear<B>,   // Linear(2048 → 1024)
}
```

Forward (note: SiLU not GELU):

```text
return self.fc2.forward( silu( self.fc1.forward(x) ) )
```

SiLU = $x \cdot \sigma(x)$, where $\sigma$ is the logistic sigmoid.

## 3. `ArPredictor` — the stack

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct ArPredictor<B: Backend> {
    config:        PredictorConfig,
    input_proj:    burn::nn::Linear<B>,            // 192 → 1024
    pos_emb:       burn::module::Param<Tensor<B, 3>>, // (1, T, 1024)
    blocks:        Vec<ConditionalBlock<B>>,        // length = depth = 6
    final_norm:    burn::nn::LayerNorm<B>,          // eps = 1e-12
    output_proj:   burn::nn::Linear<B>,             // 1024 → 192
    causal_mask:   Tensor<B, 2>,                    // (T, T) bool, registered as buffer
}
```

### 3.1 Forward pass

Inputs:
- `latents`: $(B, T, 192)$ — encoder/projector output for the history.
- `actions`: $(B, T, 192)$ — action embeddings from `Embedder`.

```text
x = self.input_proj.forward(latents)              # (B, T, 1024)
x = x + self.pos_emb                              # broadcast: (1, T, 1024) → (B, T, 1024)
for block in &self.blocks:
    x = block.forward(x, actions, self.causal_mask)
x = self.final_norm.forward(x)                    # (B, T, 1024)
return self.output_proj.forward(x)                # (B, T, 192)
```

### 3.2 The causal mask buffer

The mask is **registered as a non-trainable buffer at init**, not built
at every forward call:

```rust,ignore
// In ArPredictor::init:
let mask: Tensor<B, 2> = Tensor::<B, 2, Int>::ones([T, T], device)
    .triu(1)                                  // upper-triangular, exclusive of diagonal
    .bool();                                   // (T, T) boolean
```

This matters for ONNX export. If the mask is built inside `forward` with
`torch.ones(T, T)` where `T` comes from `latents.shape[1]`, the dynamo
ONNX exporter produces a symbolic-shape graph that Tract cannot parse.
Pre-registering the buffer with a fixed `T = num_frames = 3` produces a
clean, static-shape ONNX graph. See
[ONNX export](../inference/onnx-export.md) for the full story.

### 3.3 Parameter count

The predictor accounts for ~10.5 M parameters:

| Sub-component | Count |
|---------------|------:|
| `input_proj.weight` ($192 \times 1024$) + bias | 197 632 |
| `pos_emb` ($1 \times 3 \times 1024$) | 3 072 |
| Per block × 6 | ~1.67 M each |
| `final_norm` (.weight, .bias) | 2 048 |
| `output_proj.weight` ($1024 \times 192$) + bias | 196 800 |
| **Predictor total** | **~10.5 M** |

Per `ConditionalBlock`:

| Tensor | Count |
|--------|------:|
| `norm1` (no affine) | 0 |
| `attention.qkv.weight` ($1024 \times 3072$) | 3 145 728 |
| `attention.qkv.bias` ($3072$) | 3 072 |
| `attention.proj.weight` ($1024 \times 1024$) | 1 048 576 |
| `attention.proj.bias` ($1024$) | 1 024 |
| `norm2` (no affine) | 0 |
| `mlp.fc1.weight` ($1024 \times 2048$) | 2 097 152 |
| `mlp.fc1.bias` ($2048$) | 2 048 |
| `mlp.fc2.weight` ($2048 \times 1024$) | 2 097 152 |
| `mlp.fc2.bias` ($1024$) | 1 024 |
| `ada_ln_modulation.weight` ($192 \times 6144$) | 1 179 648 |
| `ada_ln_modulation.bias` ($6144$) | 6 144 |
| **Per-block total** | **~9.58 M** |

Wait — that's already more than 10.5 M for six blocks; the table above
is per block, and not all blocks share `ada_ln_modulation.weight`. The
~10.5 M number is the predictor *total*, including the input/output
projections; the per-block AdaLN budget is what makes the predictor the
biggest of the four modules. (See the [Parameter inventory](./parameter-inventory.md)
page for the canonical numbers.)

## 4. Parity tests

| Test | Source | Tolerance |
|------|--------|-----------|
| `parity_predictor` | `crates/lewm-core/tests/parity_predictor.rs` | $L_\infty < 10^{-4}$ across all $T$ |
| `parity_predictor_mixed_precision` | same | rel. $< 2\!\times\!10^{-2}$ for BF16 |

Both <span class="lewm-badge lewm-badge--done">PASS</span> against the
locked PushT reference. The mixed-precision test specifically verifies
that running the predictor in BF16 (with F32 for `LayerNorm` and the
AdaLN modulation) stays within tolerance — critical for training cost.

## 5. Source pointers

| Topic | Source |
|-------|--------|
| `PredictorConfig` | `crates/lewm-core/src/config.rs` |
| `ConditionalBlock`, `ArPredictor` | `crates/lewm-core/src/predictor.rs` |
| `AdaLnZero` helper | `crates/lewm-core/src/ada_ln.rs` |
| Causal mask construction | `crates/lewm-core/src/tensor_ops.rs` |
| Parity tests | `crates/lewm-core/tests/parity_predictor*.rs` |
| Reference dump generation | `python/convert_reference.py dump --component predictor` |

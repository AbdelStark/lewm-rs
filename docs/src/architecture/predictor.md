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
| `hidden_dim` (token dim $D$) | 192 |
| `heads` | 16 |
| `dim_head` | 64 |
| $\text{heads} \times \text{dim\_head}$ (attention inner dim) | 1024 |
| `mlp_dim` (FFN inner dim) | 2048 |
| `action_emb_dim` | 192 |
| `dropout` (after pos add) | 0.1 |
| `emb_dropout` (after pos-emb element-wise add) | 0.0 |
| `layer_norm_eps` (final norm) | 1.0e-5 |
| `pos_emb_kind` | learned absolute, shape $(1, T, D)$ |

The predictor's *token dim* matches the encoder's, $D = 192$.
Internally each block expands to a wider *attention inner dim* of
$\text{heads}\times\text{dim\_head} = 16 \times 64 = 1024$ for the
$Q, K, V$ projection — see §2.3 — and to $\text{mlp\_dim} = 2048$ in
the feed-forward sublayer. The block's input and output dims remain
$192$.

## 2. AdaLN-zero `ConditionalBlock`

The unit of the predictor. Conceptually it is a pre-norm transformer
block whose two LayerNorms are *adaptive* — modulated by the action
embedding — and whose residuals are *gated* by a third modulation
parameter. See [AdaLN-zero concepts](../concepts/adaln.md) for the
intuition.

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct ConditionalBlock<B: Backend> {
    norm1: AffineFreeLayerNorm,        // hidden_dim = 192, no learnable affine
    attn:  CausalSelfAttention<B>,     // inner_dim = 1024 (= heads × dim_head)
    norm2: AffineFreeLayerNorm,        // hidden_dim = 192, no learnable affine
    mlp:   PredictorMlpBlock<B>,       // GELU, 192 → 2048 → 192
    adaln: AdaLNZero<B>,               // Linear(192 → 6·192 = 1152), zero-init
}
```

The crucial detail: **`norm1` and `norm2` have *no* learnable affine
parameters**. The modulated scale ($\gamma$) and shift ($\beta$) are
produced fresh on every step by `adaln` (the conditioning head). This
is the AdaLN convention: the LayerNorm is purely "centre and rescale to
unit variance", with the affine transform delegated to the conditioner.

### 2.1 Forward of a single block

Given the running hidden state $\mathbf x \in \mathbb R^{B \times T
\times D}$ with $D = 192$ and the action embedding
$\mathbf c \in \mathbb R^{B \times T \times E_a}$ with $E_a = 192$:

```rust,ignore
// Produce 6 modulation parameters per block, per time step.
// Each modulation parameter is D-wide (192), so the head emits 6·D = 1152 features.
let mods = self.adaln.forward(c);                     // (B, T, 6·D) = (B, T, 1152)
let [g1, b1, a1, g2, b2, a2] = split_last(mods, D);   // each (B, T, 192)

let n1 = self.norm1.forward(x);                       // (B, T, 192)
let modulated1 = n1 * (1.0 + g1) + b1;                // FiLM-style modulation
let attn_out = self.attn.forward(modulated1);         // causal mask is internal
let x = x + a1 * attn_out;                            // gated residual

let n2 = self.norm2.forward(x);
let modulated2 = n2 * (1.0 + g2) + b2;
let mlp_out = self.mlp.forward(modulated2);
let x = x + a2 * mlp_out;
return x;
```

The `1 + g` parametrisation is critical: it makes $g = 0$ correspond
to *unit* scale. At AdaLN-zero init, $g_1 = g_2 = 0$, $b_1 = b_2 = 0$,
$a_1 = a_2 = 0$, so the whole block is the identity.

### 2.2 Initialisation

**RFC0002-AdaLN-Init [MUST]** — `adaln` is *fully zero-init*: both its
weight $\in \mathbb R^{6D \times E_a} = \mathbb R^{1152 \times 192}$
and its bias $\in \mathbb R^{6D} = \mathbb R^{1152}$ are the zero
tensor at the end of `Module::init`.

The other parameters of the block — `attn.qkv`, `attn.proj`, `mlp.fc1`,
`mlp.fc2` — use the standard truncated-normal init with
$\sigma = 0.02$. The attention QKV layer has no bias (matching
upstream); the projection and MLP layers have zero-initialised biases.

### 2.3 Attention sub-layer (predictor variant)

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    qkv:  burn::nn::Linear<B>,    // Linear(192 → 3·1024 = 3072), bias = false
    proj: burn::nn::Linear<B>,    // Linear(1024 → 192), bias = true
    num_heads: usize,             // 16
    head_dim:  usize,              // 64
    inner_dim: usize,              // num_heads · head_dim = 1024
    scale:     f64,                // 1.0 / sqrt(head_dim) = 0.125
}
```

This is the same algorithmic block as the encoder's `Attention`, but
with one critical difference: **the causal mask is applied to the
score matrix** before softmax. The mask is upper-triangular boolean,
pre-registered as a buffer on the attention module (built once at
init, see §3.2).

Forward:

```text
qkv      = self.qkv.forward(x)                         # (B, T, 3072) — 3·1024
q, k, v  = split_to_heads(qkv, num_heads=16)            # each (B, 16, T, 64)
scores   = q @ k.transpose(-1, -2) * self.scale          # (B, 16, T, T)
scores   = mask_fill(scores, causal_mask, -inf)
probs    = softmax(scores, dim=-1)
out      = probs @ v                                     # (B, 16, T, 64)
out      = combine_heads(out)                            # (B, T, 1024)
return self.proj.forward(out)                            # (B, T, 192)
```

The mask is `(T, T)` upper-triangular boolean with `True` above the
diagonal. `mask_fill(scores, mask, -inf)` sets the masked positions to
negative infinity, which become $0$ after softmax. This is the standard
"causal decoder" attention. Critically, the QKV expansion to $1024$ is
*internal* to the attention sublayer; the token stream exits the
sublayer back at $D = 192$ via `proj`.

### 2.4 MLP sub-layer

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct PredictorMlpBlock<B: Backend> {
    norm: AffineFreeLayerNorm,    // hidden_dim = 192
    fc1:  burn::nn::Linear<B>,    // Linear(192 → 2048)
    fc2:  burn::nn::Linear<B>,    // Linear(2048 → 192)
}
```

Forward (exact-erf GELU):

```text
return self.fc2.forward( gelu( self.fc1.forward( self.norm.forward(x) ) ) )
```

The activation is the exact-erf GELU
$\text{GELU}(x) = x \cdot \Phi(x)$, matching upstream LeWM (not the
fast tanh approximation).

## 3. `ArPredictor` — the stack

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct ArPredictor<B: Backend> {
    pos_embed: Param<Tensor<B, 3>>,    // (1, num_frames, hidden_dim) = (1, 3, 192)
    dropout:   burn::nn::Dropout,      // applied after pos add
    blocks:    Vec<ConditionalBlock<B>>, // length = depth = 6
    norm:      burn::nn::LayerNorm<B>, // affine LayerNorm on the 192-D output
    config:    Ignored<PredictorConfig>,
}
```

Note that — unlike many transformer implementations — there is **no
`input_proj` or `output_proj`**. The encoder, projector, and predictor
all operate on the same token dim $D = 192$, so no entry / exit lift is
needed.

### 3.1 Forward pass

Inputs:
- `tokens`:  $(B, T, D) = (B, T, 192)$ — projected encoder output for
  the history (i.e. `projector(encode(pixels))[:, 0:T, :]`).
- `actions`: $(B, T, E_a) = (B, T, 192)$ — action embeddings from
  `Embedder`.

```text
T_in  = tokens.shape[1]                                   # T_in <= num_frames
pos   = self.pos_embed.narrow(1, 0, T_in)                 # (1, T_in, 192)
x     = self.dropout.forward(tokens + pos)                # (B, T_in, 192)
for block in &self.blocks {
    x = block.forward(x, actions)                          # mask is internal to attn
}
return self.norm.forward(x)                                # (B, T_in, 192)
```

### 3.2 The causal mask buffer

The mask is **registered as a non-trainable buffer at init**, not
built at every forward call:

```rust,ignore
// Inside CausalSelfAttention::init, given num_frames T = 3:
let mask: Tensor<B, 2, Bool> = build_causal_mask(T, device);
// upper-triangular bool tensor of shape (T, T), True above the diagonal
```

This matters for ONNX export. If the mask is built inside `forward`
with `torch.ones(T, T)` where `T` comes from `latents.shape[1]`, the
dynamo ONNX exporter produces a symbolic-shape graph that Tract cannot
parse. Pre-registering the buffer with a fixed
$T = \text{num\_frames} = 3$ produces a clean, static-shape ONNX
graph. See [ONNX export](../inference/onnx-export.md) for the full
story.

### 3.3 Parameter count

The predictor accounts for $\sim 10.8$ M parameters, dominated by the
six `ConditionalBlock`s. There are no entry/exit projections, so the
only non-block parameters are the position embedding and the final
LayerNorm:

| Sub-component | Count |
|---------------|------:|
| `pos_embed` ($1 \times 3 \times 192$) | 576 |
| `norm.weight` + `norm.bias` (final LayerNorm, $192 + 192$) | 384 |
| $6 \times$ `ConditionalBlock` (per-block total below) | 10 785 792 |
| **Predictor total** | **10 786 752 (~10.8 M)** |

Per `ConditionalBlock`:

| Tensor | Shape | Count |
|--------|------:|------:|
| `norm1` (affine-free) | – | 0 |
| `attn.qkv.weight` | $3072 \times 192$ | 589 824 |
| `attn.qkv.bias`   | – (none) | 0 |
| `attn.proj.weight` | $192 \times 1024$ | 196 608 |
| `attn.proj.bias`   | $192$ | 192 |
| `norm2` (affine-free) | – | 0 |
| `mlp.fc1.weight` | $2048 \times 192$ | 393 216 |
| `mlp.fc1.bias`   | $2048$ | 2 048 |
| `mlp.fc2.weight` | $192 \times 2048$ | 393 216 |
| `mlp.fc2.bias`   | $192$ | 192 |
| `adaln.weight`   | $1152 \times 192$ | 221 184 |
| `adaln.bias`     | $1152$ | 1 152 |
| **Per-block total** | | **1 797 632** |

Six blocks: $6 \times 1\,797\,632 = 10\,785\,792$. Together with the
$960$ parameters of the position embedding and final LayerNorm, the
predictor totals $10\,786\,752$ — consistent with the headline
$\sim 10.8$ M cited above. The dominant per-block term is the attention
QKV projection at $\sim 0.59$ M. See
[Parameter inventory §3](./parameter-inventory.md#3-predictor-tensor-breakdown)
for the canonical row-by-row table.

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

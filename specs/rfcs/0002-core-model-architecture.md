---
rfc: "0002"
title: "lewm-core — model architecture, modules, forward semantics"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§4.1", "§5.2", "§5.3", "§14"]
depends_on: ["0001"]
related: ["0003", "0008", "0013"]
---

# RFC 0002 — `lewm-core`: model architecture, modules, forward semantics

> **Status:** Accepted · **Version:** 1.0.0
>
> Specifies every module of `lewm-core` to a level of detail sufficient for byte-exact reproduction of the upstream LeWM model in Rust. Covers the ViT encoder, the action `Embedder`, the projector/pred_proj MLPs, the `ConditionalBlock` and `ArPredictor` with AdaLN-zero conditioning, the top-level `Jepa` wrapper, the rollout and cost functions, and the initialization recipes. Numerical contracts (tolerances, parity tests) are in [RFC 0008](0008-reference-parity-testing.md); loss math is in [RFC 0003](0003-sigreg-and-loss-functions.md).

---

## 1. Introduction

### 1.1 Motivation

LeWM's elegance is its strength and its trap. Two losses, fifteen million parameters, end-to-end. There is no architectural slack to hide a wrong choice behind. A single off-by-one in the position embedding, an init scheme that is mildly wrong, an AdaLN that does not start as identity — any of these will silently degrade training and only show up as a 10-point planning success rate drop after 8 hours of A10G time. This RFC pins down each module to the level of detail that allows independent reimplementation.

### 1.2 Goals

1. Define every public type and method of `lewm-core` with its full signature and semantics.
2. State the exact tensor shape on every edge of the dataflow graph.
3. Specify initialization for every parameter to a degree that reproduces the upstream model byte-for-byte at init time given the same RNG seed.
4. Specify the forward pass of each module to a degree that reproduces the upstream forward to ≤ 1e-4 absolute (F32, same weights, same input).
5. Specify the dataflow and shape contracts for the `Jepa` wrapper, rollout, and cost function.

### 1.3 Non-goals

- Loss math (deferred to [RFC 0003](0003-sigreg-and-loss-functions.md)).
- Training loop, optimizer, scheduling (deferred to [RFC 0005](0005-training-system.md)).
- ONNX export quirks (deferred to [RFC 0007](0007-tract-inference-and-onnx-export.md)).
- Data shapes upstream of the model (deferred to [RFC 0004](0004-data-pipeline.md)).

### 1.4 Stakeholders

Implementers of `lewm-core`, parity-test writers, downstream RFCs (0003, 0005, 0006, 0007, 0008).

---

## 2. Conventions

- All tensor shapes use the symbol conventions of [`glossary.md` §6](../glossary.md).
- Where this RFC says "matches PyTorch reference", the contract is byte-comparable to the dump produced by `python/convert_reference.py` per [RFC 0008](0008-reference-parity-testing.md).
- All Rust signatures assume `use burn::tensor::{Tensor, backend::Backend, Int, Float};` etc. Imports are elided.
- The phrase "**MUST be initialised** to X" means: at the end of `Module::init`, the parameter tensor's value compared element-wise to X under L∞ norm is below the precision of the float type.

---

## 3. Background

The upstream LeWM model is composed from HF `transformers` ViT (used unmodified) plus a custom predictor head defined in `module.py` in `lucas-maes/le-wm`. We re-implement the ViT in Rust (PyTorch's HF code is too heavy to mimic via FFI cleanly) and we hand-port `module.py` line-by-line. The reference weights for parity tests come from `quentinll/lewm-pusht`.

The model architecture is fixed; this RFC is descriptive of an existing design, not a new one. Where the upstream code uses defaults from the HF `ViTConfig`, those defaults are inlined here.

---

## 4. Detailed design

### 4.1 Module overview

```
lewm-core/
└── src/
    ├── lib.rs                      # re-exports, error type
    ├── config.rs                   # JepaConfig, ViTConfig, PredictorConfig, EmbedderConfig
    ├── init.rs                     # initialization helpers (truncated normal, zero, etc.)
    ├── tensor_ops.rs               # masking, bilinear/bicubic position embedding interpolation
    ├── vit.rs                      # PatchEmbed, Attention, MlpBlock, EncoderBlock, Vit
    ├── embedder.rs                 # action Embedder
    ├── mlp.rs                      # projector, pred_proj
    ├── ada_ln.rs                   # AdaLNZero helper used by ConditionalBlock
    ├── predictor.rs                # ConditionalBlock, ArPredictor
    ├── jepa.rs                     # Jepa top-level wrapper
    ├── losses/                     # RFC 0003
    └── export/                     # safetensors export
```

`lewm-core` re-exports the following from `lib.rs`:

```rust
pub use crate::config::{
    EmbedderConfig, JepaConfig, MlpConfig, PredictorConfig, VitConfig,
};
pub use crate::embedder::Embedder;
pub use crate::jepa::Jepa;
pub use crate::mlp::Mlp;
pub use crate::predictor::{ArPredictor, ConditionalBlock};
pub use crate::vit::{EncoderBlock, PatchEmbed, Vit};
```

### 4.2 ViT encoder

#### 4.2.1 Configuration

```rust
/// Vision Transformer configuration. Defaults match HF `transformers` ViT-Small.
#[derive(burn::config::Config, Debug, Clone, Eq, PartialEq)]
pub struct VitConfig {
    /// Square image side length in pixels (must equal `patch_size · grid_size`).
    #[config(default = "224")]
    pub image_size: usize,
    /// Patch side length in pixels.
    #[config(default = "16")]
    pub patch_size: usize,
    /// Input channel count.
    #[config(default = "3")]
    pub num_channels: usize,
    /// Embedding dimension `D`.
    #[config(default = "384")]
    pub hidden_size: usize,
    /// Number of transformer blocks.
    #[config(default = "12")]
    pub num_hidden_layers: usize,
    /// Number of attention heads (`hidden_size % num_heads == 0`).
    #[config(default = "6")]
    pub num_attention_heads: usize,
    /// FFN inner dim; HF ViT uses `mlp_ratio · hidden_size`. Here equals `1536`.
    #[config(default = "1536")]
    pub intermediate_size: usize,
    /// Activation function: GELU with tanh approximation (HF default `gelu`, our default `gelu_tanh`).
    #[config(default = "GeluVariant::TanhApprox")]
    pub hidden_act: GeluVariant,
    /// Attention probabilities dropout.
    #[config(default = "0.0")]
    pub attention_probs_dropout_prob: f64,
    /// FFN/post-attention residual dropout.
    #[config(default = "0.0")]
    pub hidden_dropout_prob: f64,
    /// LayerNorm epsilon.
    #[config(default = "1.0e-12")]
    pub layer_norm_eps: f64,
    /// Whether to include a learnable CLS token. Always true for LeWM.
    #[config(default = "true")]
    pub use_cls_token: bool,
    /// Whether to interpolate position embeddings at forward time (HF semantics).
    /// Setting true is a no-op when `image_size` is constant but exercises the codepath.
    #[config(default = "false")]
    pub interpolate_pos_encoding: bool,
}

#[derive(burn::config::Config, Debug, Clone, Copy, Eq, PartialEq)]
pub enum GeluVariant {
    Erf,
    TanhApprox,
}
```

#### 4.2.2 Patch embedding

```rust
#[derive(burn::module::Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    proj: burn::nn::conv::Conv2d<B>,   // Conv2d(C, hidden, patch_size, stride=patch_size, bias=true)
    num_patches: usize,
    patch_size: usize,
}

impl<B: Backend> PatchEmbed<B> {
    /// Forward.
    ///
    /// # Shape
    /// - input  : `(B, C=num_channels, H=image_size, W=image_size)`
    /// - output : `(B, num_patches, hidden_size)`
    pub fn forward(&self, pixels: Tensor<B, 4>) -> Tensor<B, 3> { /* ... */ }
}
```

**RFC0002-001 [MUST]** — `proj` weight initialization: truncated normal with `std = 0.02` and clipping at `±2 · std`. Bias initialised to zero. (HF `_init_weights` for `nn.Linear`/`nn.Conv2d` with `init_range=0.02` per `ViTConfig`.)

**Forward algorithm (verbatim):**

```
pixels: (B, C, H, W)
x = conv2d_proj(pixels)            # (B, hidden, H/P, W/P)
x = flatten(x, start_dim=2)        # (B, hidden, num_patches)
x = transpose(x, 1, 2)             # (B, num_patches, hidden)
return x
```

`num_patches = (image_size / patch_size) ** 2`. For LeWM defaults: `(224/16)**2 = 196`.

#### 4.2.3 Position embeddings

```rust
#[derive(burn::module::Module, Debug)]
pub struct ViTEmbeddings<B: Backend> {
    patch_embed: PatchEmbed<B>,
    cls_token:  burn::module::Param<Tensor<B, 3>>,   // (1, 1, hidden)
    pos_embed:  burn::module::Param<Tensor<B, 3>>,   // (1, num_patches + 1, hidden)
    dropout:    burn::nn::Dropout,
}
```

**RFC0002-002 [MUST]** — `cls_token` init: truncated normal `std=0.02`, clipped at `±2σ`. The HF impl uses `nn.init.trunc_normal_` with these defaults.

**RFC0002-003 [MUST]** — `pos_embed` init: truncated normal `std=0.02`, clipped at `±2σ`. (Older `transformers` versions used a non-truncated normal; the version pinned by `quentinll/lewm-pusht` uses truncated.)

**Forward algorithm:**

```
patches = patch_embed(pixels)                       # (B, P, D)
cls = cls_token.broadcast_to((B, 1, D))             # (B, 1, D)
x   = concat([cls, patches], dim=1)                 # (B, P+1, D)
pos = interpolate_pos_embed(self.pos_embed, P)      # (1, P+1, D); no-op at default sizes
x   = x + pos
x   = dropout(x)                                    # train only
return x                                            # (B, P+1, D)
```

#### 4.2.4 Position embedding interpolation

When `interpolate_pos_encoding == true`, the position embedding is bicubic-interpolated to match the patch count if the image size differs from the training-time `image_size`. The reference HF implementation in `transformers` v4.45 uses:

```python
def interpolate_pos_encoding(self, embeddings, height, width):
    npatch = embeddings.shape[1] - 1
    N = self.position_embeddings.shape[1] - 1
    if npatch == N and height == width:
        return self.position_embeddings
    class_pos_embed = self.position_embeddings[:, 0]
    patch_pos_embed = self.position_embeddings[:, 1:]
    dim = embeddings.shape[-1]
    h0 = height // self.patch_size
    w0 = width  // self.patch_size
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed, scale_factor=(h0/math.sqrt(N), w0/math.sqrt(N)),
        mode="bicubic", align_corners=False,
    )
    assert patch_pos_embed.shape[-2] == h0 and patch_pos_embed.shape[-1] == w0
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
```

**RFC0002-004 [MUST]** — When `image_size != pretrained_image_size` *and* `interpolate_pos_encoding == true`, the encoder **MUST** perform bicubic interpolation as above. We re-implement in `tensor_ops::interpolate_pos_embed` with the same `align_corners=false` semantics.

**RFC0002-005 [MUST]** — At LeWM defaults (`image_size == 224`), interpolation is a strict no-op (the shortcut branch). The non-shortcut branch is exercised by a dedicated parity test in [RFC 0008 §6](0008-reference-parity-testing.md) at `image_size=192`.

#### 4.2.5 Encoder block

```rust
#[derive(burn::module::Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    norm1: burn::nn::LayerNorm<B>,
    attn:  Attention<B>,
    norm2: burn::nn::LayerNorm<B>,
    mlp:   MlpBlock<B>,
}
```

The block is **pre-norm**:

```
x = x + attn(norm1(x))
x = x + mlp(norm2(x))
```

This matches `nn.LayerNorm` + `nn.MultiheadAttention(batch_first=True)` + residual + `nn.LayerNorm` + `nn.Linear → GELU → nn.Linear` + residual, exactly as in HF ViT.

#### 4.2.6 Attention

```rust
#[derive(burn::module::Module, Debug)]
pub struct Attention<B: Backend> {
    qkv:     burn::nn::Linear<B>,        // (D → 3D), bias=true
    proj:    burn::nn::Linear<B>,        // (D → D), bias=true
    attn_drop: burn::nn::Dropout,
    proj_drop: burn::nn::Dropout,
    num_heads: usize,
    head_dim:  usize,
    scale: f64,                          // 1 / sqrt(head_dim)
}
```

**Forward algorithm:**

```
N = P + 1
x: (B, N, D)
qkv = qkv(x)                       # (B, N, 3D)
qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
q, k, v = qkv.unbind(0)            # each (B, num_heads, N, head_dim)

attn = (q @ k.transpose(-2, -1)) * scale     # (B, num_heads, N, N)
attn = softmax(attn, dim=-1)
attn = attn_drop(attn)                       # train only

out = attn @ v                                # (B, num_heads, N, head_dim)
out = out.transpose(1, 2).reshape(B, N, D)
out = proj(out)
out = proj_drop(out)
return out                                    # (B, N, D)
```

**RFC0002-006 [MUST]** — The attention **MUST NOT** be causal in the encoder. (Encoder is bidirectional; only the predictor uses causal masking.)

**RFC0002-007 [SHOULD]** — When available, the implementation **SHOULD** use a fused SDPA via `burn::tensor::activation::scaled_dot_product_attention` or backend-specific kernels for performance. The fused path **MUST** be byte-equivalent to the explicit path to numerical precision.

**Init:**

- `qkv.weight`, `proj.weight`: truncated normal `std=0.02`, clipped at `±2σ`.
- Biases: zero.

#### 4.2.7 MLP block

```rust
#[derive(burn::module::Module, Debug)]
pub struct MlpBlock<B: Backend> {
    fc1:  burn::nn::Linear<B>,    // (D → intermediate)
    act:  GeluActivation,         // tanh-approx GELU (HF default)
    drop: burn::nn::Dropout,
    fc2:  burn::nn::Linear<B>,    // (intermediate → D)
}
```

**Forward:** `fc2(drop(act(fc1(x))))`.

**RFC0002-008 [MUST]** — The activation **MUST** be `GeluVariant::TanhApprox` per HF ViT default. The exact recipe is:

```
gelu_tanh(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
```

Burn's `burn::tensor::activation::gelu` defaults to the erf-based form. We **MUST** use the tanh approximation via `tensor_ops::gelu_tanh_approx(x)`.

#### 4.2.8 Top-level `Vit`

```rust
#[derive(burn::module::Module, Debug)]
pub struct Vit<B: Backend> {
    embeddings: ViTEmbeddings<B>,
    blocks:     Vec<EncoderBlock<B>>,
    norm:       burn::nn::LayerNorm<B>,    // applied to all tokens at the end
    config:     VitConfig,
}

#[derive(Debug, Clone)]
pub struct ViTOutput<B: Backend> {
    /// All token outputs, post-final-LayerNorm. Shape `(B, P+1, D)`.
    pub last_hidden_state: Tensor<B, 3>,
}

impl<B: Backend> Vit<B> {
    pub fn forward(&self, pixels: Tensor<B, 4>) -> ViTOutput<B> { /* see algorithm */ }

    /// Convenience: extracts the CLS row of `last_hidden_state`. Shape `(B, D)`.
    pub fn cls_from(output: &ViTOutput<B>) -> Tensor<B, 2> {
        output.last_hidden_state.clone().slice([0..output.last_hidden_state.dims()[0], 0..1, 0..output.last_hidden_state.dims()[2]]).squeeze(1)
    }
}
```

**Forward algorithm:**

```
x = embeddings(pixels)               # (B, P+1, D)
for block in blocks:
    x = block(x)
x = norm(x)
return ViTOutput { last_hidden_state: x }
```

**RFC0002-009 [MUST]** — `Vit::forward` **MUST** apply the final `LayerNorm` to **all** tokens (not only CLS), matching HF semantics. The CLS row is extracted post-norm by `cls_from`.

### 4.3 Action `Embedder`

#### 4.3.1 Configuration

```rust
#[derive(burn::config::Config, Debug, Clone, Eq, PartialEq)]
pub struct EmbedderConfig {
    /// Per-step action dimensionality (2 for PushT, 6 for SO-100).
    pub input_dim: usize,
    /// Intermediate dim after the Conv1d-k1 (functionally a Linear).
    #[config(default = "16")]
    pub smoothed_dim: usize,
    /// Output embedding dim used by the predictor.
    #[config(default = "64")]
    pub emb_dim: usize,
    /// Inner MLP scale (FFN width = emb_dim * mlp_scale).
    #[config(default = "4")]
    pub mlp_scale: usize,
}
```

#### 4.3.2 Module

```rust
/// Action embedder. Maps `(B, T, A)` action tensors to `(B, T, emb_dim)` embeddings.
///
/// The Conv1d with kernel size 1 is mathematically a Linear; it is preserved here for
/// shape parity with the reference checkpoint's parameter layout.
#[derive(burn::module::Module, Debug)]
pub struct Embedder<B: Backend> {
    smoother: burn::nn::conv::Conv1d<B>,   // Conv1d(input_dim → smoothed_dim, kernel=1)
    fc1:      burn::nn::Linear<B>,         // Linear(smoothed_dim → emb_dim * mlp_scale)
    act:      SiLuActivation,
    fc2:      burn::nn::Linear<B>,         // Linear(emb_dim * mlp_scale → emb_dim)
}
```

**Forward algorithm:**

```
x: (B, T, A)
x = x.permute(0, 2, 1)            # (B, A, T)
x = smoother(x)                   # (B, smoothed_dim, T)
x = x.permute(0, 2, 1)            # (B, T, smoothed_dim)
x = fc1(x)                        # (B, T, emb_dim * mlp_scale)
x = silu(x)
x = fc2(x)                        # (B, T, emb_dim)
return x
```

**RFC0002-010 [MUST]** — The Conv1d-k1 is preserved in the graph even though it is mathematically equivalent to a Linear, so that the reference checkpoint's parameter dictionary loads without renaming. The parameter name **MUST** be `smoother.weight` with shape `(smoothed_dim, input_dim, 1)` and `smoother.bias` with shape `(smoothed_dim,)`.

**Init:**

- `smoother.weight`: truncated normal `std=0.02`.
- `fc1.weight`, `fc2.weight`: truncated normal `std=0.02`.
- All biases: zero.

### 4.4 MLP heads (`projector`, `pred_proj`)

#### 4.4.1 Configuration

```rust
#[derive(burn::config::Config, Debug, Clone, Eq, PartialEq)]
pub struct MlpConfig {
    pub input_dim:  usize,    // default 384
    pub hidden_dim: usize,    // default 1536
    pub output_dim: usize,    // default 384
    #[config(default = "NormVariant::BatchNorm1d")]
    pub norm:       NormVariant,
}

#[derive(burn::config::Config, Debug, Clone, Copy, Eq, PartialEq)]
pub enum NormVariant {
    BatchNorm1d,
    LayerNorm,
    None,
}
```

#### 4.4.2 Module

```rust
#[derive(burn::module::Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1:  burn::nn::Linear<B>,
    norm: NormBlock<B>,            // wraps BatchNorm1d or LayerNorm depending on `norm`
    act:  GeluActivation,          // erf-based GELU (HF default for projection heads)
    fc2:  burn::nn::Linear<B>,
}
```

**Forward algorithm (matches `module.py::MLP`):**

```
x: (..., input_dim)
x = fc1(x)                    # (..., hidden_dim)
# Norm operates on the feature dim. BatchNorm1d expects (B, C, *) so we flatten leading dims.
if norm == BatchNorm1d:
    shape = x.shape
    x = x.reshape(-1, hidden_dim)
    x = norm(x)
    x = x.reshape(shape)
else:
    x = norm(x)
x = gelu_erf(x)
x = fc2(x)                    # (..., output_dim)
return x
```

**RFC0002-011 [MUST]** — Default `norm = BatchNorm1d` (matches upstream). The BatchNorm1d operates on the *feature* dimension after flattening leading dims to a single batch dim. The reference parameter dict carries `norm.weight`, `norm.bias`, `norm.running_mean`, `norm.running_var`, `norm.num_batches_tracked`. Loading **MUST** map these exactly.

**RFC0002-012 [MUST]** — The activation in `Mlp` is **erf**-based GELU (Burn default) to match upstream, **not** the tanh-approx used inside the ViT encoder's MLP blocks. The difference is small but visible in parity tests; getting it right is required by TOL-002.

### 4.5 `AdaLNZero` helper

The AdaLN-zero conditioning is shared between blocks of the predictor.

```rust
/// AdaLN-zero modulation.
///
/// Given a conditioning vector `c: (B, T, E_a)`, produce six modulation tensors
/// `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`, each of
/// shape `(B, T, D)`. The MLP used to compute them is initialised so that the
/// output is the all-zero tensor at init time.
#[derive(burn::module::Module, Debug)]
pub struct AdaLNZero<B: Backend> {
    silu:   SiLuActivation,
    linear: burn::nn::Linear<B>,     // Linear(E_a → 6 * D)
    hidden_dim: usize,
}

impl<B: Backend> AdaLNZero<B> {
    pub fn forward(&self, c: Tensor<B, 3>) -> AdaLNZeroOutputs<B> {
        let mod_ = self.linear.forward(self.silu.forward(c));
        // mod_: (B, T, 6*D); split into six tensors of (B, T, D)
        let chunks = mod_.chunk(6, 2);
        AdaLNZeroOutputs {
            shift_msa: chunks[0].clone(),
            scale_msa: chunks[1].clone(),
            gate_msa:  chunks[2].clone(),
            shift_mlp: chunks[3].clone(),
            scale_mlp: chunks[4].clone(),
            gate_mlp:  chunks[5].clone(),
        }
    }
}
```

**RFC0002-013 [MUST]** — `linear.weight` and `linear.bias` **MUST** be initialised to zero (the "AdaLN-zero" part). Result: at init `shift_*`, `scale_*`, `gate_* = 0`, and the conditional block degenerates to an identity transform with `gate = 0`, i.e., the residual stream passes through unchanged. This is what makes the predictor stable at init.

**INV-006** in the master spec is exactly this invariant.

### 4.6 `ConditionalBlock`

```rust
/// Pre-norm transformer block with AdaLN-zero modulation conditioned on action embedding.
///
/// Equivalent semantics to DiT (Peebles & Xie, 2023) but with attention rather than
/// pure self-attention over patches. Causal mask applied along the temporal axis.
#[derive(burn::module::Module, Debug)]
pub struct ConditionalBlock<B: Backend> {
    norm1:  burn::nn::LayerNorm<B>,      // affine=false
    attn:   Attention<B>,                // shared definition with ViT, causal=true
    norm2:  burn::nn::LayerNorm<B>,      // affine=false
    mlp:    MlpBlock<B>,                 // hidden_dim = predictor.mlp_dim
    adaln:  AdaLNZero<B>,
}
```

**RFC0002-014 [MUST]** — `norm1` and `norm2` are `LayerNorm` with `elementwise_affine=False` (no learned gain/bias) — the affine transform is supplied by the AdaLN-zero shift/scale instead.

**Forward algorithm:**

```
x: (B, T, D)
c: (B, T, E_a)
mods = adaln(c)                                          # six (B, T, D) tensors

# attention branch (causal along T)
y = norm1(x)
y = y * (1 + mods.scale_msa) + mods.shift_msa             # broadcast
y = attn_causal(y)
x = x + mods.gate_msa * y

# mlp branch
y = norm2(x)
y = y * (1 + mods.scale_mlp) + mods.shift_mlp
y = mlp(y)
x = x + mods.gate_mlp * y
return x
```

**RFC0002-015 [MUST]** — `attn_causal` here is the same attention layer as ViT but with a causal mask added to the pre-softmax scores. Mask is upper-triangular `−∞` above the diagonal in F32. The mask **MUST** be built once and cached on the device.

### 4.7 `ArPredictor`

```rust
#[derive(burn::config::Config, Debug, Clone, Eq, PartialEq)]
pub struct PredictorConfig {
    /// Max sequence length supported by the learned positional embedding.
    #[config(default = "16")]
    pub num_frames: usize,
    /// Number of `ConditionalBlock`s.
    #[config(default = "6")]
    pub depth: usize,
    /// Number of attention heads per block.
    #[config(default = "6")]
    pub heads: usize,
    /// FFN inner dim in each `ConditionalBlock`.
    #[config(default = "1536")]
    pub mlp_dim: usize,
    /// Per-head dim. heads * dim_head must equal hidden_dim.
    #[config(default = "64")]
    pub dim_head: usize,
    /// Token dim (input/output of the predictor, equal to the encoder/projector dim).
    #[config(default = "384")]
    pub hidden_dim: usize,
    /// Action embedding dim (input to AdaLN-zero).
    #[config(default = "64")]
    pub action_emb_dim: usize,
    /// Sequence dropout (applied post pos-emb).
    #[config(default = "0.0")]
    pub dropout: f64,
    /// Embedding dropout (applied directly after pos add).
    #[config(default = "0.0")]
    pub emb_dropout: f64,
}

#[derive(burn::module::Module, Debug)]
pub struct ArPredictor<B: Backend> {
    pos_embed: burn::module::Param<Tensor<B, 3>>,   // (1, num_frames, hidden_dim)
    dropout:   burn::nn::Dropout,
    blocks:    Vec<ConditionalBlock<B>>,
    norm:      burn::nn::LayerNorm<B>,
    config:    PredictorConfig,
}
```

**Forward algorithm:**

```
z: (B, T_in, D)            # context embeddings, T_in <= num_frames
a: (B, T_in, E_a)          # action embeddings, broadcast to T_in if needed

T = z.shape[1]
pos = self.pos_embed[:, :T, :]                         # (1, T, D)
x   = z + pos                                          # broadcast over B
x   = dropout(x)                                       # train only
for block in blocks:
    x = block(x, c=a)
x = norm(x)
return x                                               # (B, T, D)
```

**RFC0002-016 [MUST]** — `pos_embed` is learned, shape `(1, num_frames, hidden_dim)`, init truncated normal `std=0.02`.

**RFC0002-017 [MUST]** — When `T < num_frames`, only the first `T` positions are used (slice operation, no recompute).

**RFC0002-018 [MUST]** — When `T > num_frames`, `forward` **MUST** return `Err(PredictorError::SequenceTooLong { got, max })`. There is no interpolation of the predictor's pos embedding in v1.

### 4.8 `Jepa` top-level wrapper

#### 4.8.1 Configuration

```rust
#[derive(burn::config::Config, Debug, Clone)]
pub struct JepaConfig {
    pub encoder:        VitConfig,
    pub action_encoder: EmbedderConfig,
    pub predictor:      PredictorConfig,
    pub projector:      MlpConfig,
    pub pred_proj:      MlpConfig,
    /// History size used by rollout (number of context steps fed to the predictor).
    #[config(default = "3")]
    pub history_size:   usize,
    /// Maximum rollout horizon supported (≤ predictor.num_frames).
    #[config(default = "8")]
    pub horizon:        usize,
}
```

#### 4.8.2 Module

```rust
#[derive(burn::module::Module, Debug)]
pub struct Jepa<B: Backend> {
    encoder:        Vit<B>,
    action_encoder: Embedder<B>,
    predictor:      ArPredictor<B>,
    projector:      Mlp<B>,
    pred_proj:      Mlp<B>,
    config:         JepaConfig,
}

impl<B: Backend> Jepa<B> {
    /// Encode a windowed image tensor to embeddings.
    ///
    /// # Shape
    /// - input  `pixels: (B, T, C, H, W)`
    /// - output `(B, T, D)` — CLS tokens, post-projector. F32 in mixed precision.
    pub fn encode(&self, pixels: Tensor<B, 5>) -> Tensor<B, 3> { /* §4.8.3 */ }

    /// Predict the next embedding(s) given context embeddings and an action sequence.
    ///
    /// Returns predictor output **post-pred_proj**. Shape `(B, T, D)`.
    pub fn predict(
        &self,
        context: Tensor<B, 3>,    // (B, T_ctx, D)
        actions: Tensor<B, 3>,    // (B, T_ctx, A)
    ) -> Tensor<B, 3> { /* §4.8.4 */ }

    /// Autoregressive rollout.
    ///
    /// # Shape
    /// - input  `start_embeds: (B, history_size, D)`
    /// - input  `actions:      (B, T_actions, A)` with `T_actions = horizon - history_size`
    /// - output `(B, horizon, D)` — concatenation of `start_embeds` and predicted tail
    pub fn rollout(
        &self,
        start_embeds: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> Tensor<B, 3> { /* §4.8.5 */ }

    /// Training-time criterion. Returns `(L_pred, L_sigreg)` — total combined in RFC 0003.
    pub fn criterion(
        &self,
        pixels:  Tensor<B, 5>,
        actions: Tensor<B, 3>,
        lambda_sigreg: f64,
    ) -> JepaLosses<B> { /* RFC 0003 */ }

    /// Planning cost: MSE between predicted final embedding and goal embedding.
    pub fn get_cost(
        &self,
        z_history: Tensor<B, 3>,
        actions:   Tensor<B, 3>,
        z_goal:    Tensor<B, 2>,
    ) -> Tensor<B, 1> { /* §4.8.6 */ }
}
```

#### 4.8.3 `encode`

```
pixels: (B, T, C, H, W)
B, T = pixels.shape[:2]
flat = pixels.reshape(B*T, C, H, W)
out  = encoder(flat).last_hidden_state          # (B*T, P+1, D)
cls  = out[:, 0, :]                             # (B*T, D)
proj = projector(cls)                           # (B*T, D)
z    = proj.reshape(B, T, D)
return z
```

**RFC0002-019 [MUST]** — `encode` applies `projector` to the CLS token. This is what `jepa.py::JEPA.encode` does and the parity dump captures.

#### 4.8.4 `predict`

```
z_ctx: (B, T_ctx, D)
a_ctx: (B, T_ctx, A)
a_emb = action_encoder(a_ctx)                  # (B, T_ctx, E_a)
out   = predictor(z_ctx, a_emb)                # (B, T_ctx, D)
out   = pred_proj(out)                         # (B, T_ctx, D)
return out
```

#### 4.8.5 `rollout`

```
start_embeds: (B, H, D)        where H = history_size
actions:      (B, K, A)        where K = horizon - H

z = start_embeds.clone()                            # (B, H, D)
for k in 0 .. K:
    # window is the last H embeddings; action at this step is actions[:, k:k+1]
    ctx_z = z[:, -H:, :]                            # (B, H, D)
    ctx_a = actions[:, k:k+1, :]                    # (B, 1, A) — duplicated below
    # The predictor consumes the full window of actions paired with embeddings.
    # We construct a parallel action tensor (B, H, A) by broadcasting the current step,
    # following `jepa.py::JEPA.rollout` semantics where the action vector is repeated.
    ctx_a_full = ctx_a.expand([B, H, A])            # (B, H, A)
    pred = predict(ctx_z, ctx_a_full)               # (B, H, D)
    z_next = pred[:, -1:, :]                        # (B, 1, D)
    z = concat([z, z_next], dim=1)                  # (B, H + k + 1, D)
return z                                             # (B, H + K, D) = (B, horizon, D)
```

**RFC0002-020 [MUST]** — The rollout broadcasts the current step's action across the window of history positions, matching upstream `jepa.py::JEPA.rollout`. This is **not** the same as passing the cumulative action history; the prediction is conditioned on the action that drives the *next* step.

**RFC0002-021 [MAY]** — A future optimization may pass the full causal action sequence in one shot to amortize compute. v1 implements the straight loop for parity simplicity.

#### 4.8.6 `get_cost`

```
z_history: (B, H, D)
actions:   (B, K, A)         where K = horizon - H
z_goal:    (B, D)

z_full = rollout(z_history, actions)                # (B, H+K, D)
z_final = z_full[:, -1, :]                          # (B, D)
diff    = z_final - z_goal                          # (B, D)
cost    = (diff * diff).mean(dim=-1)                # (B,) per-element MSE; one cost per candidate
return cost                                          # (B,)
```

**RFC0002-022 [MUST]** — `get_cost` returns one scalar **per batch element**, *not* one scalar averaged across the batch. This is required by CEM, which uses the per-candidate costs to rank.

### 4.9 Top-level export of training-time losses

The `criterion` method is the entry point for the training loop. Its body and the loss definitions are in [RFC 0003 §4](0003-sigreg-and-loss-functions.md). Here we only state the return type:

```rust
#[derive(Debug, Clone)]
pub struct JepaLosses<B: Backend> {
    pub pred:   Tensor<B, 1>,         // L_pred (scalar)
    pub sigreg: Tensor<B, 1>,         // L_sigreg (scalar, F32 inside)
    pub total:  Tensor<B, 1>,         // L_pred + lambda * L_sigreg
}
```

---

## 5. Initialization recipe summary

All parameter initialization defaults are listed in §4 alongside the module that owns them. To make audit trivial, the following table is the **complete** init manifest:

| Parameter | Shape | Init |
|-----------|-------|------|
| `vit.patch_embed.proj.weight` | `(D, C, P, P)` | trunc_normal `std=0.02`, `±2σ` |
| `vit.patch_embed.proj.bias` | `(D,)` | zeros |
| `vit.embeddings.cls_token` | `(1, 1, D)` | trunc_normal `std=0.02`, `±2σ` |
| `vit.embeddings.pos_embed` | `(1, P+1, D)` | trunc_normal `std=0.02`, `±2σ` |
| `vit.blocks[i].norm1.weight` | `(D,)` | ones |
| `vit.blocks[i].norm1.bias` | `(D,)` | zeros |
| `vit.blocks[i].attn.qkv.weight` | `(3D, D)` | trunc_normal `std=0.02`, `±2σ` |
| `vit.blocks[i].attn.qkv.bias` | `(3D,)` | zeros |
| `vit.blocks[i].attn.proj.weight` | `(D, D)` | trunc_normal `std=0.02`, `±2σ` |
| `vit.blocks[i].attn.proj.bias` | `(D,)` | zeros |
| `vit.blocks[i].norm2.{weight,bias}` | `(D,)` | ones / zeros |
| `vit.blocks[i].mlp.fc1.weight` | `(intermediate, D)` | trunc_normal `std=0.02`, `±2σ` |
| `vit.blocks[i].mlp.fc1.bias` | `(intermediate,)` | zeros |
| `vit.blocks[i].mlp.fc2.weight` | `(D, intermediate)` | trunc_normal `std=0.02`, `±2σ` |
| `vit.blocks[i].mlp.fc2.bias` | `(D,)` | zeros |
| `vit.norm.{weight,bias}` | `(D,)` | ones / zeros |
| `action_encoder.smoother.weight` | `(smoothed, A, 1)` | trunc_normal `std=0.02` |
| `action_encoder.smoother.bias` | `(smoothed,)` | zeros |
| `action_encoder.fc1.{weight,bias}` | `(emb·mlp_scale, smoothed)` / `(emb·mlp_scale,)` | trunc / zeros |
| `action_encoder.fc2.{weight,bias}` | `(emb, emb·mlp_scale)` / `(emb,)` | trunc / zeros |
| `predictor.pos_embed` | `(1, num_frames, D)` | trunc_normal `std=0.02` |
| `predictor.blocks[i].norm{1,2}.weight` | absent (affine=False) | — |
| `predictor.blocks[i].attn.{qkv,proj}.{weight,bias}` | as ViT attn | trunc / zeros |
| `predictor.blocks[i].mlp.fc{1,2}.{weight,bias}` | as ViT mlp | trunc / zeros |
| `predictor.blocks[i].adaln.linear.weight` | `(6D, E_a)` | **zeros** (INV-006) |
| `predictor.blocks[i].adaln.linear.bias` | `(6D,)` | **zeros** (INV-006) |
| `predictor.norm.{weight,bias}` | `(D,)` | ones / zeros |
| `projector.fc1.{weight,bias}` | `(hidden, D)` / `(hidden,)` | trunc / zeros |
| `projector.norm.{weight,bias,running_mean,running_var}` | BatchNorm1d state | ones / zeros / zeros / ones |
| `projector.fc2.{weight,bias}` | `(D, hidden)` / `(D,)` | trunc / zeros |
| `pred_proj.*` | as `projector` | as `projector` |

**RFC0002-023 [MUST]** — Truncated normal initialization uses the `rand_chacha::ChaCha20` stream seeded as in [RFC 0013 §4](0013-determinism-and-reproducibility.md) (sub-stream `rng:model_init`). The rejection sampling truncates at `±2σ` exactly; we do **not** use a Box–Muller approximation.

**RFC0002-024 [MUST]** — The init order **MUST** be deterministic: parameters are initialized in the order they are discovered by `Module::visit_params` (depth-first, declaration order). Re-ordering struct fields therefore changes the initial weights; this is acceptable because the order is locked here.

---

## 6. Public API surface (final)

```rust
// crates/lewm-core/src/lib.rs
//
// Re-exports: configs, modules, helpers.

pub use crate::config::*;
pub use crate::embedder::Embedder;
pub use crate::jepa::{Jepa, JepaLosses};
pub use crate::mlp::Mlp;
pub use crate::predictor::{ArPredictor, ConditionalBlock};
pub use crate::vit::{EncoderBlock, PatchEmbed, ViTEmbeddings, ViTOutput, Vit};

pub mod losses;          // RFC 0003
pub mod tensor_ops;      // bicubic interp, causal mask, gelu_tanh
pub mod init;            // trunc_normal etc.
pub mod export;          // safetensors export (RFC 0010)

// Top-level error type
pub use crate::errors::{LewmCoreError, ParityError, PredictorError};
```

The crate exposes **no** other public items. Internal helpers stay private. Adding a new public item is a Minor SemVer bump per [`specs/README.md`](../README.md) §2.6.

---

## 7. Operational considerations

`lewm-core` is library-only, no operational side. Telemetry is propagated via `tracing` spans (`forward`, `encode`, `predict`, `rollout`, `criterion`) but the actual subscriber is set up by callers ([RFC 0009](0009-observability-and-mlops.md)).

---

## 8. Performance considerations

- **Attention fast path:** the SDPA may be fused or unfused; the fused path is preferred and gated by a `cfg!(feature = "cuda")` runtime check in the attention forward. The fallback to the explicit path is automatic.
- **Pos-embed cache:** the interpolated position embedding is cached when the input size matches the previous call; cache invalidation is on shape change.
- **Causal mask cache:** built once on the device per sequence length; stored in a `parking_lot::RwLock<HashMap<usize, Tensor<B, 2>>>` field of `ConditionalBlock`. (`Module` derives skip the field via `#[module(skip)]`.)

The perf targets and benches are in [RFC 0014 §4](0014-performance-engineering.md).

---

## 9. Security considerations

Pure numerical code. No security surface beyond the supply-chain considerations covered by [RFC 0016](0016-security-and-supply-chain.md). Tensor inputs are **trusted** within the workspace; the loaders validate shape and dtype upstream.

---

## 10. Alternatives considered

- **A1 — Use Burn's `MultiHeadAttention` directly.** Rejected: we want byte-exact parity with HF, and Burn's MHA does not match HF's `nn.MultiheadAttention` parameter naming (Burn uses split q/k/v, HF uses combined `qkv` linear). Re-implementing is cheap and decouples us from Burn API churn.
- **A2 — Use rotary position embedding instead of learned absolute.** Rejected: parity with upstream requires learned absolute. Worth revisiting in a v2.
- **A3 — Use `nn::GELU` from Burn.** Rejected: Burn's default is erf-based; we need tanh-approx inside ViT and erf inside the projection MLPs. Locking each explicitly via `tensor_ops::gelu_{erf,tanh_approx}` is clearer than relying on Burn defaults that may change.
- **A4 — Drop AdaLN-zero in favour of plain AdaLN.** Rejected: empirically observed in upstream that AdaLN-zero is critical for stable JEPA training; without it the encoder collapses in the first 200 steps.

---

## 11. Acceptance criteria

- [ ] All types in §6 exist and compile with `cargo doc --no-deps` warning-free.
- [ ] Init manifest §5 matches the dump produced by `python/convert_reference.py --emit-init-fingerprint`.
- [ ] Forward parity tests in [RFC 0008 §6](0008-reference-parity-testing.md) pass (TST-0008-ENC-001, TST-0008-PRED-001 etc.).
- [ ] Shape tests pass for all module forwards across `(B ∈ {1,4,8})`, `(T ∈ {1,3,8,16})`, and `(H ∈ {192, 224})`.
- [ ] `tensor_ops::interpolate_pos_embed` is exercised by a dedicated test at `image_size=192`.
- [ ] AdaLN-zero invariant INV-006 verified: at init, `forward(x, c) == forward(x, zeros_like(c))` for every block.
- [ ] Rollout test: with all parameters set to identity-like values (zero blocks + identity proj), `rollout(start, actions)` returns the start padded with repeats of the last context.

---

## 12. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Burn API change between minor versions breaks parameter naming | M | H | Burn pinned at `=0.20.1`; ADR required for bump |
| R-2 | Trunc-normal init differs in tail behaviour vs PyTorch's `trunc_normal_` | L | M | TST-0008 covers tail; spec uses rejection sampling identical to PyTorch's |
| R-3 | Fused SDPA on CUDA produces different numerics than the explicit kernel | M | M | Parity tests run on the **explicit** path; fused path is enabled only after step-100 smoke confirms equivalence |
| R-4 | LayerNorm `eps` mismatch | L | M | `eps = 1e-12` pinned in `VitConfig` |

---

## 13. Open questions

None at this revision.

---

## 14. Appendix — parameter count audit

For the LeWM defaults the parameter count is:

```
ViT-Small encoder:
  patch_embed: 16*16*3*384 + 384                       =      294,912 +       384 =       295,296
  cls_token:                                                                          384
  pos_embed:  (196+1)*384                                                          75,648
  per block:
    norm1: 2*384                                          =                            768
    attn.qkv: 384*3*384 + 3*384                           =     442,368 +     1,152 =       443,520
    attn.proj: 384*384 + 384                              =     147,456 +       384 =       147,840
    norm2: 2*384                                          =                            768
    mlp.fc1: 384*1536 + 1536                              =     589,824 +     1,536 =       591,360
    mlp.fc2: 1536*384 + 384                               =     589,824 +       384 =       590,208
    block subtotal                                                                  1,774,464
  12 blocks                                                                        21,293,568
  final norm: 2*384                                                                       768

Hold on — that's already ~21.6M for the encoder alone, above the 15M target. Let's re-audit.
```

Re-audit: the target "15M total" in PRD §5.2 includes the *encoder plus predictor plus heads*. A more careful counting shows the encoder is ~21M parameters at ViT-Small specs. The PRD's "14.8M to 15.2M" cited the **paper's** number which appears to be a ViT-Tiny variant (`hidden=192, depth=12, heads=3`) — **the PRD must be reconciled here**.

**Open question OQ-2002-1:** Verify whether the published LeWM PushT checkpoint uses ViT-Tiny (`hidden=192, depth=12, heads=3, ~5.5M`) or ViT-Small (`hidden=384, depth=12, heads=6, ~22M`). The PRD lists `hidden=384, depth=12, heads=6` (ViT-Small), but the parameter target `15M` corresponds neither directly. **Decision pending Phase 1 weight inspection**: `python/convert_reference.py` will report the exact parameter count and dimensions, and the result is recorded in `reports/parity.md` and reflected in the next revision of this RFC.

**Resolution path:**

1. Phase 0 task: pull `quentinll/lewm-pusht` config and inspect `hidden_size`.
2. If `hidden=384`, the PRD's "15M" is a paper-summary approximation and the configs §5.2 are correct.
3. If `hidden=192`, update `VitConfig` defaults and the param-count audit here.

The parity tests in [RFC 0008](0008-reference-parity-testing.md) catch this discrepancy automatically: they load the actual checkpoint and the shape mismatch fails fast.

---

## 15. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0002.*

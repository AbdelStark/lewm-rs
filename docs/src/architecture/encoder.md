# The ViT-Tiny encoder

> **Motivation.** The encoder is the *information bottleneck* of the
> entire system: every other module operates on its output. This page
> documents the encoder at the level of detail required for byte-exact
> reproduction of the locked PushT reference checkpoint.
>
> **Position.** First module page in [Part II](./overview.md).
>
> **What you should leave with.** The exact dataflow through the
> encoder, all defaults, all initialization rules, and a pointer to the
> source for every component.

## 1. Configuration

`VitConfig` in `crates/lewm-core/src/config.rs`:

```rust,ignore
#[derive(burn::config::Config, Debug, Clone, Eq, PartialEq)]
pub struct VitConfig {
    pub image_size: usize,           // default 224
    pub patch_size: usize,           // default 14
    pub num_channels: usize,         // default 3
    pub hidden_size: usize,          // D, default 192
    pub num_hidden_layers: usize,    // default 12
    pub num_attention_heads: usize,  // default 3  (head dim = 192/3 = 64)
    pub intermediate_size: usize,    // default 768 (= 4 · D)
    pub hidden_act: GeluVariant,     // default Erf (NOT TanhApprox)
    pub attention_probs_dropout_prob: f64, // default 0.0
    pub hidden_dropout_prob: f64,    // default 0.0
    pub layer_norm_eps: f64,         // default 1.0e-12
    pub use_cls_token: bool,         // default true
    pub interpolate_pos_encoding: bool, // default false
}
```

Two defaults to highlight:

- **`layer_norm_eps = 1e-12`.** This is *not* the PyTorch / HF default
  (`1e-5`). Upstream LeWM was trained with `1e-12`. Using anything else
  drifts the encoder output by $\sim 10^{-3}$ — over the parity floor.
- **`hidden_act = GeluVariant::Erf`.** Modern PyTorch defaults to the
  tanh approximation; LeWM uses the exact erf form. See
  [ViT concepts §4](../concepts/vit.md).

## 2. Patch embedding

`PatchEmbed` in `crates/lewm-core/src/vit.rs`:

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    proj: burn::nn::conv::Conv2d<B>,
    num_patches: usize,
    patch_size: usize,
}
```

- `proj` is a `Conv2d` with kernel `(patch_size, patch_size)`, stride
  `patch_size`, no padding, **bias enabled**, input channels
  `num_channels = 3`, output channels `hidden_size = 192`.
- `num_patches = (image_size / patch_size)² = (224/14)² = 256`.

Forward algorithm:

```text
input  pixels: (B, 3, 224, 224)
       x = conv2d_proj(pixels)             # (B, 192, 16, 16)
       x = flatten(x, start_dim=2)         # (B, 192, 256)
       x = transpose(x, 1, 2)              # (B, 256, 192)
output:                                    # (B, num_patches, D)
```

**RFC0002-001 [MUST]** — `proj` weight is initialised by *truncated
normal* with $\mu = 0$, $\sigma = 0.02$, clipped at $\pm 2\sigma$.
The bias is initialised to zero. This matches HF's `_init_weights`
for `nn.Conv2d` at `init_range = 0.02`.

## 3. CLS token and position embeddings

`ViTEmbeddings` in `crates/lewm-core/src/vit.rs`:

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct ViTEmbeddings<B: Backend> {
    patch_embed: PatchEmbed<B>,
    cls_token: burn::module::Param<Tensor<B, 3>>,        // (1, 1, D)
    position_embeddings: burn::module::Param<Tensor<B, 3>>, // (1, num_patches + 1, D)
}
```

Forward algorithm:

```text
patch_tokens = patch_embed(pixels)         # (B, 256, 192)
cls          = expand(cls_token, (B, 1, D))
x            = concat([cls, patch_tokens], dim=1)  # (B, 257, 192)
x            = x + position_embeddings              # broadcast: (1, 257, 192) → (B, 257, 192)
```

Initialization:

- `cls_token`: truncated normal, $\sigma = 0.02$ (RFC0002-002).
- `position_embeddings`: truncated normal, $\sigma = 0.02$ (RFC0002-003).
- LeWM uses *learned absolute* position embeddings (not rotary, not
  sinusoidal). The same single tensor is used at every block; the
  transformer attention does not inject any further positional signal.

## 4. The transformer block

`EncoderBlock` in `crates/lewm-core/src/vit.rs`:

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    norm1: burn::nn::LayerNorm<B>,        // eps = 1e-12
    attention: Attention<B>,
    norm2: burn::nn::LayerNorm<B>,        // eps = 1e-12
    mlp: MlpBlock<B>,
}
```

Forward (pre-norm Transformer):

```text
y = self.attention.forward(self.norm1.forward(x))
x = x + y
y = self.mlp.forward(self.norm2.forward(x))
x = x + y
return x
```

Note: this is **pre-norm**, not post-norm. The LayerNorm precedes the
sub-layer.

### 4.1 Attention sub-layer

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct Attention<B: Backend> {
    qkv: burn::nn::Linear<B>,    // Linear(D, 3D, bias=True)
    proj: burn::nn::Linear<B>,   // Linear(D, D,  bias=True)
    num_heads: usize,            // = 3 for ViT-Tiny
    head_dim: usize,             // = D / num_heads = 64
}
```

Forward:

```text
qkv = self.qkv.forward(x)           # (B, T, 3D)
q, k, v = split(qkv, 3, dim=-1)     # each (B, T, D)
q = reshape(q, (B, T, num_heads, head_dim)).transpose(1, 2)  # (B, H_a, T, head_dim)
k = reshape(k, ...).transpose(1, 2)
v = reshape(v, ...).transpose(1, 2)

scores = q @ k.transpose(-1, -2) / sqrt(head_dim)     # (B, H_a, T, T)
probs  = softmax(scores, dim=-1)                       # (B, H_a, T, T)
out    = probs @ v                                      # (B, H_a, T, head_dim)
out    = out.transpose(1, 2).reshape(B, T, D)
return self.proj.forward(out)                          # (B, T, D)
```

- `qkv` linear is one $D \times 3D$ matrix that produces all three
  Q/K/V projections in a single matmul (HF convention).
- `qkv` and `proj` are both bias-enabled, matching the HF ViT default.

**RFC0002-006/007 [MUST]** — Both `qkv` and `proj` are initialised by
truncated normal with $\sigma = 0.02$, biases to zero.

### 4.2 MLP sub-layer

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct MlpBlock<B: Backend> {
    fc1: burn::nn::Linear<B>,         // Linear(D, intermediate_size, bias=True)
    fc2: burn::nn::Linear<B>,         // Linear(intermediate_size, D, bias=True)
    act: GeluVariant,                  // Erf by default
}
```

Forward:

```text
return self.fc2.forward( gelu( self.fc1.forward(x), variant=self.act ) )
```

`intermediate_size = 4D = 768`. The MLP expansion ratio is 4, matching
DeiT / HF ViT.

## 5. The transformer stack

`Vit` composes 12 `EncoderBlock`s and a final LayerNorm:

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct Vit<B: Backend> {
    embeddings: ViTEmbeddings<B>,
    blocks: Vec<EncoderBlock<B>>,            // length = num_hidden_layers = 12
    final_norm: burn::nn::LayerNorm<B>,      // eps = 1e-12
    config: VitConfig,
}
```

Forward:

```text
x = self.embeddings.forward(pixels)              # (B, 257, 192)
for block in &self.blocks:
    x = block.forward(x)
x = self.final_norm.forward(x)                    # (B, 257, 192)
return x
```

The final LayerNorm is applied *after* the 12-block stack but *before*
any pooling or CLS read-off. This matches the HF ViT convention.

## 6. The CLS read-off

For the LeWM downstream loss, we take only the CLS token:

```text
out = jepa.vit.forward(pixels)        # (B, 257, 192)
cls = out[:, 0, :]                     # (B, 192)
```

This is wrapped in `Jepa::encode` (see [Jepa wrapper](./jepa-wrapper.md))
which also handles the temporal axis: for a windowed input
`(B, T, 3, 224, 224)`, encode iterates over `T` and stacks the CLS
outputs to produce `(B, T, 192)`.

## 7. Parameter count and breakdown

The 12-block ViT-Tiny accounts for ~5.5 M parameters of the total
18 M:

| Sub-component | Shape | Count |
|---------------|------:|------:|
| `patch_embed.proj.weight` | $192 \times 3 \times 14 \times 14$ | 112 896 |
| `patch_embed.proj.bias`   | $192$ | 192 |
| `cls_token`               | $1 \times 1 \times 192$ | 192 |
| `position_embeddings`     | $1 \times 257 \times 192$ | 49 344 |
| Per block × 12 | (see below) | ~430 000 |
| `final_norm.weight`       | $192$ | 192 |
| `final_norm.bias`         | $192$ | 192 |
| **Encoder total** | | **~5.5 M** |

Per block (12 of these):

| Tensor | Shape | Count |
|--------|------:|------:|
| `norm1.weight, .bias` | 192, 192 | 384 |
| `attention.qkv.weight` | $192 \times 576$ | 110 592 |
| `attention.qkv.bias`   | $576$ | 576 |
| `attention.proj.weight` | $192 \times 192$ | 36 864 |
| `attention.proj.bias`   | $192$ | 192 |
| `norm2.weight, .bias`  | 192, 192 | 384 |
| `mlp.fc1.weight`       | $192 \times 768$ | 147 456 |
| `mlp.fc1.bias`         | $768$ | 768 |
| `mlp.fc2.weight`       | $768 \times 192$ | 147 456 |
| `mlp.fc2.bias`         | $192$ | 192 |
| **Per-block total** | | **444 864** |

## 8. Parity tests

The encoder is the most-tested module in the system. Two parity tests
in `crates/lewm-core/tests/`:

1. `parity_encoder_cls`: compares `jepa.encode(fixture_pixels)[:, 0,
   :]` to the upstream PyTorch reference's CLS output. Tolerance:
   $L_\infty < 10^{-4}$.
2. `parity_encoder_all`: compares the full
   `jepa.vit.forward(fixture_pixels)` to the reference's `(B, 257,
   192)` output. Tolerance: $L_\infty < 10^{-4}$.

Both currently <span class="lewm-badge lewm-badge--done">PASS</span>
against `quentinll/lewm-pusht@22b330c`. The reference dumps live at
[`AbdelStark/lewm-rs-parity-dumps`](https://huggingface.co/datasets/AbdelStark/lewm-rs-parity-dumps).

## 9. Source pointers

| Topic | Source |
|-------|--------|
| `VitConfig` | `crates/lewm-core/src/config.rs` |
| `PatchEmbed`, `ViTEmbeddings`, `Attention`, `MlpBlock`, `EncoderBlock`, `Vit` | `crates/lewm-core/src/vit.rs` |
| Truncated-normal init helper | `crates/lewm-core/src/init.rs` |
| GELU variants | `crates/lewm-core/src/tensor_ops.rs` |
| Parity tests | `crates/lewm-core/tests/parity_encoder_*.rs` |
| Reference dump generation | `python/convert_reference.py dump --component encoder` |

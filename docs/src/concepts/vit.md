# Vision Transformers in latent prediction

> **Motivation.** The encoder in LeWM is a ViT-Tiny — not "a ViT", but
> *specifically* a Hugging Face–compatible ViT-Tiny with hand-picked
> implementation details. This page is for readers who want to understand
> why this encoder, why these defaults, and which knobs matter for parity.
>
> **Position.** Conceptual prelude to [The ViT-Tiny encoder](../architecture/encoder.md),
> which gives the byte-level shape contracts and source-level details.
>
> **What you should leave with.** A clear mental model of the encoder's
> dataflow, an appreciation of the design choices (patch size, head count,
> exact-erf GELU, $\varepsilon$), and a sense of why this is the right
> *size* of model for the task.

## 1. Why a Vision Transformer at all

LeWM consumes 224 × 224 RGB images. Three obvious encoder choices were
available to the upstream authors:

| Family | Pros | Cons |
|--------|------|------|
| **CNN backbone** (ResNet, ConvNeXt) | Inductive bias for locality and translation equivariance; strong on small datasets. | Architecturally heavy; bag of normalization choices; less natural to feed into a transformer-based predictor. |
| **ViT** (Dosovitskiy et al., 2021) | Uniform token interface to the downstream predictor; clean attention semantics; well-understood in the JEPA literature. | Weaker inductive bias; needs more data than a CNN at the same parameter count. |
| **Hybrid** (CvT, CoAtNet, etc.) | A middle ground. | Many sub-variants; less consensus on defaults. |

LeWM picks **ViT** for one practical reason: the predictor downstream is
also a transformer, and uniform token interfaces compose without friction.
The encoder's output, $(B, T, D = 192)$, is exactly the input shape the
predictor wants.

## 2. The ViT-Tiny size

Following Touvron et al.'s DeiT-Tiny convention, "ViT-Tiny" denotes:

| Knob | Value |
|------|-------|
| Hidden dim $D$ | 192 |
| Depth | 12 |
| Attention heads | 3 (head dim 64) |
| MLP inner dim | 768 (= $4D$) |
| Patch size | 16 (DeiT default); LeWM uses **14** |

LeWM's choice of patch size 14 over the DeiT default of 16 is driven by
the input resolution: $224 / 14 = 16$ patches per side, $16^2 = 256$
patches per image (plus one CLS = 257 tokens). With patch size 16 we
would have $14 \times 14 = 196$ patches, which is what the original
DeiT-Tiny uses, but LeWM's upstream code follows the HF
`google/vit-base-patch16-224` family's preference for patch 14 + image
224. This is purely a checkpoint-compatibility choice; it has no
algorithmic significance.

At 18 M parameters total (whole model, including predictor), the system
sits at the small end of modern visual encoders. ViT-Base would be 86 M;
ViT-Large would be 304 M. The small size is deliberate: the project
wants to demonstrate **CPU planning on a laptop**, which sets a hard
ceiling on encoder cost.

## 3. The forward pass, in five steps

1. **Patch embedding.** A 2-D convolution with kernel $14 \times 14$,
   stride 14, no padding, maps `(B, 3, 224, 224)` to `(B, 192, 16, 16)`.
   This is flattened and transposed to `(B, 256, 192)`. Each row is a
   192-D embedding of a patch.

2. **CLS token + position embeddings.** A learned vector `cls_token` of
   shape `(1, 1, 192)` is prepended, giving `(B, 257, 192)`. A learned
   `position_embeddings` tensor of shape `(1, 257, 192)` is added
   element-wise. (LeWM does not use the rotary or sinusoidal embeddings
   used by some ViT variants; the HF default is *learned absolute*.)

3. **Transformer stack.** Twelve identical encoder blocks process the
   tokens. Each block is

```text
   x ← x + Attention( LayerNorm(x) )
   x ← x + MLP(       LayerNorm(x) )
   ```

   This is the *pre-norm* arrangement (LayerNorm before sub-layer), not
   the post-norm of the original transformer. Pre-norm is the modern
   default; it eliminates the warmup-LR pathology of post-norm transformers.

4. **Final LayerNorm.** A single `LayerNorm` is applied to the output of
   the last block, matching the HF ViT convention.

5. **CLS read-off.** For LeWM, the "embedding of the image" is the CLS
   token at the output of the final LayerNorm: `output[:, 0, :]` of shape
   `(B, 192)`. The other 256 patch tokens are computed but discarded for
   the downstream prediction loss. (They are still useful for parity
   tests: the full 257-token output is checked component-by-component.)

## 4. The four knobs that decide parity

Most ViT implementations are nearly identical. The places where they
quietly differ — and the places where lewm-rs had to be careful to match
upstream LeWM — are:

| Knob | Value used in `lewm-rs` | HF / PyTorch default | Source |
|------|--------------------------|----------------------|--------|
| LayerNorm $\varepsilon$ | $10^{-12}$ | $10^{-5}$ | [`VitConfig::layer_norm_eps`] |
| GELU variant | **exact-erf** | tanh-approx (modern PyTorch default for ViT) | [`GeluVariant::Erf`] |
| QKV bias | enabled | enabled (matches HF) | – |
| Attention probs dropout | 0.0 | 0.0 | – |

**$\varepsilon = 10^{-12}$.** A LayerNorm with default $\varepsilon = 10^{-5}$
on a tensor whose variance is on the order of $10^{-4}$ — which can
happen in deep transformer activations — clips against the $\varepsilon$
floor and produces visibly different gradients downstream. Upstream LeWM
was trained with $\varepsilon = 10^{-12}$, so we use the same. This is
not a tunable knob; changing it breaks parity.

**Exact-erf GELU.** Modern PyTorch defaults to the *tanh approximation*
of GELU for speed:

$$
\text{GELU}_{\text{tanh}}(x) \;=\; 0.5\, x\,\Bigl(1 + \tanh\!\bigl(\sqrt{2/\pi}\,(x + 0.044715\, x^3)\bigr)\Bigr).
$$

The exact form is

$$
\text{GELU}(x) \;=\; x \,\Phi(x) \;=\; \frac{1}{2}\,x\,\Bigl(1 + \mathrm{erf}\bigl(x / \sqrt{2}\bigr)\Bigr),
$$

where $\Phi$ is the standard normal CDF. The two differ by ~$10^{-4}$ in
the activation values, which compounds over 12 transformer blocks to
produce $\sim 10^{-3}$ output drift — well above the $10^{-4}$ parity
tolerance. lewm-rs uses the exact form throughout.

These two choices, together, account for the bulk of the implementation
gotchas in [Parity gotchas](../parity/gotchas.md).

## 5. Why the CLS token, not pooled patches?

When the "embedding" of an image is needed, three common choices are
available:

1. The CLS token at the output of the final block.
2. The mean of all 256 patch tokens.
3. A learned attention pooler over the patch tokens (perceiver-style).

LeWM uses option 1, matching upstream. The CLS token has the advantage
that the entire transformer is free to write into it through the
attention mechanism, so the 192 dimensions of `output[:, 0, :]` can in
principle encode any function of the input. Mean-pooling, by contrast,
forces a "symmetric" combination of patch features that may not be
optimal for action-conditioned prediction. The third option (learned
pooler) is a trainable upgrade but adds parameters and a small amount
of complexity; LeWM does not use it.

## 6. Attention details

For completeness, the attention sublayer at each block is:

```text
Linear(D → 3D, bias=True)              # qkv projection
  → split into q, k, v of shape (B, H_a, T, D/H_a)
softmax( q · kᵀ / sqrt(D/H_a) )         # scaled dot-product
  · v
  → reshape to (B, T, D)
Linear(D → D, bias=True)               # output projection
```

with $H_a = 3$, head dim $= D/H_a = 64$. There is no dropout in either
the attention probabilities or the output projection (LeWM uses
`attention_probs_dropout_prob = 0.0` and `hidden_dropout_prob = 0.0`).

The Burn implementation uses `tensor.matmul` directly rather than a fused
SDPA kernel, because Burn's SDPA support was not stable at v0.20.1 when
we needed parity. The performance cost on a ViT-Tiny is negligible.

## 7. Position embeddings and the "interpolate" option

The HF ViT class has an option `interpolate_pos_encoding` that bilinearly
resizes the learned 1-D position embedding sequence when the input image
size differs from the training size. lewm-rs preserves this code path
for forward compatibility but **defaults to `false`** because every LeWM
input is exactly $224 \times 224$. Setting `interpolate_pos_encoding =
true` is a no-op at the LeWM defaults but exercises the bilinear
interpolation code, which is useful for parity testing the path against
HF.

## 8. Putting it together

After all of this, the encoder is the function

$$
f_\theta : \;(B, 3, 224, 224) \to (B, 257, 192).
$$

For the prediction loss, we take only the CLS row: $(B, 192)$. The
remaining $256$ patch tokens are computed but unused in LeWM v1. The
projector then rewrites $(B, 192)$ into a same-dimensional $192$-D
"loss space" via a non-linear $192 \to 2048 \to 192$ MLP with
BatchNorm1d, where both the prediction loss and the SIGReg sketch
operate (see [Projector and pred-proj MLPs](../architecture/projector.md)).

This compact, $192$-D image embedding is what makes everything else
cheap. The predictor operates on $(B, T, 192)$ — i.e. one $192$-D
vector per historical frame, $T = 3$ frames per window — so a CEM
planner can run $1024$ candidates per iteration at well under a second
on CPU.

## 9. Bibliography

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
  et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image
  Recognition at Scale*. ICLR.
- Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jegou, H.
  (2021). *Training data-efficient image transformers & distillation
  through attention*. ICML — introduces DeiT-Tiny.
- Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H.,
  Lan, Y., Wang, L., Liu, T.-Y. (2020). *On Layer Normalization in the
  Transformer Architecture*. ICML — pre-norm motivation.
- Hendrycks, D., Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*.
  arXiv:1606.08415.

[`VitConfig::layer_norm_eps`]: https://github.com/AbdelStark/lewm-rs/blob/main/crates/lewm-core/src/config.rs
[`GeluVariant::Erf`]: https://github.com/AbdelStark/lewm-rs/blob/main/crates/lewm-core/src/config.rs

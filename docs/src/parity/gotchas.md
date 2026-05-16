# Implementation gotchas

> **Motivation.** The four bugs that broke parity until they were
> fixed, written up so the next person porting JEPA to a new framework
> doesn't repeat them.
>
> **Position.** Sub-page of [Part VI](./why-parity.md).
>
> **What you should leave with.** Four specific, recurring traps and
> how to avoid them.

## 1. LayerNorm $\varepsilon = 10^{-12}$, not $10^{-5}$

**Default PyTorch.** `torch.nn.LayerNorm(D)` uses `eps=1e-5`.

**Default HF ViT.** `transformers.ViTConfig.layer_norm_eps = 1e-12`.

**Upstream LeWM.** Trained with `eps=1e-12` (inherited from HF ViT).

If you build a Burn `LayerNorm` with the default $10^{-5}$ and run
it against the upstream LeWM checkpoint, you get drift on the order of
$10^{-3}$ in the encoder output. The drift compounds across 12 blocks
and the parity test fails by a wide margin.

**Fix.** Every LayerNorm in `lewm-core` explicitly sets `eps = 1e-12`
in its config:

```rust,ignore
pub layer_norm_eps: f64,        // default 1.0e-12, not 1.0e-5
```

This applies to the ViT block LayerNorms, the ViT final LayerNorm, the
predictor's `norm1` / `norm2` / `final_norm`, and any other
LayerNorm-flavoured op that touches an upstream-weights tensor.

## 2. Exact-erf GELU, not the tanh approximation

**Default PyTorch.** `F.gelu(x)` uses the exact erf form *by default*.

**Default HF ViT.** Configurable via `hidden_act = "gelu"` (exact) or
`hidden_act = "gelu_new"` (tanh approximation).

**Default many modern transformers.** Tanh approximation, for speed.

**Upstream LeWM.** Uses the exact erf form.

The two formulae differ by $\sim 10^{-4}$ pointwise; over 12 blocks the
drift accumulates to $\sim 10^{-3}$, breaking parity.

**Fix.** The `GeluVariant` enum in `crates/lewm-core/src/config.rs`
defaults to `Erf`, not `TanhApprox`. The tensor op in
`tensor_ops.rs::gelu(x, GeluVariant::Erf)` uses

$$
\text{GELU}(x) = \frac{x}{2}\,\bigl(1 + \mathrm{erf}(x / \sqrt{2})\bigr).
$$

Switching the variant to `TanhApprox` would use the fast formula. The
exact form is what matches upstream and is what lewm-rs uses.

## 3. Causal mask diagonal: `triu(1)`, not `triu(0)`

The causal mask should be **upper-triangular, strictly above the
diagonal**. In PyTorch:

```python
mask = torch.ones(T, T).triu(1).bool()      # correct: diagonal excluded
# mask:
#   [[F, T, T],
#    [F, F, T],
#    [F, F, F]]

# WRONG variant:
mask = torch.ones(T, T).triu(0).bool()      # diagonal included
#   [[T, T, T],
#    [F, T, T],
#    [F, F, T]]
```

The `triu(1)` form lets position $t$ attend to positions $0, \dots, t$
(including itself). The `triu(0)` form forbids position $t$ from
attending to itself — which produces an attention output that is
identically zero on the diagonal and degrades the model.

This is an off-by-one that is invisible at the loss-curve level
(it sometimes converges to a different floor) but **obvious in the
parity test**: the predictor output drifts by $O(1)$ on the
diagonal.

**Fix.** The Burn-side mask is built with `triu(1)`:

```rust,ignore
let mask: Tensor<B, 2> = Tensor::<B, 2, Int>::ones([T, T], device)
    .triu(1)                                       // <-- diagonal=1, exclusive
    .bool();
```

The choice is verified by the `parity_predictor` test.

## 4. SIGReg in F32, not BF16

SIGReg's high-frequency terms ($\cos(3 \cdot \mathbf p^\top \mathbf
z)$) are near stationary points of $\cos$ when $\mathbf p^\top \mathbf
z \approx 1$. BF16's 7-bit mantissa cannot resolve perturbations on
the order of $\sin(\delta \cdot 3) \approx 3\delta$ for $\delta$
small. The result is that the SIGReg gradient drifts under BF16,
typically *down-weighting* the regularizer relative to F32, which
allows partial collapse.

**Fix.** [INV-005] pins SIGReg's compute to F32. The Burn-side
implementation casts inputs to F32 at the call boundary and casts the
scalar back to BF16 at the output. The encoder LayerNorm and AdaLN
modulation linear are also F32 (separate F32 islands; see
[Mixed precision](../training/mixed-precision.md)).

## 5. Less catastrophic but still worth knowing

A few smaller details that didn't break parity but did slow down
development:

- **Conv2d patch embedder has a bias.** The HF ViT patch embedder is
  `Conv2d(3, 192, k=14, stride=14, bias=True)`. Some Burn examples
  default to `bias=False`. The reference loader expects the bias
  tensor, so a `bias=False` model fails the safetensors-load step.
- **Predictor `norm1` / `norm2` have *no* learnable affine.** In the
  HF and standard Burn LayerNorm, the affine is on by default. For
  the predictor's adaptive-LN setup, the affine must be disabled —
  the affine role is taken by `ada_ln_modulation`'s output.
- **Action smoother is a kernel-1 Conv1d on packed actions.**
  `Conv1d(10, 10, k=1, stride=1, padding=0)`. The frameskip-equivalent
  pooling is done upstream in the data plane (concatenation of
  `frameskip` raw actions into one $A_p = 10$ vector); the encoder
  Conv1d only does a per-timestep linear lift. Treating it as a
  temporal smoother (e.g. swapping in `k = frameskip`) breaks parity
  and the predictor's expected `T` alignment.

## 6. The take-away

These four (plus three minor) bugs together account for the bulk of
the parity-test churn during early lewm-rs development. They are now
captured in the test suite, the configs, and this page. If you are
porting JEPA-style models to a new framework, *check these four
specifically* before debugging anything else.

[INV-005]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0003-sigreg-and-loss-functions.md

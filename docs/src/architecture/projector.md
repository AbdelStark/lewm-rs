# Projector and pred-proj MLPs

> **Motivation.** Two small MLPs — `projector` and `pred_proj` —
> bracket the prediction loss and feed the SIGReg regulariser. They
> are not glamorous, but they are exactly where the prediction loss is
> computed and where the random-projection sketch of SIGReg is applied.
> This page documents both.
>
> **Position.** Fourth module page in [Part II](./overview.md).
>
> **What you should leave with.** The shape contract of each MLP — input
> and output both equal to $D = 192$, with a $2048$-wide hidden layer —
> the role of the BatchNorm1d between `fc1` and `fc2`, and which loss
> is computed in which space.

## 1. The two MLPs

Both are instances of the same `Mlp` struct in
`crates/lewm-core/src/mlp.rs`:

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1:  burn::nn::Linear<B>,        // Linear(input_dim  → hidden_dim)
    norm: NormBlock<B>,               // BatchNorm1d on hidden_dim (default)
    fc2:  burn::nn::Linear<B>,        // Linear(hidden_dim → output_dim)
    act:  GeluVariant,                // exact-erf GELU
}
```

with `MlpConfig`:

| Field | `projector` | `pred_proj` |
|-------|------------:|------------:|
| `input_dim`  | 192 | 192 |
| `hidden_dim` | 2048 | 2048 |
| `output_dim` | 192 | 192 |
| `norm`  | `BatchNorm1d` | `BatchNorm1d` |
| `act`   | exact-erf GELU | exact-erf GELU |

Forward (defined in [RFC 0002 §4.4](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0002-core-model-architecture.md#44-mlp-heads-projector-pred_proj)):

```text
x   = self.fc1.forward(x)               # (..., 2048)
x   = self.norm.forward(x)              # BatchNorm1d on the 2048 feature axis
x   = gelu(x)                           # exact-erf GELU
out = self.fc2.forward(x)               # (..., 192)
```

The BatchNorm1d normalises the $2048$-D hidden activation across the
batch (and, for these MLPs, the time axis after flattening). At
inference, the layer uses its running statistics rather than the batch
statistics; both the affine parameters and the running buffers are
mapped exactly from the upstream PyTorch reference (see
`python/param_name_map.py::_mlp_rules`).

## 2. Where each MLP sits in the dataflow

```text
                   encoder output           predictor output
                   (B, T, 192)             (B, T, 192)
                       │                        │
                       ▼                        ▼
                ┌──────────────┐         ┌──────────────┐
                │  projector   │         │  pred_proj   │
                │  192 → 2048  │         │  192 → 2048  │
                │  BN1d + GELU │         │  BN1d + GELU │
                │  2048 → 192  │         │  2048 → 192  │
                └──────┬───────┘         └──────┬───────┘
                       │                        │
                       │ z̃ (target arm,         │ ẑ_next (source arm,
                       │ aligned to t+1)        │ prediction)
                       │                        │
                       ▼                        ▼
                ┌────────────────────────────────────────────┐
                │            L_pred = MSE(ẑ_next, z̃)        │
                └────────────────────────────────────────────┘

         z̃ also feeds SIGReg:    L_sigreg = SIGReg(z̃)
```

The same `projector` is applied to both arms:

- **Source arm**: `projector(encoder(pixels_source))` → fed to the
  predictor, then `pred_proj` after the predictor.
- **Target arm**: `projector(encoder(pixels_target))` → directly
  becomes the prediction target $\tilde{\mathbf z}_{t+1}$.

So `projector` is run on every frame in the window, and its output is
both the predictor's *input embedding* and the prediction loss's
*target*. `pred_proj` is run only on the predictor's output, mapping
it back into the same $D = 192$ space where the comparison happens.

## 3. Why a wide non-linear projection if the output dim equals the input dim?

The encoder produces $192$-D CLS vectors. The most parsimonious
arrangement would be to compute the prediction MSE directly on those
vectors. The architecture instead routes them through a non-linear
$192 \to 2048 \to 192$ MLP at both arms. Two design reasons motivate
this choice:

1. **Decoupling the SIGReg "view" from the encoder.** The regulariser
   acts on the projector's output, not on the encoder's CLS directly.
   This lets the encoder be free to use whatever internal coordinates
   it likes, while the projector reshapes those coordinates into a
   distribution that SIGReg can compare against $\mathcal N(0, I_D)$.
   BatchNorm1d in the projector further whitens the activations along
   the hidden axis, which empirically stabilises the SIGReg gradient.
2. **A learnable comparison metric for the prediction loss.** Because
   $\text{MSE}(\text{pred\_proj}(\hat{\mathbf z}), \text{projector}(\mathbf z_{t+1}))$
   is taken in the projector-output space (not pixel space and not
   encoder-CLS space), the loss is effectively MSE under a *learned*
   inner-product structure. The two MLPs jointly carve out the metric
   under which predictability and Gaussianity are simultaneously
   enforced.

The projector-output space is where the *loss* lives. The encoder /
predictor token dim ($D = 192$) is where the *model* lives. Both
spaces are $192$-D in `lewm-rs`; the projectors are non-linear, not
dimensionality-changing.

## 4. The end-to-end gradient contract

Recall from [Concepts §2.4](../concepts/lewm.md): there is **no
stop-gradient, no EMA**. Both arms use the *same* encoder and the
*same* projector. Gradient flows freely.

- Source arm gradient path:
  `pixels → encoder → projector → predictor → pred_proj → L_pred`.
- Target arm gradient path:
  `pixels → encoder → projector → L_pred` (and also
  `→ SIGReg → L_sigreg`).

So `encoder` and `projector` each receive gradient from **two** paths
on every step: the source-arm prediction path and the target-arm
prediction path (plus the target arm's SIGReg path). This symmetry —
both arms using identical, jointly-updated weights — is the
"end-to-end" character of LeWM that distinguishes it from earlier
JEPAs. The contract is pinned in [RFC 0003 §4.1.2] under
*RFC0003-001 [MUST]*.

## 5. Initialisation

Both linear layers in both MLPs use truncated-normal init with
$\sigma = 0.02$, biases zero. The BatchNorm1d's affine parameters are
initialised to $\gamma = 1$, $\beta = 0$, and its running statistics
to $\mu = 0$, $\sigma^2 = 1$ — Burn's defaults, matching upstream.
No special zero-init trick is applied; the MLPs are not conditioned on
anything, so they benefit from a vanilla init.

## 6. Parameter count

Per MLP (trainable):

| Tensor | Shape | Count |
|--------|------:|------:|
| `fc1.weight` | $2048 \times 192$ | 393 216 |
| `fc1.bias`   | $2048$ | 2 048 |
| `norm.weight` (BN1d scale) | $2048$ | 2 048 |
| `norm.bias`   (BN1d shift) | $2048$ | 2 048 |
| `fc2.weight` | $192 \times 2048$ | 393 216 |
| `fc2.bias`   | $192$ | 192 |
| **MLP total (trainable)** | | **792 768 (~0.79 M)** |

There are two such MLPs (`projector` and `pred_proj`), so the combined
budget is $\sim 1.59\text{ M}$ trainable parameters out of $18.04$ M
total. Each MLP additionally carries three BatchNorm1d *buffers* —
`running_mean` ($2048$), `running_var` ($2048$), `num_batches_tracked`
(scalar) — that are loaded from the reference checkpoint but not
optimised. See the [Parameter inventory](./parameter-inventory.md) for
the canonical breakdown.

## 7. Parity tests

`crates/lewm-core/tests/parity_pred_proj.rs` runs both `projector` and
`pred_proj` on the fixture inputs and compares to the upstream dump.

| Test | Tolerance | Status |
|------|-----------|--------|
| `parity_projector_output` | $L_\infty < 10^{-4}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |
| `parity_pred_proj_output` | $L_\infty < 10^{-4}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |

## 8. Source pointers

| Topic | Source |
|-------|--------|
| `MlpConfig` | `crates/lewm-core/src/config.rs` |
| `Mlp` | `crates/lewm-core/src/mlp.rs` |
| GELU variants | `crates/lewm-core/src/tensor_ops.rs` |
| Parity tests | `crates/lewm-core/tests/parity_pred_proj*.rs` |

[RFC 0003 §4.1.2]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0003-sigreg-and-loss-functions.md#412-gradient-contract


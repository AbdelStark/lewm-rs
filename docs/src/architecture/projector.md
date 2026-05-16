# Projector and pred-proj MLPs

> **Motivation.** Two small MLPs — `projector` and `pred_proj` —
> bracket the SIGReg space. They are not glamorous, but they are
> exactly where the prediction loss is computed and where the SIGReg
> regulariser is applied. This page documents both.
>
> **Position.** Fourth module page in [Part II](./overview.md).
>
> **What you should leave with.** The shape contract of each MLP, the
> rationale for the lift to 1024-D, and which loss is computed in which
> space.

## 1. The two MLPs

Both are instances of the same `Mlp` struct in
`crates/lewm-core/src/mlp.rs`:

```rust,ignore
#[derive(burn::module::Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: burn::nn::Linear<B>,    // Linear(in_dim → hidden_dim)
    fc2: burn::nn::Linear<B>,    // Linear(hidden_dim → out_dim)
    act: GeluVariant,
}
```

with `MlpConfig`:

| Field | `projector` | `pred_proj` |
|-------|------------:|------------:|
| `in_dim` | 192 | 192 |
| `hidden_dim` | 2048 | 2048 |
| `out_dim` | 1024 | 1024 |
| `act` | exact-erf GELU | exact-erf GELU |

Forward:

```text
out = self.fc2.forward( gelu( self.fc1.forward(x) ) )
```

## 2. Where each MLP sits in the dataflow

```text
                   encoder output           predictor output
                   (B, T, 192)             (B, T, 192)
                       │                        │
                       ▼                        ▼
                ┌─────────────┐          ┌──────────────┐
                │  projector  │          │  pred_proj   │
                │  192 →2048  │          │  192 →2048   │
                │  GELU       │          │  GELU        │
                │  2048→1024  │          │  2048→1024   │
                └──────┬──────┘          └──────┬───────┘
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
*target*. `pred_proj` is run only on the predictor's output, mapping it
back into the same 1024-D space where the comparison happens.

## 3. Why lift to 1024-D before computing the loss?

The encoder produces 192-D CLS vectors. The prediction MSE could in
principle be computed in 192-D space; the architecture chose to lift to
1024-D first. Two reasons:

1. **SIGReg lives in 1024-D.** The regulariser is applied to the
   projector's output. SIGReg's $K = 1024$ random directions need a
   sufficiently wide ambient space to be approximately orthogonal in
   expectation. 1024-D is the natural choice — directly matching $K$.
2. **The prediction loss benefits from a wider comparison space.** In
   the 192-D encoder dim, small errors in the predictor are amplified
   relative to total feature magnitude. In the 1024-D projected space,
   the encoder's CLS is spread over a wider basis and the MSE per-dim
   becomes smaller, which empirically gives smoother gradients.

The 1024-D space is where the *loss* lives. The 192-D space is where
the *model* lives. The two are connected by `projector` and `pred_proj`.

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
$\sigma = 0.02$, biases zero. No special zero-init trick is applied; the
MLPs are not conditioned on anything, so they benefit from a vanilla
init.

## 6. Parameter count

Per MLP:

| Tensor | Shape | Count |
|--------|------:|------:|
| `fc1.weight` | $192 \times 2048$ | 393 216 |
| `fc1.bias`   | $2048$ | 2 048 |
| `fc2.weight` | $2048 \times 1024$ | 2 097 152 |
| `fc2.bias`   | $1024$ | 1 024 |
| **MLP total** | | **2 493 440 (~2.5 M)** |

There are two of these (`projector` and `pred_proj`), so the combined
projector/pred-proj budget is ~5 M parameters out of 18 M total. This
is larger than the headline LeWM paper number suggests — most readers
think of the projector as "small" — but at the LeWM scale, the
projector and pred-proj together carry significant capacity. See the
[Parameter inventory](./parameter-inventory.md) for the canonical
breakdown.

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

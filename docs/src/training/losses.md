# Loss functions: prediction + SIGReg

> **Motivation.** The whole training signal is in two losses. This page
> documents them at the level of detail required for a from-scratch
> implementation to agree with `lewm-rs` to $10^{-3}$ on the SIGReg
> scalar.
>
> **Position.** Concrete companion to [SIGReg concepts](../concepts/sigreg.md).
>
> **What you should leave with.** The full equations, the F32 invariant,
> the RNG-substream contract, and pointers to the Rust source.

## 1. The total loss

$$
\mathcal L(\theta, \phi) \;=\; \mathcal L_{\text{pred}}(\theta, \phi) \;+\; \lambda\, \mathcal L_{\text{sigreg}}(\theta).
$$

LeWM's default: $\lambda = 1.0$.

The two terms are computed from the same forward pass; they share the
encoder output and the projector's output. Gradient flows from both
into the encoder and projector, and from $\mathcal L_{\text{pred}}$
also into the predictor, action encoder, and `pred_proj`.

## 2. Prediction loss

Given:
- `pred_z_proj`: the source-arm prediction (output of `pred_proj`
  applied to the predictor output), shape $(B, T, D)$,
- `target_z_proj`: the projector applied to the encoder embedding of
  the *next-step* frame, shape $(B, T, D)$.

$$
\mathcal L_{\text{pred}} \;=\; \frac{1}{B\cdot T\cdot D}\sum_{b,t,d} \bigl(\hat z_{b,t,d} - \tilde z_{b,t,d}\bigr)^2.
$$

That is, ordinary MSE over all axes. In `lewm-rs` v1, $D = 192$ —
both arms live in the projector-output space, which has the same
dimensionality as the encoder's CLS embedding.

```rust,ignore
pub fn prediction_loss<B: Backend>(
    pred:   Tensor<B, 3>,   // (B, T, D)
    target: Tensor<B, 3>,   // (B, T, D)
) -> Tensor<B, 1> {
    debug_assert_eq!(pred.dims(), target.dims());
    let diff = pred - target;
    (diff.clone() * diff).mean()
}
```

### 2.1 Gradient contract

[RFC 0003 §4.1.2] pins:

- **RFC0003-001 [MUST]** — `target` shares its encoder with the source
  arm. **No stop-gradient.** Gradient flows through `target`'s encoder.
- **RFC0003-002 [MUST]** — `pred` flows back through `pred_proj`,
  `predictor`, `action_enc`, `projector`, and `encoder`.

Together, these mean each gradient step updates the encoder with two
distinct prediction-related signals (source path and target path) plus
the SIGReg path.

## 3. SIGReg loss

Given the projector output for one batch, $z \in \mathbb R^{B \times
(T+1) \times D}$ with $D = 192$ (flattened to $\mathbb R^{N \times D}$
with $N = B \cdot (T+1)$), SIGReg is:

$$
\boxed{\;\mathcal L_{\text{sigreg}}(z) \;=\; \frac{1}{K}\sum_{k=1}^{K} \sum_{j=0}^{J-1} q_j\, w(t_j)\, \bigl[(c_{k,j} - \phi(t_j))^2 + s_{k,j}^2\bigr]\;}
$$

with

$$
c_{k,j} = \frac{1}{N}\sum_n \cos(t_j\,\mathbf p_k^\top \mathbf z_n),\quad
s_{k,j} = \frac{1}{N}\sum_n \sin(t_j\,\mathbf p_k^\top \mathbf z_n),
$$

$$
\phi(t) = w(t) = e^{-t^2/2},\quad t_j = j\,\Delta t,\;\Delta t = t_{\max}/(J-1),\quad t_{\max}=3,\;K=1024,\;J=17.
$$

Trapezoid weights: $q_0 = q_{J-1} = \Delta t / 2$, otherwise $q_j = \Delta t$.

See [SIGReg concepts](../concepts/sigreg.md) for the derivation. The
implementation is in `crates/lewm-core/src/losses/sigreg.rs`:

```rust,ignore
pub fn sigreg_loss<B: Backend>(
    z: Tensor<B, 2>,          // (N, D) in F32
    proj_matrix: Tensor<B, 2>, // (K, D), unit-norm rows, sampled fresh
    t_grid: &[f32],            // length J, in F32
    trap_weights: &[f32],      // length J
) -> Tensor<B, 1> {
    let n = z.dims()[0] as f32;
    let y = z.matmul(proj_matrix.transpose());   // (N, K)
    let mut total = Tensor::<B, 1>::zeros([1], &z.device());
    for (j, &t) in t_grid.iter().enumerate() {
        let phi  = (-0.5 * t * t).exp();
        let w_j  = phi;
        let q_j  = trap_weights[j];

        let yt = y.clone() * t;                  // (N, K)
        let c  = yt.cos().mean_dim(0);           // (K,)
        let s  = yt.sin().mean_dim(0);           // (K,)
        let r  = (c - phi).powf(2.0) + s.powf(2.0);  // (K,)
        let i  = (q_j * w_j) * r;                // (K,)
        total = total + i.mean();
    }
    total
}
```

### 3.1 The F32 invariant

**INV-005 [MUST]** — Every operation inside `sigreg_loss` — projection,
$\cos/\sin$, $\phi$, characteristic-function difference, trapezoid
weighting — **runs in F32 even when the surrounding loop is in BF16**.

The reason: the highest-frequency knot is $t_{\max} = 3$, and the
projection $\mathbf p_k^\top \mathbf z_n$ can be on the order of unity,
so $\cos(3 \cdot 1) = \cos(3) \approx -0.99$ is near a stationary
point of $\cos$. BF16's 7-bit mantissa is too coarse to resolve
fluctuations on the order of $\sin(\delta t \cdot \mathbf p^\top \mathbf
z)$ for small perturbations, which destabilises the gradient. F32 is
required.

The cast back to the surrounding (BF16 mixed) graph happens at the
**scalar output**:

```rust,ignore
let l_sigreg_f32 = sigreg_loss(z_proj_f32, proj_f32, &t_grid, &trap_w);
let l_sigreg     = l_sigreg_f32.cast::<BF16>();   // or float_elem<B>::dtype
```

### 3.2 RNG substream

**RFC0003-003 [MUST]** — The projection matrix is re-sampled at every
call from the named sub-stream `rng:sigreg_sketch`:

```rust,ignore
let mut raw: Tensor<B, 2> = Tensor::random_with(
    [K, D],
    Distribution::Normal(0.0, 1.0),
    rng.substream("sigreg_sketch"),
);
let norm = raw.clone().powf(2.0).sum_dim(1).sqrt();    // (K, 1)
let proj_matrix = raw / norm;                          // unit-norm rows
```

The substream is advanced on every step. The seed for the substream is
derived deterministically from the master seed and the substream name,
so a resume from step $n$ produces the same projection sequence as a
fresh run.

## 4. The total loss in code

In `crates/lewm-train/src/step.rs`:

```rust,ignore
// Cast the projector output to F32 for SIGReg.
let z_proj_f32 = z_proj.clone().cast::<f32>().reshape([B*(T+1), D]);

let l_pred   = prediction_loss(pred_z_proj, target_z);          // BF16-mixed OK
let l_sigreg = sigreg_loss(z_proj_f32, proj_matrix, &t_grid, &trap_w)
                   .cast::<float_elem<B>>();
let l_total  = l_pred + lambda * l_sigreg;
```

The `lambda` constant comes from the TOML config (`configs/pusht.toml`,
`configs/so100.toml`), defaulting to $1.0$. Here $D = 192$ for both
the PushT and SO-100 defaults.

## 5. Per-step diagnostics

Beyond the two scalars, RFC 0003 §5 specifies three collapse-detection
probes computed on the *unprojected* CLS embeddings of the batch:

| Probe | Computation | Threshold (TOL-007/008/009) |
|-------|-------------|------------------------------|
| Per-dim CLS variance | $\min_d \mathrm{Var}_b(z_b^{(d)})$ | $\geq 0.05$ |
| Mean abs CLS | $\max_d \lvert\mathbb E_b[z_b^{(d)}]\rvert$ | $\leq 5.0$ |
| Pairwise CLS cosine | $\mathbb E_{b\neq b'}[\cos(z_b, z_{b'})]$ | $\leq 0.85$ |

If any probe trips, the run emits a `collapse_suspected_{step}.json`
diagnostic and aborts. In both PushT and SO-100 training runs to date,
none of the probes tripped at any step.

## 6. Source pointers

| Topic | Source |
|-------|--------|
| Prediction loss | `crates/lewm-core/src/losses/prediction.rs` |
| SIGReg loss | `crates/lewm-core/src/losses/sigreg.rs` |
| Collapse probes | `crates/lewm-core/src/losses/collapse_probes.rs` |
| Total-loss assembly | `crates/lewm-train/src/step.rs` |
| Reference Python loss | upstream `module.py::SIGReg` |
| Parity test (scalar) | `crates/lewm-core/tests/parity_sigreg.rs` |

[RFC 0003 §4.1.2]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0003-sigreg-and-loss-functions.md#412-gradient-contract

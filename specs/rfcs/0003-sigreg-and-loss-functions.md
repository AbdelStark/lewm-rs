---
rfc: "0003"
title: "SIGReg, prediction loss, gradient contracts"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§4.1 SIGReg", "§5.2 [loss]", "§6.2 Collapse detection"]
depends_on: ["0001", "0002"]
related: ["0008", "0013"]
---

# RFC 0003 — SIGReg, prediction loss, gradient contracts

> **Status:** Accepted · **Version:** 1.0.0
>
> The two-loss objective is what makes LeWM tick. Prediction MSE pulls the encoder toward predictive features; SIGReg keeps the latent distribution standard-Gaussian, preventing collapse without the multi-term gymnastics of VICReg/Barlow Twins. This RFC fixes the numerical recipe for both losses, the F32 invariant inside SIGReg, the gradient contract (what flows where), and the per-step diagnostics needed for collapse detection.

---

## 1. Introduction

### 1.1 Motivation

SIGReg is mathematically straightforward but operationally fiddly: 1024 random unit-norm projections, 17 frequency knots, a Gaussian window, and trapezoid-rule integration. Implemented wrong, it silently under-regularizes and the encoder collapses to a constant in 200 steps. Implemented right but in BF16, the high-frequency trig ops drift and the same collapse happens. The point of this RFC is to make the implementation exactly right and exactly stable.

### 1.2 Goals

1. Define `L_pred` and `L_sigreg` to the level of a worked numerical example.
2. Pin the F32 invariant inside SIGReg (INV-005).
3. Specify what the gradient looks like (which arms receive it; which are detached).
4. Specify the random projection sketch and its RNG sub-stream.
5. Specify the collapse-detection probes that read off the same module.
6. Specify the per-step metrics emitted to the observability layer.

### 1.3 Non-goals

- Loss scheduling (the `λ` ramp, if any) — defer to [RFC 0005 §5](0005-training-system.md).
- Auxiliary losses beyond the two in this RFC — none planned.
- ONNX export of the loss computation — not required (losses are not in the inference path).

---

## 2. Conventions

Symbols per [`glossary.md` §6](../glossary.md). In particular `B`, `T`, `D`, `K=1024`, `J=17`, `λ`.

---

## 3. Background

The Epps–Pulley test ([Epps & Pulley, 1983]) is a goodness-of-fit test against a Gaussian by comparing the empirical characteristic function (CF) to the standard normal CF on a set of frequencies. SIGReg uses a *sketch* of the empirical CF: instead of operating in `D` dimensions where empirical-CF estimation is statistically expensive, it projects the latent onto `K` random unit-norm 1-D directions and applies the test in each. The aggregate loss is the average across projections.

The standard normal CF at real frequency `t` is `ϕ(t) = exp(−t²/2)` (the moment generating function evaluated at `it`, real part). The Epps–Pulley statistic compares `Re(ψ_emp(t)) → ϕ(t)` and `Im(ψ_emp(t)) → 0`. The integral over `t` is approximated by trapezoid quadrature on a finite grid weighted by a Gaussian window `exp(−t²/2)` to taper the tail.

Upstream `module.py::SIGReg` lines 13–39 (in `lucas-maes/le-wm`) implements this. Our Rust implementation reproduces it line-by-line.

---

## 4. Detailed design

### 4.1 Prediction loss `L_pred`

#### 4.1.1 Definition

Given:

- `pred: (B, T_p, D)` — predictor output (post `pred_proj`).
- `target: (B, T_p, D)` — encoder forward of the next-step pixels, then `projector` applied.

```
L_pred = mean( (pred - target)^2 )    # mean over B, T_p, D
```

#### 4.1.2 Gradient contract

**RFC0003-001 [MUST]** — `target` is computed from the **same** encoder used for the source arm. It is **not** stop-gradient and **not** EMA. Gradient flows through `target` back to the encoder.

**RFC0003-002 [MUST]** — `pred` is computed via `pred_proj(predictor(projector(encoder(pixels_source)), action_encoder(actions)))`. Gradient flows through `pred_proj`, `predictor`, `action_encoder`, `projector`, and `encoder` for the source arm.

The combination of (1) target-arm encoder gradient and (2) SIGReg pulling latents to standard normal is exactly what makes the design "end-to-end stable" (PRD §5.4): without (1), gradient asymmetry would induce instability; without (2), the encoder would collapse to a constant satisfying the MSE trivially.

#### 4.1.3 Implementation

```rust
pub fn prediction_loss<B: Backend>(
    pred: Tensor<B, 3>,    // (B, T_p, D)
    target: Tensor<B, 3>,  // (B, T_p, D)
) -> Tensor<B, 1> {
    debug_assert_eq!(pred.dims(), target.dims());
    let diff = pred - target;
    (diff.clone() * diff).mean()
}
```

This is the trivial path; no precision tricks needed (BF16-mixed handles it inside the autodiff backend).

### 4.2 SIGReg `L_sigreg`

#### 4.2.1 Algorithm

Given input `z: (B, T, D)` (the *projected* target embeddings — see §4.2.7 for which tensor to feed):

1. **Flatten** to `(N, D)` where `N = B · T`.
2. **Sample** a projection matrix `P ∈ ℝ^{K × D}` once per call (re-sampled per step), with each row unit-norm: `p_k ~ N(0, I_D / D)` then normalize.
3. **Project**: `Y = P · z^T ∈ ℝ^{K × N}`, i.e., `Y[k, n] = ⟨p_k, z[n]⟩`. Each row of `Y` is the empirical 1-D projection of the latent under direction `k`.
4. **Compute** the empirical characteristic function at the `J` frequency knots `t_j ∈ {0, 3/(J-1), 2·3/(J-1), …, 3}`. For each `(k, j)`:
   - `c_{k,j} = mean_n cos(t_j · Y[k, n])`
   - `s_{k,j} = mean_n sin(t_j · Y[k, n])`
5. **Compute** the target characteristic function (real part) `ϕ_j = exp(−t_j² / 2)`. The imaginary part of the standard-normal CF is zero.
6. **Compute** the integrand at each knot:
   - `r_{k,j} = (c_{k,j} − ϕ_j)² + s_{k,j}²`
7. **Apply** the Gaussian window `w_j = exp(−t_j² / 2)` (which equals `ϕ_j` in this design — note this is intentional, per upstream).
8. **Integrate** over `t` using the trapezoid rule. For knots equally spaced by `Δt = 3/(J-1)`, the trapezoid weights are:
   - `q_0 = q_{J-1} = Δt / 2`
   - `q_j = Δt for j ∈ {1, …, J-2}`
9. **Compute** `I_k = Σ_j q_j · w_j · r_{k,j}`. The Gaussian window inside the integral makes this a windowed Epps–Pulley statistic.
10. **Aggregate** across projections: `L_sigreg = mean_k I_k`.

#### 4.2.2 Mathematical statement

$$
\mathcal{L}_{\text{sigreg}}(z) = \frac{1}{K}\sum_{k=1}^{K} \sum_{j=0}^{J-1} q_j\, w(t_j)\, \Bigl[\bigl(c_{k,j} - \phi(t_j)\bigr)^2 + s_{k,j}^2\Bigr]
$$

with `c_{k,j} = E_n[cos(t_j p_k^T z_n)]`, `s_{k,j} = E_n[sin(t_j p_k^T z_n)]`, `ϕ(t) = w(t) = exp(−t²/2)`, `q_j` trapezoid weights on `[0, 3]` with `J=17` knots.

#### 4.2.3 Hyperparameter values

| Symbol | Value | Source |
|--------|-------|--------|
| `K` (num_proj) | `1024` | upstream `module.py::SIGReg.__init__` default |
| `J` (knots) | `17` | upstream default |
| `t_max` | `3.0` | upstream default |
| `t_grid` | `linspace(0, 3, 17)` | derived |
| `λ` | `1.0` (default; tunable) | PRD §5.2 |

#### 4.2.4 Random projection sampling (RNG contract)

**RFC0003-003 [MUST]** — The projection matrix `P` is sampled at every SIGReg call from the named RNG sub-stream `rng:sigreg_sketch` defined in [RFC 0013 §4](0013-determinism-and-reproducibility.md). The sampling is:

```
1. raw ~ N(0, 1) of shape (K, D)        # sampled element-wise
2. norm = sqrt(sum(raw**2, dim=-1, keepdim=True))    # (K, 1)
3. P = raw / norm                                     # rows unit-norm
```

**RFC0003-004 [MUST]** — `P` is **not** stored as a parameter. It is re-sampled every step. Two reasons: (a) it is intended to be a Monte-Carlo estimator; (b) storing it would inflate the checkpoint without benefit.

**RFC0003-005 [MUST]** — Sampling **MUST** consume the RNG sub-stream `rng:sigreg_sketch` deterministically: given the same step number and the same global seed, the matrix `P` is identical.

#### 4.2.5 F32 invariant (INV-005)

**RFC0003-006 [MUST]** — Every internal tensor in `L_sigreg` computation **MUST** be `f32`. This includes:

- `t_grid`, `phi`, `weights` (constants — declared `Tensor<B::Float<F32>>` even when the outer backend is BF16-mixed).
- `P` (the projection matrix).
- `Y` (the projections).
- `c, s, r` (intermediate stats).
- `I` (per-projection integrals).
- `L_sigreg` (final scalar).

**Implementation strategy:** if the outer autodiff backend is `Autodiff<Cuda<BF16>>`, the SIGReg input `z` is cast `z.into_data().convert::<f32>()` and then re-wrapped into an `Autodiff<Cuda<F32>>` tensor for the SIGReg subgraph. The gradient out of SIGReg is cast back to BF16 at the boundary. This requires a `mixed_precision::scope` helper documented in [RFC 0005 §5.6](0005-training-system.md).

**RFC0003-007 [MUST]** — The F32 cast `MUST NOT` round-trip through CPU memory (i.e., is a device-side cast). Burn supports this via `Tensor::cast::<F32, _>()`.

#### 4.2.6 Constants are precomputed

```rust
struct SigRegConstants<B: Backend> {
    t_grid:    Tensor<B, 1>,            // shape (J,)  values = linspace(0, 3, J)
    phi:       Tensor<B, 1>,            // shape (J,)  values = exp(-t**2 / 2)
    window:    Tensor<B, 1>,            // shape (J,)  values = exp(-t**2 / 2) (same as phi)
    trap:      Tensor<B, 1>,            // shape (J,)  trapezoid weights
    knots:     usize,                   // J
    num_proj:  usize,                   // K
    t_max:     f32,                     // 3.0
}
```

**RFC0003-008 [MUST]** — These constants are computed once at module construction and stored as registered buffers (`#[module(register)]`). They are **not** trainable. They serialize to the checkpoint but their values are recomputable from `(J, t_max)`.

#### 4.2.7 Which tensor to feed?

Upstream code applies SIGReg to **the projector output of the target embedding** — i.e., the same tensor that the predictor is supposed to match. In our wrapper:

```
target_emb = projector(encoder(next_frames))  # (B, T_target, D)
L_sigreg   = sigreg(target_emb.cast::<F32>()) # F32 path
```

**RFC0003-009 [MUST]** — SIGReg is computed on `projector(encoder(next_frames))`, **not** on raw CLS, **not** on `pred_proj` output, **not** on the predictor input. This matches upstream.

#### 4.2.8 Reference implementation skeleton

```rust
// crates/lewm-core/src/losses/sigreg.rs

use burn::tensor::{Tensor, backend::Backend};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

#[derive(burn::module::Module, Debug)]
pub struct SigReg<B: Backend> {
    consts: SigRegConsts<B>,
}

#[derive(burn::module::Module, Debug)]
pub struct SigRegConsts<B: Backend> {
    /// t_grid, phi, window, trap pre-computed
    t_grid: burn::module::Param<Tensor<B, 1>>,
    phi:    burn::module::Param<Tensor<B, 1>>,
    window: burn::module::Param<Tensor<B, 1>>,
    trap:   burn::module::Param<Tensor<B, 1>>,
}

impl<B: Backend> SigReg<B> {
    /// Build SIGReg with the given knot count and t_max, in F32.
    pub fn new(num_proj: usize, knots: usize, t_max: f32, device: &B::Device) -> Self {
        let t_grid = Tensor::<B, 1>::from_data(/* linspace(0, t_max, knots) */, device);
        let phi    = (-t_grid.clone() * t_grid.clone() * 0.5).exp();
        let window = phi.clone();
        let trap   = build_trapezoid_weights(knots, t_max, device);
        Self {
            consts: SigRegConsts {
                t_grid: t_grid.into(),
                phi: phi.into(),
                window: window.into(),
                trap: trap.into(),
            }
        }
    }

    /// Forward.
    ///
    /// # Shape
    /// - `z`: (B, T, D), any dtype on entry — cast to F32 internally.
    /// - return: (1,), F32.
    pub fn forward(&self, z: Tensor<B, 3>, rng: &mut ChaCha20Rng) -> Tensor<B, 1> {
        // 1. flatten and cast
        let [bsz, t, d] = z.dims();
        let z_flat = z.reshape([bsz * t, d]).cast::<f32>();        // (N, D)
        let n = bsz * t;

        // 2. sample P (K, D) on device, unit-normalized rows
        let p = sample_unit_norm(self.num_proj(), d, rng, z_flat.device());

        // 3. project: Y = z @ P^T -> (N, K)
        let y = z_flat.matmul(p.transpose());                       // (N, K)

        // 4. multiply by t grid: arg = t_j * Y, broadcast over knots
        //    arg shape: (J, K, N) by outer product, then permute to (K, J, N)
        let t = self.consts.t_grid.val();                           // (J,)
        let arg = t.unsqueeze::<3>().unsqueeze::<3>() * y.transpose().unsqueeze::<3>();
        //        ^^^ careful with broadcasting; see Appendix A.1 for the exact shape walk

        let c = arg.clone().cos().mean_dim(2);                      // (K, J), mean over N
        let s = arg.sin().mean_dim(2);                              // (K, J)

        // 5. compute residual: (c - phi)^2 + s^2
        let phi = self.consts.phi.val();                            // (J,)
        let diff = c - phi.unsqueeze::<2>();                        // (K, J)
        let res = diff.clone() * diff + s.clone() * s;              // (K, J)

        // 6. weight and integrate
        let w = (self.consts.window.val() * self.consts.trap.val()).unsqueeze::<2>();   // (1, J)
        let per_proj = (res * w).sum_dim(1);                        // (K,)

        // 7. mean over projections
        per_proj.mean()                                              // (1,) F32
    }
}
```

**RFC0003-010 [MUST]** — The exact broadcasting in step 4 **MUST** match the Python reference. Appendix A.1 contains the explicit shape walk to avoid off-by-axis errors.

### 4.3 Total loss

```rust
pub fn total_loss<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    sigreg_input: Tensor<B, 3>,
    sigreg: &SigReg<B>,
    rng: &mut ChaCha20Rng,
    lambda: f64,
) -> JepaLosses<B> {
    let l_pred  = prediction_loss(pred.clone(), target.clone());
    let l_sigreg = sigreg.forward(sigreg_input, rng);
    let total = l_pred.clone() + l_sigreg.clone() * lambda as f32;
    JepaLosses { pred: l_pred, sigreg: l_sigreg, total }
}
```

**RFC0003-011 [MUST]** — `lambda` is configured per [RFC 0018 §4.6](0018-configuration-system.md), default `1.0`. Sweeping is allowed only via ml-intern over `{0.3, 0.5, 1.0, 2.0, 5.0}` per PRD §6.6.

### 4.4 Collapse detector

The collapse detector reads three quantities every `eval_every_n_steps` (default 100) on a held-out 32-frame batch:

```rust
pub struct CollapseProbe {
    /// E[‖cls‖_∞] — mean absolute over the batch. Floor: must stay < cls_mean_abs_ceiling.
    pub mean_abs_cls: f32,
    /// Mean over feature dims of Var(cls). Floor: must stay > cls_var_floor.
    pub cls_variance_per_dim_mean: f32,
    /// E[cos(cls_i, cls_j)] for random pairs i,j. Ceiling: < cls_cosine_pair_ceiling.
    pub mean_pairwise_cosine: f32,
}
```

**RFC0003-012 [MUST]** — Thresholds for "trip" are TOL-007/008/009 from [`glossary.md` §4](../glossary.md):

- `mean_abs_cls > 5.0` → trip
- `cls_variance_per_dim_mean < 0.05` → trip
- `mean_pairwise_cosine > 0.85` → trip

**RFC0003-013 [MUST]** — Three consecutive trips → emit `CRITICAL collapse_suspected step={N}` to stdout, write `collapse_suspected_{N}.json`, and add a flag to the per-run report. The training loop **MUST NOT** auto-abort — the operator decides based on the full trace.

**RFC0003-014 [MUST]** — The probe runs the encoder on the held-out batch with **no_grad** semantics (Burn `Tensor::no_grad()`), to avoid polluting gradients.

### 4.5 Metrics emitted

Every step the loss subsystem emits the following metrics via `lewm-telemetry`:

```
loss/total                : f32   — L_pred + λ · L_sigreg
loss/pred                 : f32   — L_pred
loss/sigreg               : f32   — L_sigreg
loss/sigreg_per_proj_min  : f32   — min over k of I_k (degenerate-direction detector)
loss/sigreg_per_proj_max  : f32   — max over k of I_k
```

Every 100 steps additionally:

```
model/encoder_cls_var       : f32
model/encoder_cls_mean_abs  : f32
model/cls_cosine_pair_mean  : f32
```

These names are stable and form the dashboard contract with [RFC 0009](0009-observability-and-mlops.md).

---

## 5. Numerical contracts

### 5.1 Precision invariant

**INV-005 (reproduced from master spec §9.3):** SIGReg internal computation is F32 regardless of outer training precision. Violation is a defect.

The cast points are precisely:

1. On entry: `sigreg_input: Tensor<B::Float, 3>` → `Tensor<B::Float<F32>, 3>` via `cast::<f32>()`.
2. Inside SIGReg: every tensor is F32.
3. On exit: the scalar `L_sigreg` is F32; `total_loss` casts back to the outer dtype before combining if needed.

### 5.2 Reproducibility

Two runs with the same global seed and same backend produce identical `L_sigreg` values to the bit on a fixed input batch (TOL-003). Different seeds produce values within `ε_sigreg_seedfree_rel = 5e-2` (TOL-004) — the spread reflects the Monte-Carlo over `K=1024` projections.

### 5.3 Stability

**RFC0003-015 [SHOULD]** — In any single training run, `L_sigreg` should remain monotonically non-increasing (modulo noise) once the model converges. A regression of more than 50 % in `L_sigreg` over 1000 steps without a corresponding decrease in `L_pred` is a soft signal of instability and should be alerted by the dashboard.

---

## 6. Public API

```rust
// crates/lewm-core/src/losses/mod.rs

pub mod prediction;
pub mod sigreg;
pub mod collapse;

pub use prediction::prediction_loss;
pub use sigreg::{SigReg, SigRegConsts};
pub use collapse::{CollapseProbe, run_collapse_probe};
```

The `JepaLosses` type is owned by `jepa.rs`.

---

## 7. Testing strategy

### 7.1 Test inventory

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0003-PRED-001 | `pred_loss_shapes_and_value` | unit | FR-007, basic correctness |
| TST-0003-SR-001 | `sigreg_constants_match_python` | unit | Constants `t_grid`, `phi`, `trap` bit-exact |
| TST-0003-SR-002 | `sigreg_forward_value_on_fixed_input` | unit | Numerical value vs reference dump |
| TST-0003-SR-003 | `sigreg_f32_invariant_under_bf16_outer` | unit | INV-005 |
| TST-0003-SR-004 | `sigreg_gradient_flows_through_z` | unit | Gradient correctness via finite differences |
| TST-0003-SR-005 | `sigreg_rng_determinism` | unit | Same seed → same `P` → same loss |
| TST-0003-SR-006 | `sigreg_rng_independence_across_steps` | unit | Different step → different `P` |
| TST-0003-TOTAL-001 | `total_loss_lambda_scaling` | unit | `lambda=0` recovers `L_pred`; `lambda=2 → L_pred + 2 L_sigreg` |
| TST-0003-COL-001 | `collapse_probe_on_synthetic_collapsed_encoder` | unit | Probe trips on a constant-output encoder |
| TST-0003-COL-002 | `collapse_probe_on_synthetic_healthy_encoder` | unit | Probe does not trip on N(0, I) latents |

### 7.2 Reference dumps

`python/dump_sigreg_reference.py` runs upstream SIGReg on a fixed input (`B=4, T=4, D=384` with a fixed seed) and dumps `{P, Y, c, s, residual, L_sigreg}` to `tests/fixtures/sigreg_reference.npz`. The Rust test compares each intermediate.

### 7.3 Property-based tests

Property `P-1` — *scaling invariance up to expected change.* Multiplying `z` by `α` should change `L_sigreg` in a predictable way: SIGReg measures distance from N(0, I), so scaling by `α` shifts the distribution to N(0, α²) and the loss should increase quadratically in `α − 1`. Tested with `proptest` over `α ∈ [0.1, 10.0]`.

Property `P-2` — *translation sensitivity.* Adding a constant `c` to all of `z` shifts the empirical CF; the loss should increase, never decrease, for any `c ≠ 0`. Tested with `proptest` over `c ∈ [−5, 5]`.

Property `P-3` — *batch-invariance to a degree.* `L_sigreg(z) ≈ L_sigreg(shuffle(z))` to numerical precision when the same RNG seed is used. (Both sides flatten over the batch dim before projection.)

### 7.4 Negative tests

- `sigreg_rejects_rank2_input` — passing a 2-D tensor returns `LewmCoreError::InvalidShape`.
- `sigreg_rejects_zero_norm_projection` — if a sampled `p_k` has zero norm (rounding edge case), the implementation **MUST** resample. Tested by injecting a degenerate seed.

---

## 8. Operational considerations

### 8.1 Observability

Per §4.5 the loss subsystem emits a small fixed set of metrics. All emit through the `lewm-telemetry` facade; see [RFC 0009 §4](0009-observability-and-mlops.md).

### 8.2 Runbook

- **"`loss/sigreg` is NaN."** — almost certainly an F32 invariant violation. Verify INV-005 by enabling `RUST_LOG=lewm_core::losses=trace`; the trace logs the dtype of every tensor inside SIGReg.
- **"`loss/sigreg` is monotonically increasing."** — possible collapse in progress; check `model/encoder_cls_var`. If low, abort and resume from the last clean checkpoint.
- **"`loss/sigreg_per_proj_max / loss/sigreg_per_proj_min > 100`."** — a small number of projection directions are dominating the loss. Confirm `K=1024` is set; if so, this is a transient noise artefact and resolves within an epoch.

### 8.3 Capacity planning

SIGReg forward+backward cost is bounded by the `Y = P z^T` matmul and the per-knot trig ops:

```
matmul: (N x D) x (D x K)   = O(N · D · K)
trig:   (N · K · J)          = O(N · K · J)
```

For `B=64, T=8 → N=512`, `D=384`, `K=1024`, `J=17`: matmul is `2e8` ops, trig is `9e6` ops. Negligible compared to ViT forward (`O(layers · seq · D²) ≈ 12 · 197 · 384² ≈ 3.5e8` per sample). SIGReg adds ~5 % to step time on A10G.

---

## 9. Performance considerations

The fused path uses `burn::tensor::activation::cos` and `sin` which dispatch to cuDNN element-wise kernels. Memory peak inside SIGReg is the `(N, K)` projections tensor: `512 · 1024 · 4 bytes = 2 MB`. Trivial.

For very long horizon training (`T > 16`), the `(N, K)` size grows linearly; still within bounds.

---

## 10. Security considerations

The RNG sub-stream `rng:sigreg_sketch` is internal and non-secret. Reproducibility is the security-adjacent property at stake; an attacker who could influence the RNG could in principle bias SIGReg, but the only attack surface is the global seed which is configured by the operator. No mitigation needed beyond INV-007.

---

## 11. Alternatives considered

- **A1 — VICReg-style multi-term loss.** Rejected: explicitly out of scope per PRD §2 non-goals (no new JEPA architectures). Also empirically dominated by SIGReg's simplicity.
- **A2 — Anderson–Darling instead of Epps–Pulley.** Rejected: AD is sensitive to tail behaviour but does not vectorize as cleanly to random 1-D projections. Epps–Pulley is the right operator for the sketch-based form.
- **A3 — Reuse the projection matrix `P` across steps.** Rejected: with `K=1024` fixed projections, the gradient signal becomes biased. Re-sampling every step keeps the estimator unbiased.
- **A4 — Compute SIGReg in BF16.** Rejected by INV-005. The high-frequency trig is sensitive to mantissa precision at `t=3`.

---

## 12. Acceptance criteria

- [ ] `prediction_loss` implementation matches `mean(diff**2)` on a fixed input batch to 1e-6.
- [ ] `SigReg::forward` matches `python/dump_sigreg_reference.py` output to 1e-5 (same seed).
- [ ] INV-005 holds under `Autodiff<Cuda<BF16>>` — verified by `TST-0003-SR-003`.
- [ ] Gradient through `L_sigreg` is finite for arbitrary finite `z` — verified by fuzzing.
- [ ] `total_loss` implements `L_pred + λ L_sigreg` exactly.
- [ ] Collapse probe correctly trips on a synthetic collapsed encoder (TST-0003-COL-001).

---

## 13. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Off-by-axis in SIGReg broadcasting | M | H | Appendix A.1 explicit shape walk; reference dump compared element-wise |
| R-2 | BF16 trig drift undetected | L | H | INV-005 unit-tested; dedicated parity test at BF16 outer |
| R-3 | `λ` sweep regressions vs baseline | M | M | ml-intern sweep gated on T2 SHORT only; baseline `λ=1.0` always re-run |
| R-4 | Collapse detector false positive | M | L | Three-in-a-row trip required; operator decides |
| R-5 | RNG sub-stream drift between Rust `rand_chacha` and PyTorch `torch.manual_seed` | M | M | Parity test uses **same** stream by passing the projection matrix as input |

---

## 14. Open questions

OQ-2003-1 — Should `λ` be a schedule (warmup-style) rather than a constant? Upstream uses constant; v1 inherits. Re-evaluate after the lambda sweep in Phase 3.

---

## A. Appendix — explicit shape walk for SIGReg

The broadcasting in step 4 of §4.2.1 is the easiest place to introduce a silent bug. The explicit walk:

```
inputs:
    z      : (N=B·T, D)            F32
    t_grid : (J,)                  F32  values = [0, 3/16, 6/16, ..., 3]
    P      : (K, D)                F32  unit-norm rows

step A. Y = z @ P.T
    z:   (N, D)
    P.T: (D, K)
    Y:   (N, K)

step B. arg = t_grid * Y                 (we want shape (J, N, K))
    t_grid.shape: (J,)        → unsqueeze axes 1 and 2 → (J, 1, 1)
    Y.shape:      (N, K)      → unsqueeze axis 0       → (1, N, K)
    multiply → arg.shape: (J, N, K)

step C. cos/sin over arg                       → both (J, N, K)
    c_full = cos(arg)
    s_full = sin(arg)

step D. mean over N
    c = c_full.mean(dim=1)           → (J, K)
    s = s_full.mean(dim=1)           → (J, K)

step E. residual
    diff = c - phi.unsqueeze(1)      → (J, K)    phi.shape: (J,) -> (J,1)
    res  = diff*diff + s*s            → (J, K)

step F. weight and integrate
    weight = window * trap            → (J,)     element-wise
    weighted = res * weight.unsqueeze(1)  → (J, K)
    per_proj = weighted.sum(dim=0)    → (K,)

step G. final
    L_sigreg = per_proj.mean()        → scalar
```

Note: the **order of axes** in `arg` is `(J, N, K)`. Earlier draft code had `(K, J, N)` which is also valid but differs in stride pattern. The reference Python uses `(J, N, K)`; we match.

---

## 15. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0003.*

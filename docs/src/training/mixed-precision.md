# Mixed precision and F32 islands

> **Motivation.** Training at full F32 costs ~2× the memory and ~2× the
> wall time of BF16-mixed training. LeWM benefits from BF16 throughout
> most of the forward — except for three "F32 islands" where the BF16
> mantissa is too coarse and causes drift.
>
> **Position.** Fifth sub-page in [Part III](./pipeline.md).
>
> **What you should leave with.** Which ops run in F32 and which in
> BF16, why, and the BF16-vs-F32 tolerance (TOL-010).

## 1. The mixed-precision policy

| Op category | Datatype | Why |
|-------------|---------|-----|
| **Convolution / matmul (encoder, predictor)** | BF16 | Compute-bound; BF16 halves both memory and time without measurable accuracy cost. |
| **LayerNorm** | F32 | The variance computation is sensitive to mantissa precision; BF16 produces visible drift on long-tailed activations. |
| **AdaLN modulation linear** | F32 | Same reason; the modulation is added to a normalized signal, so any drift becomes a constant bias. |
| **Softmax (attention)** | F32 | The exponential of large negative numbers (masked positions) underflows in BF16. |
| **SIGReg sketch** | F32 | The $\cos(t \cdot \mathbf p^\top \mathbf z)$ for $t = 3$ requires F32 precision to resolve small perturbations near stationary points. |
| **Optimizer state (Adam moments)** | F32 | Standard practice; BF16 moments destabilise long training. |
| **Master weights** | F32 | Held in F32; cast to BF16 once per step for the forward, then F32-accumulator backward. |

This pattern — BF16 for compute, F32 for normalization, reduction, and
optimizer — is the standard "AMP" mixed-precision recipe and follows
Micikevicius et al. (2018) "Mixed Precision Training".

## 2. The F32 islands

Three ops sit inside the otherwise-BF16 forward and **must** run in
F32 for parity:

### 2.1 LayerNorm

Burn's `LayerNorm<B>` uses `B::FloatElem` for its internal accumulator
by default. For mixed-precision training, lewm-rs sets the LayerNorm's
internal accumulator to F32 even when `B::FloatElem = BF16`. This is
the "F32 island": the input is BF16, but the mean / variance / scale
are computed in F32 and only the output is cast back to BF16.

### 2.2 AdaLN modulation linear

The `ada_ln_modulation.weight` matmul that produces $(\gamma, \beta,
\alpha)$ runs in F32. The downstream `(1 + \gamma) \cdot \text{ln}_1 +
\beta` operation then runs in BF16 with F32-cast operands. This costs
~ 5 % extra wall time on A10G-large vs running everything in BF16, but
is required for parity.

### 2.3 SIGReg sketch

The entire SIGReg loss runs in F32. See [Losses](./losses.md) §3.1 for
the rationale.

## 3. The tolerance contract

Two tolerances pin mixed-precision behaviour:

| ID | Symbol | Default | Where used |
|----|--------|---------|------------|
| TOL-010 | `bf16_to_f32_max_rel` | $2\!\times\!10^{-2}$ | BF16 mixed-precision run vs full F32 run, end-of-epoch loss |
| TOL-005 | `ε_loss_smoke_rel` | $1\!\times\!10^{-2}$ | Local CPU smoke vs cloud BF16 step-100 loss |

TOL-010 is the headline tolerance: a full F32 training run and a
BF16-mixed run with the same seed and hyperparameters should converge
to losses within 2 % relative of each other at the end of every epoch.
Larger drift indicates a precision invariant has been broken.

## 4. The parity tests

| Test | Tolerance | Status |
|------|-----------|--------|
| `parity_encoder_mixed_precision` | rel. $< 2 \!\times\! 10^{-2}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |
| `parity_predictor_mixed_precision` | rel. $< 2 \!\times\! 10^{-2}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |

Both verify that the BF16 forward agrees with the F32 forward to within
2 % relative on the locked PushT fixture.

## 5. Picking the precision

The Burn backend type encodes the precision. Common choices:

- `Backend = burn_cuda::Cuda<f32>`: full F32 on CUDA. Used for parity
  tests and the SO-100 short-run that converges fast enough to not need
  BF16.
- `Backend = burn_cuda::Cuda<bf16>`: BF16 master weights — *not used*;
  destabilises long training.
- **`Backend = burn::backend::Autodiff<Cuda<f32>>` with explicit BF16
  casts at the matmul level**: the lewm-rs production training path.
  Master weights F32, matmuls BF16, F32 islands as listed in §2.

This last pattern is implemented in
`crates/lewm-train/src/mixed_precision.rs` by inserting explicit
`.cast::<bf16>()` calls at the matmul entry and exit, with F32
accumulators inside.

## 6. Source pointers

| Topic | Source |
|-------|--------|
| Cast helpers | `crates/lewm-train/src/mixed_precision.rs` |
| LayerNorm F32 wrapper | `crates/lewm-core/src/tensor_ops.rs` |
| Parity tests | `crates/lewm-core/tests/parity_*_mixed_precision.rs` |
| Burn `f32`/`bf16` types | `burn::tensor::ElementConversion` |

## 7. Bibliography

- Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., et al.
  (2018). *Mixed Precision Training*. ICLR.

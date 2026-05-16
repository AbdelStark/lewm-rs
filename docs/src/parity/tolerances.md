# Tolerances and what they bound

> **Motivation.** Every numerical tolerance in `lewm-rs` is named,
> versioned, and bounded. This page is the single canonical table.
>
> **Position.** Sub-page of [Part VI](./why-parity.md).
>
> **What you should leave with.** What each TOL- identifier means,
> where it is enforced, and the rationale.

## 1. The tolerance table

These come from [`specs/glossary.md` §4](https://github.com/AbdelStark/lewm-rs/blob/main/specs/glossary.md#4-numerical-tolerances-default-constants).
Override of any default requires an ADR.

| ID | Symbol | Default | Bounds |
|----|--------|--------:|--------|
| **TOL-001** | $\varepsilon_{\text{CLS,abs}}$ | $1.0 \times 10^{-4}$ | Parity: encoder CLS output, F32 |
| **TOL-002** | $\varepsilon_{\text{pred,abs}}$ | $1.0 \times 10^{-4}$ | Parity: predictor output, F32 |
| **TOL-003** | $\varepsilon_{\text{sigreg,abs}}$ | $1.0 \times 10^{-3}$ | Parity: SIGReg scalar, F32, identical RNG seed |
| **TOL-004** | $\varepsilon_{\text{sigreg,seed-free,rel}}$ | $5.0 \times 10^{-2}$ | Parity: SIGReg scalar, different RNG seed (sketch resampled) |
| **TOL-005** | $\varepsilon_{\text{loss,smoke,rel}}$ | $1.0 \times 10^{-2}$ | Local CPU smoke vs cloud smoke step-100 loss |
| **TOL-006** | $\varepsilon_{\text{warm-start,delta,abs}}$ | $0.0$ | SO-100 warm-start latent-MSE must beat from-scratch by ≥ 0 |
| **TOL-007** | $\text{cls\_var\_floor}$ | $0.05$ | Collapse: per-dim CLS variance lower bound |
| **TOL-008** | $\text{cls\_mean\_abs\_ceiling}$ | $5.0$ | Collapse: mean absolute CLS upper bound |
| **TOL-009** | $\text{cls\_cosine\_pair\_ceiling}$ | $0.85$ | Collapse: mean pairwise CLS cosine upper bound |
| **TOL-010** | $\varepsilon_{\text{BF16}\to\text{F32,rel}}$ | $2.0 \times 10^{-2}$ | BF16 mixed run vs full F32 run, end-of-epoch loss |
| **TOL-011** | $\text{grad\_norm\_ceiling}$ | $1.0 \times 10^{3}$ | Pre-clip grad norm; above this we abort with a diagnostic |

## 2. Rationale per tolerance

### TOL-001, TOL-002 ($10^{-4}$ L∞)

This is the dominant precision tolerance and is set by the
**numerical noise of F32 in a single transformer block**. The
typical magnitude of an attention output for the LeWM scale is
$O(1)$; a single F32 matmul accumulates rounding error on the
order of $10^{-7}$ per dot-product, scaled by $\sim$ token count
($\sim 256$) per layer. Twelve layers give $\sim 10^{-5}$ of
intrinsic drift. The $10^{-4}$ tolerance is 10× that floor, leaving
room for non-bit-identical kernel paths between PyTorch and Burn
while still catching algorithmic discrepancies.

### TOL-003 ($10^{-3}$ absolute, sigreg seeded)

SIGReg's value is in the range $[0, 1)$ during normal training,
peaking around $0.5$ at init and decaying to $\sim 10^{-6}$ at
convergence. An absolute tolerance of $10^{-3}$ is appropriate for
the value range at any point of training. The "seeded" qualifier
means both implementations use **the same projection matrix** —
otherwise the loss varies by ~5 % across resamples (see TOL-004).

### TOL-004 ($5 \times 10^{-2}$ relative, sigreg seed-free)

When the projection matrix is independently sampled in the two
implementations, the sigreg scalar varies due to the finite-batch,
finite-sketch Monte Carlo estimator. The variance is on the order of
$1/\sqrt{K} \approx 1/\sqrt{1024} \approx 3 \%$. The tolerance is 5 %
to leave headroom.

### TOL-005 ($1 \times 10^{-2}$ relative, smoke loss)

The CPU NdArray smoke run and the GPU CUDA smoke run on the same seed
should converge to the same step-100 loss within 1 %. Larger drift
indicates either a precision-island misconfiguration or a non-deterministic
op in the smoke path.

### TOL-006 (warm-start non-regression)

Warm-start should be at least as good as from-scratch at the same
training budget. This is a one-sided contract: warm-start may be
materially better, but it must not be worse.

### TOL-007 / TOL-008 / TOL-009 (collapse probes)

These are the runtime tripwires that detect partial encoder collapse.
The numerical values come from empirical inspection of the upstream
LeWM training: a healthy encoder satisfies
$\min_d \mathrm{Var} \geq 0.05$ and $\max_d |\mathbb E| \leq 5$ and
pairwise cosine $\leq 0.85$. Any of these failing is a strong
indicator that the SIGReg + prediction balance has tipped.

### TOL-010 ($2 \times 10^{-2}$ BF16 vs F32)

A full F32 training run and a BF16-mixed run with the same seed and
hyperparameters should converge to the same end-of-epoch loss within
2 %. This bounds the precision-island design's effectiveness: if the
two diverge more than 2 %, an F32 island is missing or a BF16 op
introduces too much drift.

### TOL-011 ($10^3$ pre-clip grad norm)

The natural pre-clip gradient norm in LeWM training is typically in
the $[10^{-3}, 1]$ range. A pre-clip norm of $10^3$ would indicate a
serious instability — an NaN propagating, a precision underflow, or
a data corruption. The training loop aborts in that case with a
diagnostic, rather than silently clipping and continuing into a
poisoned optimizer state.

## 3. Source pointers

| Topic | Source |
|-------|--------|
| Canonical table | `specs/glossary.md` §4 |
| Probes (collapse) | `crates/lewm-core/src/losses/collapse_probes.rs` |
| Grad-norm ceiling | `crates/lewm-train/src/step.rs` |
| Parity tolerances | `crates/lewm-core/tests/parity_*.rs` |

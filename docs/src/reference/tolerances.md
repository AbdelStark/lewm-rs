# Numerical tolerances

Mirror of [`specs/glossary.md` §4](https://github.com/AbdelStark/lewm-rs/blob/main/specs/glossary.md#4-numerical-tolerances-default-constants).
See [Tolerances and what they bound](../parity/tolerances.md) for the
rationale per tolerance.

| ID | Symbol | Default | Where used |
|----|--------|---------|------------|
| **TOL-001** | $\varepsilon_{\text{CLS,abs}}$ | $1.0\!\times\!10^{-4}$ | Parity: encoder CLS output, F32 |
| **TOL-002** | $\varepsilon_{\text{pred,abs}}$ | $1.0\!\times\!10^{-4}$ | Parity: predictor output, F32 |
| **TOL-003** | $\varepsilon_{\text{sigreg,abs}}$ | $1.0\!\times\!10^{-3}$ | Parity: SIGReg scalar, F32, identical RNG seed |
| **TOL-004** | $\varepsilon_{\text{sigreg,seed-free,rel}}$ | $5.0\!\times\!10^{-2}$ | Parity: SIGReg scalar, different RNG seed |
| **TOL-005** | $\varepsilon_{\text{loss,smoke,rel}}$ | $1.0\!\times\!10^{-2}$ | Local CPU smoke vs cloud BF16 step-100 loss |
| **TOL-006** | $\varepsilon_{\text{warm-start,delta,abs}}$ | $0.0$ | SO-100 warm-start latent-MSE must beat from-scratch by ≥ 0 |
| **TOL-007** | $\text{cls\_var\_floor}$ | $0.05$ | Collapse: per-dim CLS variance lower bound |
| **TOL-008** | $\text{cls\_mean\_abs\_ceiling}$ | $5.0$ | Collapse: mean absolute CLS upper bound |
| **TOL-009** | $\text{cls\_cosine\_pair\_ceiling}$ | $0.85$ | Collapse: mean pairwise CLS cosine upper bound |
| **TOL-010** | $\varepsilon_{\text{BF16}\to\text{F32,rel}}$ | $2.0\!\times\!10^{-2}$ | BF16 mixed run vs full F32 run, end-of-epoch loss |
| **TOL-011** | $\text{grad\_norm\_ceiling}$ | $1.0\!\times\!10^{3}$ | Pre-clip grad norm; above this we abort with a diagnostic |

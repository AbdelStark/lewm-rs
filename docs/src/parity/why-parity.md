# Why parity matters

> **Motivation.** A reproduction without numerical parity is a
> *re-interpretation*, not a reproduction. This section pins what
> parity means here, why it matters, and how it is enforced.
>
> **Position.** Top of [Part VI — Numerical parity](./why-parity.md).
>
> **What you should leave with.** A clear definition of activation-level
> parity, an understanding of what could go wrong without it, and the
> roadmap of the rest of Part VI.

## 1. The promise of parity

`lewm-rs` is a reimplementation of an existing PyTorch model. The
question naturally arises: *how do we know the Rust version computes
the same function as the PyTorch version?*

The answer is **activation-level parity testing**: we run the same
input through both implementations and compare the activations at
chosen depths to a numerical tolerance. If the implementations agree
at every checked point, we have strong evidence that they compute the
same function — much stronger than agreeing on a final loss or final
output alone, because the chance of a quietly-wrong implementation
producing the same activations *at every layer* is vanishingly small.

The contract is: **L∞ distance < 10⁻⁴ on every checked activation
(F32), for the locked PushT reference checkpoint.**

## 2. What "the locked reference" means

The reference is the published checkpoint
[`quentinll/lewm-pusht@22b330c`](https://huggingface.co/quentinll/lewm-pusht/tree/22b330c).
The commit hash `22b330c` is pinned in
`tests/fixtures/reference_model.meta.json`. Any update to the
reference must update the hash *and* re-run the parity dumps.

We do not require parity against the upstream PyTorch *source*
generally — only against this specific checkpoint. The reason: floats
are not deterministic across PyTorch versions, CUDA versions, and
GPU architectures, so "the same Python code on a different GPU"
already drifts at the 10⁻⁴ level. Pinning to a single checkpoint
fixes the parity contract.

## 3. The input fixture

Both implementations are run on the same input, the *parity fixture*:

| Tensor | Shape | Source |
|--------|------:|--------|
| `pixels`  | $(1, 3, 224, 224)$ | Episode 0, frame 100 of PushT, raw uint8 → f32 / 255 |
| `actions` | $(1, 3, 2)$        | Episode 0, frames 100..103, normalized |

The fixture is committed to the repo under `tests/fixtures/`. Its
SHA256 is pinned in `tests/fixtures/reference_model.meta.json`.

## 4. The reference dumps

For each parity test, the reference's per-stage activations are dumped
once and stored as Safetensors on
[`AbdelStark/lewm-rs-parity-dumps`](https://huggingface.co/datasets/AbdelStark/lewm-rs-parity-dumps).
The dump is keyed by the fixture's hash and the reference checkpoint's
SHA256, so any change to either invalidates the cache.

Dumps include per-stage activations like:

- `encoder.cls.safetensors` — encoder output, CLS slice, $(1, 192)$.
- `encoder.all.safetensors` — encoder output, all tokens, $(1, 257, 192)$.
- `action_encoder.safetensors` — action embedding, $(1, T, 192)$.
- `predictor.safetensors` — predictor output, $(1, T, 192)$.
- `pred_proj.safetensors` — pred-proj output, $(1, T, 192)$.
- `sigreg_scalar.json` — SIGReg scalar value plus its RNG seed.

The dumps are produced by
`python/convert_reference.py dump --component <name>`.

## 5. What could go wrong without parity

Without parity tests, the following bugs could ship unnoticed:

- **Wrong LayerNorm $\varepsilon$.** The default PyTorch $\varepsilon =
  10^{-5}$ produces 10⁻³ drift versus upstream's $10^{-12}$. Without
  parity tests, this only shows up as a 5–10 % drop in PushT success
  rate after 5 hours of A10G training.
- **Tanh-approx vs erf GELU.** Off by ~10⁻⁴ per activation, drift
  ~10⁻³ over 12 blocks. Catastrophic at parity, invisible at any
  scalar metric.
- **AdaLN-zero modulation ordering.** If the modulation tensor is
  split as `[γ, β, α, γ, β, α]` instead of the upstream
  `[γ₁, β₁, α₁, γ₂, β₂, α₂]`, the predictor becomes a different
  function. Loss will eventually go down, but to a different floor.
- **Causal mask diagonal off by one.** Subtly leaks future information
  to the predictor, which gets a slightly easier loss with worse
  generalisation.

Every one of the above has been caught by the parity tests during
lewm-rs development. They are the most cost-effective bug catcher in
the project.

## 6. The 10 tests

The parity harness runs 10 tests, listed exhaustively in
[The 10-test parity harness](./tests.md). Each test reports L∞ and
RMSE; the test passes if L∞ is below its tolerance.

Status: **10 / 10 PASS** against `quentinll/lewm-pusht@22b330c`.

## 7. Where parity sits in CI

The `parity` GitHub Actions workflow runs on every push:

- Downloads the dumps from
  `AbdelStark/lewm-rs-parity-dumps` (requires `HF_TOKEN`).
- If no token: runs shape-only checks (model loads, forward runs, shape
  matches dump).
- If token: runs full numerical checks.

The shape-only fallback ensures that a contributor without an HF token
still gets *some* coverage; the numerical contract is enforced for
maintainers.

## 8. Roadmap of Part VI

- **[The 10-test parity harness](./tests.md)** — the exhaustive list
  of tests with their tolerances.
- **[Tolerances and what they bound](./tolerances.md)** — TOL-001..010
  and the rationale for each.
- **[Implementation gotchas](./gotchas.md)** — the four implementation
  details that broke parity until they were fixed.

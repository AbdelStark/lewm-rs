# Project status

A single page summarising what is verified, what is in flight, and what is
deferred, as of the most recent reproducible state. The authoritative
source is the [release checklist] in the repository.

## Numerical parity vs. PyTorch reference

Reference checkpoint: [`quentinll/lewm-pusht@22b330c`].
All 10 activation-level parity tests are pinned by [RFC 0008] and gated in
CI under the `parity` workflow when `HF_TOKEN` is present.

| Component                       | Tolerance        | Status |
|---------------------------------|------------------|--------|
| Encoder — CLS token             | $L_\infty < 10^{-4}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |
| Encoder — all patch tokens      | $L_\infty < 10^{-4}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |
| Action encoder output           | $L_\infty < 10^{-4}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |
| Predictor output (all $T$)      | $L_\infty < 10^{-4}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |
| Pred-proj MLP output            | $L_\infty < 10^{-4}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |
| SIGReg loss scalar (seeded)     | $\lvert\Delta\rvert < 10^{-3}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |
| SIGReg loss scalar (seed-free)  | rel. $< 5\!\times\!10^{-2}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |
| Encoder (BF16 mixed)            | rel. $< 2\!\times\!10^{-2}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |
| Predictor (BF16 mixed)          | rel. $< 2\!\times\!10^{-2}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |
| End-to-end forward (10-step)    | $L_\infty < 10^{-4}$ | <span class="lewm-badge lewm-badge--done">PASS</span> |

Activation dumps are stored at
[`AbdelStark/lewm-rs-parity-dumps`](https://huggingface.co/datasets/AbdelStark/lewm-rs-parity-dumps).
The CI workflow downloads them automatically when an HF token is available
and falls back to a shape-only check otherwise.

## Training

### PushT — `pusht-minimal-lewm` mode

| Metric | Value |
|--------|-------|
| Steps completed | 50 000 / 50 000 |
| Wall time | 318 min |
| Hardware | A10G-large (HF Jobs) |
| Initial loss | 0.4912 |
| Final loss | 3.17 × 10⁻⁶ |
| Loss ratio | $1.55 \times 10^5$-fold reduction |
| Gradient explosions | 0 |
| Seed | 0 |
| Config hash | `438eb30f4bb0` |

Artifacts: [`abdelstark/lewm-rs-pusht`](https://huggingface.co/abdelstark/lewm-rs-pusht).
Full report: [`reports/pusht_training.md`].

### SO-100 — pick-and-place

| Metric | Value |
|--------|-------|
| Steps completed | 5 000 / 5 000 |
| Wall time | 864 s |
| Hardware | A10G-large (HF Jobs) |
| Initial loss | 0.5002 |
| Final loss | 9.56 × 10⁻⁵ |
| Gradient explosions | 0 |
| Seed | 0 |

Artifacts: [`abdelstark/lewm-rs-so100`](https://huggingface.co/abdelstark/lewm-rs-so100).
Full report: [`reports/so100_training.md`].

## Inference

| Variant | Median latency / episode | Hardware | Build |
|---------|--------------------------|----------|-------|
| Tract CPU runner (`lewm-infer`) | 4.08 s (p50), 4.13 s (p95) | Apple M-series | release |
| Burn `NdArray` CPU runner | benchmark pending | – | release |
| Burn CUDA runner | benchmark pending | A10G | release |

CEM configuration: 5 iterations × 1024 candidates, $H = 3$ history steps,
action dim $= 10$ (frameskip-packed PushT actions).

## Eval

| Item | Status |
|------|--------|
| PushT planning success rate | <span class="lewm-badge lewm-badge--partial">Eval pending</span> |
| SO-100 latent-MSE Spearman   | <span class="lewm-badge lewm-badge--partial">Eval pending</span> |
| SO-100 warm-start ablation   | <span class="lewm-badge lewm-badge--partial">Pending</span> |

## Deployment surface

- **Hugging Face Hub**: model + ONNX artifacts at
  [`abdelstark/lewm-rs-pusht`](https://huggingface.co/abdelstark/lewm-rs-pusht)
  and [`abdelstark/lewm-rs-so100`](https://huggingface.co/abdelstark/lewm-rs-so100).
- **Demo Space**: live Gradio CEM planning at
  [`abdelstark/lewm-rs-demo`](https://huggingface.co/spaces/abdelstark/lewm-rs-demo).
- **Container**: `ghcr.io/abdelstark/lewm-rs:latest`
  (training + checkpoint upload, see [`Dockerfile`](https://github.com/AbdelStark/lewm-rs/blob/main/Dockerfile)).

## Cost ledger

Total confirmed spend on training compute: **\$11.70 USD** at \$1.50 /hr for
A10G-large — \$3.75 for SO-100 attempts and pre-training, \$7.95 for the
50 k-step PushT full run. Cap: \$200. See
[`reports/cost.md`](https://github.com/AbdelStark/lewm-rs/blob/main/reports/cost.md).

## Known gaps

The following items are tracked in [`ROADMAP.md`] and are visible on this
site as <span class="lewm-badge lewm-badge--partial">Partial</span> or
<span class="lewm-badge lewm-badge--todo">Planned</span> badges where
relevant:

1. **`lewm_core::Jepa` end-to-end training**: the current 50 k-step PushT
   run uses `PushtFullLewmCore` (a simplified core); the full Burn ViT
   (`lewm_core::Jepa`, 303 parameter tensors / 18.04 M parameters,
   parity-validated) is not yet wired into the training loop. The ONNX
   export therefore uses converted PyTorch reference weights, not a
   natively Rust-trained ViT checkpoint.
2. **Planning eval**: CEM planning success rate on PushT and latent-MSE /
   Spearman on SO-100.
3. **Warm-start ablation**: training SO-100 from PushT weights vs. random.
4. **Release-build CPU benchmark** on standard hardware.
5. **Multi-camera SO-100 inputs** (RFC 0012 §4.3).
6. **Quantised Tract inference** (INT8 ONNX).

[release checklist]: https://github.com/AbdelStark/lewm-rs/blob/main/reports/release_checklist.md
[`quentinll/lewm-pusht@22b330c`]: https://huggingface.co/quentinll/lewm-pusht/tree/22b330c
[RFC 0008]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0008-reference-parity-testing.md
[`reports/pusht_training.md`]: https://github.com/AbdelStark/lewm-rs/blob/main/reports/pusht_training.md
[`reports/so100_training.md`]: https://github.com/AbdelStark/lewm-rs/blob/main/reports/so100_training.md
[`ROADMAP.md`]: https://github.com/AbdelStark/lewm-rs/blob/main/ROADMAP.md

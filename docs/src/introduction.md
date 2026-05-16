# Introduction

`lewm-rs` is a **pure-Rust reproduction and extension of LeWorldModel** (LeWM;
Maes, Le Lidec, Scieur, Balestriero, and LeCun, 2026 — arXiv [2502.16560]).
LeWM is a *Joint-Embedding Predictive Architecture* (JEPA) world model for
robotic manipulation: it learns a compact latent representation of visual
observations, then trains an autoregressive predictor to roll out latent
states one step at a time, conditioned on an action history. Planning uses
the *Cross-Entropy Method* (CEM) over the predictor's latent rollout, with
the cost given by the latent distance to a goal embedding.

This documentation site is the **technical companion** to the codebase. It is
written for readers who already know transformers and self-supervised learning
and want to understand:

1. **Why** LeWM is built the way it is — what design pressures lead to a
   two-loss objective (prediction MSE + SIGReg), an AdaLN-zero predictor, an
   end-to-end gradient (no EMA, no stop-gradient), and an end-to-end CEM
   pipeline.
2. **How** every module is specified down to the level of byte-exact
   reproduction — patch embedding, position embeddings, attention, AdaLN
   modulation, the SIGReg sketch, the trapezoid integration, the AdamW
   decay groups, the cosine schedule, the determinism contract.
3. **What** has been built in Rust, and what numerical parity is established
   against the upstream PyTorch reference (`quentinll/lewm-pusht`).
4. **Where** the system currently stands — training results, inference
   latency, cost, and the remaining engineering gaps.

The site is organized as ten parts mirroring the conceptual stack:

| Part | Topic | Audience |
|------|-------|----------|
| I    | Foundational concepts (JEPA, SIGReg, ViTs, CEM) | New readers, classroom |
| II   | Architecture, layer by layer | Implementers, parity checkers |
| III  | Training pipeline, optimizer, determinism | Practitioners |
| IV   | Planning and evaluation | Robotics researchers |
| V    | Inference, ONNX export, Tract CPU runner | Deployment engineers |
| VI   | Numerical parity contracts and gotchas | Anyone porting JEPA to a new backend |
| VII  | Results, costs, discussion | Reviewers, replicators |
| VIII | Crate-by-crate API tour | Contributors |
| IX   | Reproducing the results | Replicators |
| X    | Reference (glossary, tolerances, RFCs, bibliography) | All |

## How the project is organized

```text
lewm-rs/
├── crates/        Rust workspace (lewm-core, lewm-data, lewm-train,
│                   lewm-plan, lewm-infer, lewm-hub, lewm-telemetry, lewm-gpu)
├── specs/         Normative specs: 18 RFCs + 2 ADRs + glossary
├── docs/          THIS DOCUMENTATION SITE (mdBook)
├── paper/         Paper-style writeup (paper/lewm-rs.md)
├── reports/       Training, inference, and cost reports
├── configs/       Training and eval TOML configs (pusht, so100)
├── python/        Edge adapters: ONNX export, decoding, stats, upload, eval
├── jobs/          Hugging Face Jobs launch YAML
├── tests/         Workspace-level integration tests and fixtures
└── conformance/   Conformance test suite
```

`docs/` (you are reading it) is the **explanatory layer** on top of `specs/`
and the source code. The specs are normative — every claim made on this site
is grounded in either an accepted RFC, an ADR, the source, or a published
report. Cross-references to those sources are inline throughout.

## Reading paths

Three suggested entry points:

- **The 30-minute tour** —
  [Foundations of JEPA](./concepts/jepa.md) →
  [Architecture at a glance](./architecture/overview.md) →
  [Training pipeline](./training/pipeline.md) →
  [Results](./results/pusht.md).
- **The implementer's path** —
  [Architecture](./architecture/overview.md) →
  [Shape contracts](./architecture/shape-contracts.md) →
  [Numerical parity](./parity/why-parity.md) → the relevant
  [crate page](./crates/workspace.md).
- **The reproducer's path** —
  [Quickstart](./reproducing/quickstart.md) →
  [PushT training](./reproducing/training-pusht.md) →
  [Inference](./reproducing/inference.md).

## What this project is, and is not

**This project is**:

- A faithful, parity-verified reimplementation of LeWM in safe Rust on top of
  the [Burn](https://github.com/tracel-ai/burn) framework.
- A deployment story: a single, statically-linked binary for CPU inference
  via the [Tract](https://github.com/sonos/tract) runtime, with no Python
  runtime required.
- A worked example of how to implement a non-trivial modern ML architecture
  in Rust with full numerical-parity contracts, hardened CI, and a
  reproducible training pipeline.

**This project is not**:

- A new algorithmic contribution. The architecture, losses, and training
  procedure follow Maes et al. (2026) without modification.
- A from-scratch ML framework. We use Burn v0.20.1 for differentiable
  compute and Tract for inference.
- A production robotics stack. The SO-100 pipeline is a research-grade
  reproduction; real-robot deployment is out of scope.

## Status at a glance

<span class="lewm-badge lewm-badge--done">Parity</span>
All 10 activation-level parity tests pass against the locked PushT reference
checkpoint, with $L_\infty < 10^{-4}$ on encoder, action encoder, predictor,
and pred-proj outputs, and $|\Delta| < 10^{-3}$ on the SIGReg scalar.

<span class="lewm-badge lewm-badge--done">PushT training</span>
50 000 steps on A10G-large, 318 min, loss 0.491 → 3.17 × 10⁻⁶, zero gradient
explosions.

<span class="lewm-badge lewm-badge--done">SO-100 training</span>
5 000 steps on A10G-large, 864 s, loss 0.50 → 9.56 × 10⁻⁵, zero gradient
explosions.

<span class="lewm-badge lewm-badge--done">CPU inference</span>
ONNX export + Tract runner working end-to-end at 4.08 s/episode (p50, release
build, Apple M-series, 5 × 1024 CEM candidates).

<span class="lewm-badge lewm-badge--partial">Planning success rate</span>
Eval pending; target is ≥ 87 % on the 50-episode PushT test set (matching
upstream).

<span class="lewm-badge lewm-badge--partial">Warm-start ablation</span>
SO-100 latent-MSE comparison from-scratch vs. PushT warm-start pending.

[2502.16560]: https://arxiv.org/abs/2502.16560

# Summary

[Introduction](./introduction.md)
[How to read these docs](./reading-guide.md)
[Project status](./status.md)

---

# Part I — Concepts

- [Foundations of JEPA](./concepts/jepa.md)
- [LeWorldModel: the specialization](./concepts/lewm.md)
- [World models for robotics](./concepts/world-models.md)
- [SIGReg: sketch-isotropic Gaussian regularization](./concepts/sigreg.md)
- [Vision Transformers in latent prediction](./concepts/vit.md)
- [AdaLN-zero conditioning](./concepts/adaln.md)
- [The Cross-Entropy Method](./concepts/cem.md)
- [Why latent prediction works](./concepts/why-latents.md)

# Part II — Architecture

- [Architecture at a glance](./architecture/overview.md)
- [The ViT-Tiny encoder](./architecture/encoder.md)
- [The autoregressive predictor](./architecture/predictor.md)
- [The action encoder](./architecture/action-encoder.md)
- [Projector and pred-proj MLPs](./architecture/projector.md)
- [The Jepa wrapper and rollout](./architecture/jepa-wrapper.md)
- [Shape contracts and tensor flow](./architecture/shape-contracts.md)
- [Parameter inventory](./architecture/parameter-inventory.md)

# Part III — The training paradigm

- [Training pipeline](./training/pipeline.md)
- [Data plane and window sampling](./training/data.md)
- [Loss functions: prediction + SIGReg](./training/losses.md)
- [Gradient flow and end-to-end stability](./training/gradient-flow.md)
- [AdamW, decay groups, and the schedule](./training/optimizer.md)
- [Mixed precision and F32 islands](./training/mixed-precision.md)
- [The training state machine](./training/state-machine.md)
- [Determinism and reproducibility](./training/determinism.md)
- [Checkpointing and crash-resume](./training/checkpoints.md)
- [Observability and OTLP telemetry](./training/observability.md)

# Part IV — Planning and evaluation

- [Planning with CEM](./planning/cem.md)
- [PushT evaluation protocol](./planning/pusht-eval.md)
- [SO-100 evaluation protocol](./planning/so100-eval.md)
- [Warm-start ablation](./planning/warm-start.md)

# Part V — Inference and deployment

- [ONNX export pipeline](./inference/onnx-export.md)
- [Tract CPU runner](./inference/tract.md)
- [Burn NdArray and CUDA runners](./inference/burn-runners.md)
- [Latency benchmarks](./inference/benchmark.md)
- [The Hugging Face demo Space](./inference/demo.md)

# Part VI — Numerical parity

- [Why parity matters](./parity/why-parity.md)
- [The 10-test parity harness](./parity/tests.md)
- [Tolerances and what they bound](./parity/tolerances.md)
- [Implementation gotchas](./parity/gotchas.md)

# Part VII — Results

- [PushT 50k-step training](./results/pusht.md)
- [SO-100 5k-step training](./results/so100.md)
- [Cost ledger](./results/cost.md)
- [Discussion and limitations](./results/discussion.md)

# Part VIII — Workspace and crates

- [Workspace map](./crates/workspace.md)
- [`lewm-core`](./crates/lewm-core.md)
- [`lewm-data`](./crates/lewm-data.md)
- [`lewm-train`](./crates/lewm-train.md)
- [`lewm-plan`](./crates/lewm-plan.md)
- [`lewm-infer`](./crates/lewm-infer.md)
- [`lewm-telemetry`](./crates/lewm-telemetry.md)
- [`lewm-hub`](./crates/lewm-hub.md)
- [`lewm-gpu`](./crates/lewm-gpu.md)

# Part IX — Reproducing the results

- [Quickstart](./reproducing/quickstart.md)
- [Reproducing PushT training](./reproducing/training-pusht.md)
- [Reproducing SO-100 training](./reproducing/training-so100.md)
- [Running CPU inference](./reproducing/inference.md)
- [Docker and HF Jobs](./reproducing/docker.md)
- [Local quality gate](./reproducing/quality-gate.md)

# Part X — Reference

- [Glossary](./reference/glossary.md)
- [Symbol conventions](./reference/notation.md)
- [Numerical tolerances](./reference/tolerances.md)
- [RFC index](./reference/rfcs.md)
- [ADR index](./reference/adrs.md)
- [Bibliography](./reference/bibliography.md)
- [How to cite](./reference/citation.md)

---

# Appendix

- [Contributing](./community/contributing.md)
- [Code of conduct](./community/code-of-conduct.md)
- [Security policy](./community/security.md)
- [License](./community/license.md)
- [Acknowledgments](./community/acknowledgments.md)

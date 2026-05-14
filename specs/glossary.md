# Glossary — `lewm-rs`

**Status:** Accepted · **Version:** 1.0.0 · **Last updated:** 2026-05-12

This glossary is normative. Every term that appears in any spec document with a defined technical meaning **MUST** be listed here. Code, RFCs, and ADRs use these terms verbatim.

---

## 1. Domain terms

### 1.1 JEPA-family

**JEPA** — *Joint-Embedding Predictive Architecture.* A self-supervised representation learning framework where an encoder maps inputs to embeddings and a predictor maps embeddings to embeddings under a context (e.g., an action, a positional offset). Introduced by LeCun, refined in I-JEPA, V-JEPA, and now LeWorldModel.

**LeWM** — *LeWorldModel*, Maes et al., 2026. The specific JEPA variant this project reproduces. Distinguished by:
1. End-to-end training (no EMA target network, no stop-gradient on the encoder).
2. A single regularizer (SIGReg) replacing the multi-term variance-invariance-covariance objectives used in prior JEPAs.
3. Two loss terms only: prediction MSE and SIGReg.

**SIGReg** — *Sketch Isotropic Gaussian Regularizer.* The regularizer of LeWM. Projects latents onto 1024 random unit-norm directions and applies the Epps–Pulley test against the standard Gaussian characteristic function at 17 frequency knots in `[0, 3]`. See [RFC 0003](rfcs/0003-sigreg-and-loss-functions.md) for the exact algorithm and numerical contract.

**AdaLN-zero** — Adaptive Layer Normalization where the conditioning network is zero-initialized, so the conditioned block begins as an identity function. Used in the LeWM predictor. The final adaLN linear weight is the all-zero matrix at init.

**CLS token** — A learned vector prepended to the patch token sequence in a Vision Transformer. The encoder's representation of the image is taken to be the CLS token at the output of the final block, optionally post-LayerNorm and post-projector. In `lewm-rs`, the term **CLS output** specifically refers to `vit.forward(pixels).last_hidden_state[:, 0]` before any projection — matching the HF `transformers` semantics.

**Rollout** — Autoregressive evaluation of the predictor with a sliding history window of `history_size` (default 3). Given a start embedding sequence of length `H` and an action sequence of length `T-H`, produce embeddings of length `T` by predicting one step at a time and shifting the window.

**CEM** — *Cross Entropy Method.* The action-search procedure used at planning time. See [RFC 0006](rfcs/0006-planning-and-evaluation.md). Hyperparameters in PRD §9.1.

### 1.2 Datasets

**PushT** — `quentinll/lewm-pusht`. A planar block-pushing dataset with 224×224 RGB images and 2-D actions. The paper's primary benchmark.

**SO-100** — `lerobot/svla_so100_pickplace`. A pick-and-place dataset recorded on the SO-100 robot with two camera views, 6-D actions, native 480×640 RGB at 30 Hz.

**LeRobot v2.1** — The on-disk format for SO-100. Parquet for tabular fields (actions, joint positions, timestamps), MP4 per-episode for video streams.

### 1.3 Framework terms

**Burn** — The Rust deep learning framework powering training. Version pinned in [RFC 0001 §3.2](rfcs/0001-project-foundation-and-build-system.md). Notable types used throughout the codebase:

- `burn::tensor::Tensor<B, D>` — a `D`-rank tensor over backend `B`.
- `burn::module::Module` — derive macro for modules; provides `forward`, `valid` (eval mode), and parameter iteration.
- `burn::record::Recorder` — checkpoint serializer/deserializer.

**Backend** — Burn type-parameter. `B: Backend`. Implementations used by `lewm-rs`:
- `burn_cuda::Cuda<f32>` and `Cuda<bf16>` — primary training backend.
- `burn_ndarray::NdArray<f32>` — CPU testing and local smoke.
- `burn_autodiff::Autodiff<B>` — wraps any backend to add reverse-mode AD.

**Tract** — `sonos/tract`. Pure-Rust CPU inference engine. Loads ONNX or NNEF. Used in `lewm-infer`. Version pinned in [RFC 0007 §3](rfcs/0007-tract-inference-and-onnx-export.md).

**ONNX** — *Open Neural Network Exchange.* The serialization format used for export from Burn to Tract. Opset target is fixed in [RFC 0007 §4](rfcs/0007-tract-inference-and-onnx-export.md).

**NNEF** — *Neural Network Exchange Format.* Tract-native fallback format. Used only if ONNX export of any required op fails.

### 1.4 Hugging Face

**HF Hub** — `huggingface.co/`. Hosts model, dataset, and Space repositories.

**HF Jobs** — Compute service billed per minute. The only training compute used by this project.

**Trackio** — HF-native experiment tracking, locally and via Spaces. Primary metric dashboard.

**Space** — A web app hosted on HF, used here for the public CPU planning demo.

**ml-intern** — An Apache-2.0 HF agent CLI used for dataset scouting, sweeps, and build-fix loops. Tightly leashed; see [RFC 0016 §6](rfcs/0016-security-and-supply-chain.md).

---

## 2. Workspace terms

**Crate** — A Rust compilation unit. The workspace contains 7 crates listed in [RFC 0001 §4](rfcs/0001-project-foundation-and-build-system.md).

**Workspace** — The Cargo workspace defined by the root `Cargo.toml`. All crates share a single `Cargo.lock` and a single target dir.

**Feature** — A Cargo conditional compilation flag. Features used in `lewm-rs` are enumerated in [RFC 0001 §5](rfcs/0001-project-foundation-and-build-system.md).

**Profile** — A Cargo build profile. `dev`, `release`, `bench`, and `release-lto` are defined. See [RFC 0001 §6](rfcs/0001-project-foundation-and-build-system.md).

**Binary** — A `cargo run --bin <name>` entry point. The deliverable binaries are `lewm-train`, `lewm-eval`, `lewm-infer`. See [RFC 0001 §4.3](rfcs/0001-project-foundation-and-build-system.md).

---

## 3. Pipeline terms

**Phase** — A high-level stage of the project as defined in PRD §8 (e.g., Phase 0 Bootstrap, Phase 1 Parity). Phase boundaries are deliverable gates.

**Tier** — A category of HF Jobs launch defined in PRD §6.5 (T1 SMOKE, T2 SHORT, T3 FULL).

**State** (training pipeline) — A node in the state machine defined in PRD §5.5: `INIT → PARITY_CHECK → SMOKE → WARMUP → STEADY → COOLDOWN → EVAL → UPLOAD → DONE`.

**Stage** (local validation) — A pre-cloud check defined in PRD §6.4: `L0 unit tests → L1 parity probe → L2 CPU smoke train`. All three **MUST** pass before any T1/T2/T3.

**Step** — One optimizer update. With grad-accum, one step covers `grad_accum_steps` micro-batches.

**Epoch** — One full pass over the training dataset.

**Micro-batch / batch / effective batch** — In a grad-accum setup of factor `K`: a *micro-batch* of size `M` is one forward+backward; `K` micro-batches form an *effective batch* of size `K·M`; the optimizer steps once per effective batch.

**Window** (data) — A consecutive sequence of `T = horizon` frames sampled from one episode, with optional `history_size`-frame prefix used as warm-up context for the predictor.

**Collate** — The function that stacks per-sample tensors into a batch tensor. See [RFC 0004 §6](rfcs/0004-data-pipeline.md).

---

## 4. Numerical tolerances (default constants)

These are the binding tolerance defaults. Override is allowed only via a published ADR.

| ID | Symbol | Default | Where used |
|----|--------|---------|------------|
| TOL-001 | `ε_CLS_abs` | `1.0e-4` | Parity: encoder CLS output, F32 |
| TOL-002 | `ε_pred_abs` | `1.0e-4` | Parity: predictor output, F32 |
| TOL-003 | `ε_sigreg_abs` | `1.0e-3` | Parity: SIGReg scalar, F32, identical RNG seed |
| TOL-004 | `ε_sigreg_seedfree_rel` | `5.0e-2` | Parity: SIGReg scalar, different RNG seed (sketch resampled) |
| TOL-005 | `ε_loss_smoke_rel` | `1.0e-2` | Local CPU smoke vs cloud smoke step-100 loss |
| TOL-006 | `ε_warm_start_delta_abs` | `0.0` | SO-100 warm-start latent-MSE must beat from-scratch by ≥ 0 |
| TOL-007 | `cls_var_floor` | `0.05` | Collapse detector: per-dim CLS variance lower bound |
| TOL-008 | `cls_mean_abs_ceiling` | `5.0` | Collapse detector: mean absolute CLS upper bound |
| TOL-009 | `cls_cosine_pair_ceiling` | `0.85` | Collapse detector: mean pairwise CLS cosine upper bound |
| TOL-010 | `bf16_to_f32_max_rel` | `2.0e-2` | BF16 mixed run vs full F32 run, end-of-epoch loss |
| TOL-011 | `grad_norm_ceiling` | `1.0e3` | Pre-clip grad norm; above this we abort with a diagnostic |

---

## 5. Acronyms (alphabetic)

- **ADR** — Architectural Decision Record. Single immutable decision document.
- **AdaLN** — Adaptive Layer Normalization.
- **BF16** — IEEE 16-bit brain float (8 exponent, 7 mantissa).
- **CEM** — Cross Entropy Method.
- **CLI** — Command-line interface.
- **CLS** — Class token (ViT).
- **EMA** — Exponential moving average.
- **F32** — IEEE 754 binary32 single-precision float.
- **FLOPs** — Floating-point operations.
- **FR** — Functional requirement.
- **GFM** — GitHub Flavored Markdown.
- **HDF5** — Hierarchical Data Format v5.
- **HF** — Hugging Face.
- **INV** — Invariant.
- **JEPA** — Joint-Embedding Predictive Architecture.
- **MLP** — Multi-layer perceptron.
- **MPK** — MessagePack-serialized Burn record file extension.
- **MSE** — Mean squared error.
- **NFR** — Non-functional requirement.
- **NNEF** — Neural Network Exchange Format.
- **ONNX** — Open Neural Network Exchange.
- **OTLP** — OpenTelemetry Protocol.
- **PRD** — Product Requirements Document (`../PRD.md`).
- **RFC** — Request for Comments.
- **RNG** — Random number generator.
- **RSS** — Resident set size.
- **SDPA** — Scaled Dot-Product Attention.
- **SemVer** — Semantic Versioning 2.0.0.
- **SIGReg** — Sketch Isotropic Gaussian Regularizer.
- **TFLOPS** — Tera-FLOPs per second.
- **TLDR** — A code structure CLI used in dev workflow (unrelated to ML).
- **TOML** — Tom's Obvious Minimal Language (config format).
- **TST** — Test specification.
- **ViT** — Vision Transformer.

---

## 6. Symbol conventions

In equations and code:

- `B` — batch dimension (with grad-accum semantics distinguished where it matters).
- `T` — temporal dimension (frames in a window).
- `H, W` — image height and width in pixels.
- `C` — channel dimension; `3` for RGB.
- `D` — embedding dimension; `192` for the locked PushT ViT-tiny reference.
- `K` — number of random projections in SIGReg; `1024`.
- `J` — number of frequency knots in SIGReg; `17`.
- `t` — within-window time index; `t ∈ [0, T)`.
- `λ` (`lambda`) — SIGReg loss weight; `1.0` default.
- `A` — raw action dimension; `2` for PushT, `6` for SO-100.
- `A_p` — packed action dimension after frameskip; `10` for PushT.
- `E_a` — action embedding dimension after `Embedder`; `192` for the locked PushT reference.
- `H_v` — encoder hidden dim; `192` for the locked PushT reference.
- `L_pred` — prediction loss (MSE).
- `L_sigreg` — SIGReg loss.
- `L` — total loss; `L_pred + λ · L_sigreg`.

In Rust types:

- `B: Backend` — Burn backend type parameter.
- `B::Device` — backend's device handle.
- `B::FloatElem` — backend's default float element (F32 or BF16 depending on backend instantiation).

---

## 7. File and artifact name conventions

| Pattern | Meaning |
|---------|---------|
| `step_{N}.mpk` | Burn record snapshot at training step `N`. Contains model + optimizer state. |
| `step_{N}.safetensors` | Safetensors mirror of the model parameters at step `N`. |
| `step_{N}.json` | Sidecar metadata: config hash, git SHA, RNG state, wall-time. |
| `step_{N}.parity.json` | Per-checkpoint parity probe results (fixed input). |
| `collapse_suspected_{step}.json` | Written by the collapse detector when any of TOL-007/008/009 trips. |
| `run_id.txt` | A single line; the canonical run identifier (`{date}-{shortsha}-{slot}`). |
| `cost.md` | Per-run cost ledger; appended on every job termination. |

---

## 8. Out-of-glossary terms

If a term is used in any spec document without appearing here, that is a defect: open a PR adding it to this glossary before merging the dependent change. CI enforces this via `scripts/check_specs.py --check-glossary`.

*End of glossary.*

# Glossary

> This page is the documentation-site companion to the **normative**
> glossary in [`specs/glossary.md`](https://github.com/AbdelStark/lewm-rs/blob/main/specs/glossary.md).
> The spec is authoritative; this page mirrors the most useful entries
> for in-docs cross-referencing.

## Domain terms

**JEPA** — *Joint-Embedding Predictive Architecture.* A
self-supervised learning framework where an encoder maps inputs to
embeddings and a predictor maps embeddings to embeddings under a
context. See [JEPA concepts](../concepts/jepa.md).

**LeWM** — *LeWorldModel*, Maes et al., 2026. The specific JEPA
variant `lewm-rs` reproduces. End-to-end training, no EMA, no
stop-gradient, one regulariser (SIGReg). See
[LeWM specialization](../concepts/lewm.md).

**SIGReg** — *Sketch Isotropic Gaussian Regularizer.* The regulariser
of LeWM. Projects latents onto 1024 random unit-norm directions and
applies an Epps–Pulley test at 17 frequency knots in $[0, 3]$. See
[SIGReg](../concepts/sigreg.md).

**AdaLN-zero** — *Adaptive Layer Normalization* with zero-initialised
modulation head, so each block is the identity at init. See
[AdaLN-zero](../concepts/adaln.md).

**CLS token** — The learned prefix token of a ViT. The encoder's
representation of an image is the CLS row at the output of the final
LayerNorm. See [ViT encoder](../architecture/encoder.md).

**Rollout** — Autoregressive evaluation of the predictor with a
sliding history window. See [Jepa wrapper §4](../architecture/jepa-wrapper.md).

**CEM** — *Cross-Entropy Method.* Sample-based, derivative-free
planner. See [CEM concepts](../concepts/cem.md) and
[Planning](../planning/cem.md).

## Datasets

**PushT** — `quentinll/lewm-pusht`. Planar block-pushing task,
$224 \times 224$ RGB, 2-D actions.

**SO-100** — `lerobot/svla_so100_pickplace`. SO-100 6-DOF arm
pick-and-place, native $480 \times 640$ at 30 Hz; lewm-rs uses the
re-encoded `abdelstark/so100-pickplace-lewm-ready` at $224 \times 224$
/ 10 fps.

**LeRobot v2.1** — The on-disk format for the raw SO-100 dataset:
Parquet for tabular fields, MP4 / AV1 per episode for video.

## Framework terms

**Burn** — The Rust deep learning framework. Pinned at `= 0.20.1`.

**Backend** — Burn type parameter `B: Backend`. Common impls used
by `lewm-rs`: `burn_cuda::Cuda<f32>`, `burn_ndarray::NdArray<f32>`,
`burn_autodiff::Autodiff<B>`.

**Tract** — Sonos's pure-Rust ONNX/NNEF inference engine. Pinned at
`= 0.22.1`.

**ONNX** — *Open Neural Network Exchange.* The serialisation format
used to export from Burn / PyTorch to Tract.

**NNEF** — *Neural Network Exchange Format.* Tract-native format;
fallback if ONNX export of a required op fails.

## Workspace terms

**Crate** — A Rust compilation unit. The workspace contains eight
crates; see [Workspace map](../crates/workspace.md).

**Workspace** — The Cargo workspace defined by the root `Cargo.toml`.

**Phase** — A PRD-level stage (Phase 0 Bootstrap, Phase 1 Parity, …).

**Tier** — A category of HF Jobs launch (T1 SMOKE, T2 SHORT, T3 FULL).

**State** — A node in the training state machine (`INIT → ... → DONE`).
See [State machine](../training/state-machine.md).

**Stage** — A pre-cloud local validation stage (L0 unit tests,
L1 parity probe, L2 CPU smoke train).

**Step** — One optimizer update.

**Micro-batch / batch / effective batch** — In a grad-accum setup of
factor $K$: a *micro-batch* of size $M$ is one forward+backward; $K$
micro-batches form an *effective batch* of size $K\cdot M$; the
optimizer steps once per effective batch.

**Window** — A consecutive sequence of $T$ frames sampled from one
episode.

**Collate** — The function that stacks per-sample tensors into a
batch tensor.

## Symbols

See [Symbol conventions](./notation.md) for the canonical math /
tensor-shape symbol table.

## Acronyms

Full list in [`specs/glossary.md` §5](https://github.com/AbdelStark/lewm-rs/blob/main/specs/glossary.md#5-acronyms-alphabetic):

ADR, AdaLN, BF16, CEM, CLI, CLS, EMA, F32, FLOPs, FR, GFM, HDF5, HF,
INV, JEPA, MLP, MPK, MSE, NFR, NNEF, ONNX, OTLP, PRD, RFC, RNG, RSS,
SDPA, SemVer, SIGReg, TFLOPS, TOL, TOML, TST, ViT.

## File and artifact naming

| Pattern | Meaning |
|---------|---------|
| `step_{N}.mpk` | Burn record snapshot at training step $N$. Model + optimizer state. |
| `step_{N}.safetensors` | Safetensors mirror of the model parameters at step $N$. |
| `step_{N}.json` | Sidecar metadata: config hash, git SHA, RNG state, wall-time. |
| `step_{N}.parity.json` | Per-checkpoint parity probe results (fixed input). |
| `collapse_suspected_{step}.json` | Collapse-probe diagnostic (TOL-007/008/009 trip). |
| `run_id.txt` | The canonical run identifier (`{date}-{shortsha}-{slot}`). |
| `cost.md` | The cost ledger; appended on every job termination. |

# lewm-rs

> A pure-Rust reproduction of LeWorldModel (Maes et al., 2026) — JEPA training, CEM planning, and CPU/GPU inference, numerically parity-verified against the PyTorch reference.

[![CI](https://img.shields.io/github/actions/workflow/status/AbdelStark/lewm-rs/ci.yml?style=for-the-badge&label=CI&logo=github&logoColor=white)](https://github.com/AbdelStark/lewm-rs/actions/workflows/ci.yml)
[![Spec Checks](https://img.shields.io/github/actions/workflow/status/AbdelStark/lewm-rs/specs.yml?style=for-the-badge&label=Spec+Checks)](https://github.com/AbdelStark/lewm-rs/actions/workflows/specs.yml)
[![Conformance](https://img.shields.io/github/actions/workflow/status/AbdelStark/lewm-rs/conformance.yml?style=for-the-badge&label=Conformance)](https://github.com/AbdelStark/lewm-rs/actions/workflows/conformance.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Rust 1.95](https://img.shields.io/badge/Rust-1.95_·_edition_2024-orange?style=for-the-badge&logo=rust&logoColor=white)](rust-toolchain.toml)
[![Burn 0.21.0](https://img.shields.io/badge/Burn-0.21.0-7c5cfc?style=for-the-badge)](https://github.com/tracel-ai/burn)
[![Tract 0.22.1](https://img.shields.io/badge/Tract-0.22.1-informational?style=for-the-badge)](https://github.com/sonos/tract)
[![Hub: PushT](https://img.shields.io/badge/%F0%9F%A4%97_Hub-PushT-yellow?style=for-the-badge)](https://huggingface.co/abdelstark/lewm-rs-pusht)
[![Hub: SO-100](https://img.shields.io/badge/%F0%9F%A4%97_Hub-SO--100-yellow?style=for-the-badge)](https://huggingface.co/abdelstark/lewm-rs-so100)
[![arXiv 2502.16560](https://img.shields.io/badge/arXiv-2502.16560-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.16560)

```
  TRAINING  ·  lewm-train (Burn)
  ─────────────────────────────────────────────────────────────────────────────
   HDF5 (PushT / SO-100)
    │  lewm-data loader
    │  Jepa<B>: encoder + predictor + projector
    │  loss = pred_loss + λ · SIGReg
    │  AdamW + cosine-LR + grad-clip
    ▼  checkpoint  (.safetensors + .mpk)
  ─────────────────────────────────────────────────────────────────────────────
                       │
          ┌────────────┴─────────────────────────────────┐
          │                                              │
  ONNX Export (python/export_onnx.py)        Burn runner (lewm-infer)
   encoder.onnx · predictor.onnx              NdArray (CPU) · CUDA (lewm-gpu)
          │                                              │
  Tract CPU runner (lewm-infer)                         │
   ONNX / NNEF · no Python required                     │
          │                                              │
          └────────────────────────┬─────────────────────┘
                                   │
                   CEM planner (lewm-plan / lewm-infer)
                    n_iter × n_cand
                                   │
                            action sequence
```

*Pipeline: HDF5 windows → `Jepa<B>` training → ONNX export → CPU/GPU runners → CEM planning. Each box is a crate or a binary; the dependency layering is enforced by `scripts/check_layers.py`.*

## TL;DR

- **Pure-Rust LeWM** across 8 Cargo crates on [Burn 0.21.0](https://github.com/tracel-ai/burn) + [Tract 0.22.1](https://github.com/sonos/tract); no Python at training or inference time.
- **Numerical parity** with the published `quentinll/lewm-pusht` reference: all 10 activation-level tests pass with L∞ < 1e-4 ([`reports/gpu_inference.md`](reports/gpu_inference.md)).
- **PushT** bounded trainer run completed on a single A10G-large in **318 min / 50k steps**, loss 0.4912 → 3.17e-06; artifacts at [`abdelstark/lewm-rs-pusht`](https://huggingface.co/abdelstark/lewm-rs-pusht). The full Burn/Jepa ONNX release checkpoint is still blocked on the approval-gated `train/pusht-full-burn-jepa-*` rerun ([`reports/pusht_onnx_export.md`](reports/pusht_onnx_export.md)). **SO-100** trained in **864 s / 5k steps**, loss 0.5002 → 9.56e-05; artifacts at [`abdelstark/lewm-rs-so100`](https://huggingface.co/abdelstark/lewm-rs-so100).
- **Tract CPU planning** at **4.08 s/episode (p50)** on Apple M3 release build, 5 CEM iterations × 1024 candidates ([`reports/inference.md`](reports/inference.md)).
- **One install**, four commands: `cargo build --release --workspace --locked`.

## Installation

```bash
git clone https://github.com/AbdelStark/lewm-rs.git
cd lewm-rs
rustup show active-toolchain                       # 1.95.0 pinned by rust-toolchain.toml
cargo build --release --workspace --locked
```

System dependencies (Linux: `apt install build-essential cmake pkg-config`; macOS: `brew install cmake`). `cmake` is required because `lewm-data` links `hdf5-metno` with `features = ["static"]`. No CUDA toolkit is needed unless you build `lewm-gpu` with `--features burn-cuda`.

| Component                       | Linux x86_64 | macOS arm64 (M-series) | NVIDIA CUDA |
|---------------------------------|:------------:|:----------------------:|:-----------:|
| `lewm-train` (CPU / CUDA)       | yes          | yes (CPU)              | yes (≥ 11)  |
| `lewm-infer` (Tract ONNX/NNEF)  | yes          | yes                    | n/a         |
| `lewm-infer` (Burn NdArray)     | yes          | yes                    | n/a         |
| `lewm-gpu` (Burn CUDA)          | n/a          | n/a                    | yes         |

Python helpers for ONNX export, parity-dump generation, and Hub upload live under [`python/`](python/) and are independent of the Rust runtime; see [`python/pyproject.toml`](python/pyproject.toml) (Ruff-linted, no Poetry/uv requirement, `make py-lint` works without them).

## Quick start

Three steps from clone to a verifiable end-to-end result. All run on a CPU laptop in under five minutes.

```bash
# 1. Build the workspace (release; ~3-5 min cold, hdf5+tract are the long poles).
cargo build --release --workspace --locked

# 2. Run lewm-core unit + shape tests (the parity contract surface, no fixtures needed).
cargo test -p lewm-core --release --locked
#    test result: ok. <N> passed; 0 failed
```

```bash
# 3. Run a real training smoke on CPU: deterministic 50-step loop, writes
#    checkpoint, sidecar, .mpk, .safetensors, and parity JSON to /tmp/lewm-smoke.
cargo run --release -p lewm-train -- \
  --config configs/pusht.toml --device cpu \
  --output-dir /tmp/lewm-smoke smoke --steps 50 --batch-size 4

ls /tmp/lewm-smoke/
#    step_0000050.json   step_0000050.mpk   step_0000050.parity.json
#    step_0000050.safetensors   train_losses.jsonl   train_report.json
```

The smoke path exercises the full data-plane (config load → deterministic init → AdamW step → checkpoint export) against a PushT-shaped fixture, so it dodges the most common first-user failures (missing dataset, missing CUDA, missing `HDF5_PLUGIN_PATH`). To reproduce the 4.08 s/episode CPU planning headline, see the [Benchmarks](#benchmarks) section.

## Method

`Jepa<B>` is the locked LeWM topology (RFC 0002): a 192-d **ViT-Tiny encoder** (12 layers, 3 heads, patch 14, 224×224 input), a 6-layer **autoregressive predictor** with AdaLN-zero conditioning (16 heads, dim_head 64), a 2-layer **action embedder**, and matching **projector / pred-proj** MLPs — **18,042,672 parameters across 303 tensors**, identical to the upstream reference state-dict.

Training minimises `total = pred_loss + λ · SIGReg`. **SIGReg** is a sketch-based singular-value regulariser on the projected encoder outputs that prevents latent collapse without negative pairs (RFC 0003). Inference splits the trained module into `encoder.onnx` and `predictor.onnx`; CEM planning samples `n_cand` action sequences, rolls them through the predictor, scores them by latent distance to a goal embedding, and refits a Gaussian to the elite set for `n_iter` rounds (RFC 0007).

The architecture is fixed by the upstream paper ([arXiv:2502.16560](https://arxiv.org/abs/2502.16560)); the contribution here is the Rust stack, the parity contract, and the SO-100 extension. The math is not re-derived in this README — see [`docs/`](docs/) (mdBook; build with `make docsite`) and [`paper/lewm-rs.md`](paper/lewm-rs.md).

## Public API

Eight crates, layered with no cycles by `scripts/check_layers.py`. `lewm-core` carries no internal deps; `lewm-infer` carries no CUDA or autodiff deps; only `lewm-gpu` may depend on `burn-cuda`.

| Crate            | Role                                                | Binaries                                |
|------------------|-----------------------------------------------------|-----------------------------------------|
| `lewm-core`      | ViT, predictor, AdaLN, SIGReg, init, import/export  | —                                       |
| `lewm-data`      | PushT HDF5 + LeRobot v2.1 loaders                   | —                                       |
| `lewm-train`     | Trainer, optimizer state, checkpointing             | `lewm-train`, `lewm-reference-record`   |
| `lewm-plan`      | CEM planner + planning evaluation                   | `lewm-eval`                             |
| `lewm-infer`     | Tract + Burn-NdArray runners, planning CLI          | `lewm-infer` (plan/bench/serve/verify/eval) |
| `lewm-gpu`       | Burn-CUDA glue (only crate allowed to depend on it) | —                                       |
| `lewm-telemetry` | OTel + nvml emitters                                | —                                       |
| `lewm-hub`       | Hugging Face Hub helpers, cost ledger               | —                                       |

Constructing and running the model from Rust:

```rust
use burn_ndarray::{NdArray, NdArrayDevice};
use lewm_core::{Jepa, JepaConfig};

let device = NdArrayDevice::Cpu;
let config = JepaConfig::default();              // RFC 0002 ViT-Tiny, 192-d latent
let model: Jepa<NdArray> = Jepa::init(config, &device)?;
let z = model.encode(pixels)?;                   // (B, T, 192)
```

A backend-generic runner trait drives planning and inference; the same trait object is returned for Tract, Burn-NdArray, and (via `lewm-gpu`) Burn-CUDA:

```rust
use lewm_infer::runner::{load_with_backend, BackendKind, InferenceRunner};

let mut runner: Box<dyn InferenceRunner> = load_with_backend(
    BackendKind::TractOnnx,
    &checkpoint_dir,
    safetensors_path.as_deref(),
)?;
let latent = runner.encode(&pixels)?;
let next   = runner.predict(&history, &actions, h, a)?;
```

Full rustdoc: `make docs` (warnings denied). Pedagogical docsite (Concepts → Architecture → Training → Inference → Reference): `make docsite`.

## Benchmarks

All numbers below come from runs in this repository; every row points to a reproducible command or report.

| Metric                                  | Value                   | Hardware                | Reproduce                                                                                  |
|-----------------------------------------|-------------------------|-------------------------|--------------------------------------------------------------------------------------------|
| Parameter count                         | 18,042,672 (303 tensors) | —                      | `python python/param_name_map.py`; [`docs/src/architecture/parameter-inventory.md`](docs/src/architecture/parameter-inventory.md) |
| Reference parity (encoder, projector, pred-proj, sigreg) | L∞ < 1e-4, 10 / 10 tests pass | CPU (Burn NdArray) | `cargo test -p lewm-core --features parity-fixtures parity_ -- --nocapture` |
| PushT loss reduction (50,000 steps, bs 64, bf16) | 0.4912 → 3.17e-06 (≈ 155,000× drop) | A10G-large (HF Jobs) | [`reports/pusht_training.md`](reports/pusht_training.md) |
| PushT wall time                         | 318 min                  | A10G-large              | same                                                                                       |
| SO-100 loss reduction (5,000 steps, bs 64) | 0.5002 → 9.56e-05 (≈ 5,240× drop) | A10G-large              | [`reports/so100_training.md`](reports/so100_training.md)                                   |
| SO-100 wall time                        | 864 s                    | A10G-large              | same                                                                                       |
| Tract CPU planning latency              | p50 **4.08 s/episode**, p95 4.13 s | Apple M3 (8-core ARM, release) | command below; [`reports/inference.md`](reports/inference.md)                  |
| End-to-end cloud spend                  | **$11.70**               | A10G-large @ $1.50/hr   | [`reports/cost.md`](reports/cost.md)                                                       |

Reproducing the 4.08 s/episode headline (Apple M3 release build, 5 CEM iter × 1024 cand, H=3 history steps, action dim 10):

```bash
hf download abdelstark/lewm-rs-pusht --include 'tract-compat/*' --local-dir /tmp/lewm-pusht
cargo build --release -p lewm-infer
./target/release/lewm-infer bench \
  --checkpoint-dir /tmp/lewm-pusht/tract-compat \
  --action-dim 10 --episodes 10
```

### Scope of the claims

CEM planning success rate on the standard PushT eval is **not** measured in this repository; the verified claims are the loss curves, parity tolerances, and CPU planning latency listed above. Tract is CPU-only by design — there is no Tract GPU backend; GPU inference goes through `lewm-gpu`'s `burn-cuda` runner (compile-tested in CI, runtime opt-in). Baselines are not tabulated because the published Le-WM PushT result lives in the [upstream repository](https://github.com/lucas-maes/le-wm) under a different runtime; the parity test suite is the apples-to-apples comparison.

## Reproducibility

- **Toolchain.** Rust 1.95.0 pinned in [`rust-toolchain.toml`](rust-toolchain.toml); `Cargo.lock` committed and CI enforces `--locked`.
- **Determinism.** Substream-keyed `ChaCha20Rng` for all sampling (RFC 0013); `thread_rng` is lint-banned by `scripts/check_nondet.py`. Model-init seed, dataloader seed, and planning seed are recorded in every checkpoint sidecar.
- **Hardware used for published results.** A10G-large (HF Jobs, $1.50/hr) for both training runs; Apple M3 (8-core ARM, `cargo build --release`) for the Tract CPU benchmark.
- **Datasets.** PushT mirror at `quentinll/lewm-pusht` (HDF5 + Blosc); SO-100 mirror at `abdelstark/so100-pickplace-lewm-ready` (HDF5 re-encode of `lerobot/svla_so100_pickplace`).
- **Quality gate.** `CARGO_INCREMENTAL=0 make check` runs fmt + clippy `-D warnings` + Ruff + `cargo check` + spec / layer / non-determinism validators + release blocker / Phase A handoff validators + `cargo deny` + `cargo audit`.

Approval-gated full PushT production run (HF account required; cost must be approved before launch):

```bash
python3 scripts/verify_runtime_image.py \
  --image-tag REPLACE_WITH_RUNTIME_IMAGE_TAG

scripts/launch_hf_job.py jobs/train_pusht.yaml \
  --allow-approval-required \
  --image-tag REPLACE_WITH_RUNTIME_IMAGE_TAG
```

Replace `REPLACE_WITH_RUNTIME_IMAGE_TAG` with a concrete published GHCR image
tag that contains the current full Burn/Jepa training and export-gate code.
The verifier checks the tag's OCI source and revision labels before any paid
HF Job is submitted.

After that job publishes `train/pusht-full-burn-jepa-YYYYMMDDTHHMMSSZ/`, dry-run
the F1 ONNX handoff before executing or uploading:

```bash
scripts/f1_export_pusht_onnx.py \
  --run-prefix train/pusht-full-burn-jepa-REPLACE_WITH_UTC_TIMESTAMP
```

Replace `REPLACE_WITH_UTC_TIMESTAMP` with the actual `YYYYMMDDTHHMMSSZ` suffix
published by the approved PushT job.

## Project structure

```text
crates/      Eight-crate Rust workspace (layering enforced by scripts/check_layers.py)
configs/     Locked TOML configs — pusht.toml, so100.toml, eval/warmstart variants
specs/       Accepted RFCs 0001-0018 + ADRs + traceability matrix (source of truth)
docs/        mdBook docsite — Concepts > Architecture > Training > Inference > Reference
paper/       Long-form writeup (CC-BY-4.0 intended)
reports/     pusht_training, so100_training, inference, gpu_inference, cost ledger
python/      Edge helpers: export_onnx, convert_reference, eval_compare, Hub upload
jobs/        Hugging Face Jobs YAML — smoke / short / bounded train / eval, cost-bounded
scripts/     Local validators — check_specs / check_layers / check_jobs / check_nondet
```

## Citation

```bibtex
@software{lewm_rs_2026,
  title  = {lewm-rs: A pure-Rust reproduction of LeWorldModel},
  author = {Abdel},
  year   = {2026},
  url    = {https://github.com/AbdelStark/lewm-rs}
}
```

Upstream paper:

```bibtex
@article{maes2026lewm,
  title         = {LeWorldModel: Stable JEPA world models from pixels},
  author        = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien
                   and Balestriero, Randall and LeCun, Yann},
  year          = {2026},
  eprint        = {2502.16560},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

Code is MIT-licensed ([`LICENSE`](LICENSE)). Trained checkpoints published on the Hub are intended to be Apache-2.0; the long-form writeup under [`paper/`](paper/) is intended to be CC-BY-4.0.

## Acknowledgments

Built on [LeWorldModel](https://github.com/lucas-maes/le-wm) by Maes, Le Lidec, Scieur, Balestriero, and LeCun (the upstream reference implementation); the [Burn](https://github.com/tracel-ai/burn) deep-learning framework (Tracel AI); [Tract](https://github.com/sonos/tract) for CPU inference (Sonos); the [LeRobot](https://github.com/huggingface/lerobot) ecosystem for the SO-100 source dataset; and the Hugging Face Hub for artifact distribution and compute.

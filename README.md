# lewm-rs

> Pure-Rust reproduction and extension of LeWorldModel (Maes et al., 2026).

[![Spec checks](https://github.com/AbdelStark/lewm-rs/actions/workflows/specs.yml/badge.svg)](https://github.com/AbdelStark/lewm-rs/actions/workflows/specs.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What

`lewm-rs` is a Rust workspace for reproducing LeWorldModel training, planning,
CPU inference, and artifact publication. The Burn-backed parity stack is
numerically verified against the locked PushT reference checkpoint (all 10
activation-level parity tests pass, L∞ < 1e-4). PushT 50k-step training
completed (loss 0.491 → 3.17e-06, 318 min, A10G-large). SO-100 training
completed (5000 steps, 864s, loss 0.50 → 9.56e-05). ONNX export and Tract
CPU inference are working end-to-end (4.08 s/episode, p50, release build).

The binding product and engineering contract lives in [`PRD.md`](PRD.md) and
[`specs/`](specs/). The current execution backlog is
[`ROADMAP.md`](ROADMAP.md) and
[#189](https://github.com/AbdelStark/lewm-rs/issues/189). Model artifacts
land at
[abdelstark/lewm-rs-pusht](https://huggingface.co/abdelstark/lewm-rs-pusht)
and
[abdelstark/lewm-rs-so100](https://huggingface.co/abdelstark/lewm-rs-so100).
The demo Space is live at
[abdelstark/lewm-rs-demo](https://huggingface.co/spaces/abdelstark/lewm-rs-demo).

## Quickstart

```sh
git clone https://github.com/AbdelStark/lewm-rs.git
cd lewm-rs
rustup show active-toolchain
cargo check --workspace --locked
python3 scripts/check_specs.py && python3 scripts/check_layers.py
```

Make targets mirror the local gates:

| Target | Command |
|--------|---------|
| `make fmt` | Format the Rust workspace. |
| `make lint` | Run clippy with warnings denied. |
| `make py-lint` | Lint Python helpers with Ruff (config in `python/pyproject.toml`); falls back to `py_compile` when Ruff is not installed. |
| `make test` | Run workspace tests with all features. |
| `make test-fast` | Run lib/bin tests excluding `_slow_` tests. |
| `make bench` | Run workspace benchmarks. |
| `make docs` | Build rustdoc with warnings denied. |
| `make check` | Run format, Rust lint, Python lint, cargo check, spec/layer checks, deny, and audit. |
| `make accept` | Run the current release gate: check, test, docs, `python/Makefile` Python gate, and future hub-artifact / release-inventory hooks. |
| `make clean` | Remove Cargo build outputs. |

## Results

| Result | Current state | Target |
|--------|---------------|--------|
| Parity verification | **Verified** — all 10 activation-level tests pass (L∞ < 1e-4) | Numerical match to reference |
| PushT full training | **Completed** — 50k steps, 318 min, A10G-large; loss 0.491→3.17e-06; artifacts at [`abdelstark/lewm-rs-pusht`](https://huggingface.co/abdelstark/lewm-rs-pusht) | CEM success rate ≥ 87% (eval pending) |
| SO-100 pick-and-place | **Completed** — 5000 steps, 864s, A10G-large; loss 0.50→9.56e-05; artifacts at [`abdelstark/lewm-rs-so100`](https://huggingface.co/abdelstark/lewm-rs-so100) | Warm-start ablation (pending) |
| CPU inference (Tract) | **Benchmarked** — 4.08s/episode (p50, release build, M-series Mac, 5 CEM iter × 1024 cand) | Sub-second on GPU / batched |
| CPU inference (Burn `NdArray`) | **Implemented** — `lewm-infer --backend burn-cpu` runs the in-Rust `Jepa<B>` module directly from Safetensors weights (CLS-projected encoder, 192-dim latent) | Latency baseline vs. Tract |
| GPU inference (Burn CUDA) | **Implemented** — `lewm-infer --backend burn-cuda` (feature `burn-cuda`) runs the same `Jepa<B>` module on NVIDIA GPUs; built and CI-checked | Measured A10G/CUDA latency |
| Parity eval CLI | **Implemented** — `lewm-infer eval --dumps-dir ...` compares any runner against the official reference dumps (per-stage L∞/RMSE JSON); see [`reports/gpu_inference.md`](reports/gpu_inference.md) | Continuous parity gate |
| ONNX export | **Done** — encoder + predictor for onnxruntime (opset 18) and Tract (opset 17); at [`abdelstark/lewm-rs-pusht`](https://huggingface.co/abdelstark/lewm-rs-pusht) | Stable export pipeline |
| Hub publication | Model cards + checkpoints + ONNX on Hub; demo at [`abdelstark/lewm-rs-demo`](https://huggingface.co/spaces/abdelstark/lewm-rs-demo) | Model, dataset, Space |

## Reports and paper

| Document | Link |
|----------|------|
| Paper writeup | [`paper/lewm-rs.md`](paper/lewm-rs.md) |
| PushT training report | [`reports/pusht_training.md`](reports/pusht_training.md) |
| GPU inference & parity eval | [`reports/gpu_inference.md`](reports/gpu_inference.md) |
| SO-100 training report | [`reports/so100_training.md`](reports/so100_training.md) |
| Inference + export report | [`reports/inference.md`](reports/inference.md) |
| Cost ledger | [`reports/cost.md`](reports/cost.md) — confirmed $11.70 total |
| Release checklist | [`reports/release_checklist.md`](reports/release_checklist.md) |

## Architecture at a glance

```text
dataset mirrors
    |
    v
lewm-data -> lewm-train -> checkpoints + telemetry + Hub upload
                    |
                    v
             lewm-plan -> planning metrics
                    |
                    v
             lewm-infer -> Tract CPU runner -> demo Space
```

## Optional telemetry

Real training runs can export OTLP traces to the self-hosted local stack in
[`infra/otel`](infra/otel/README.md). CI and smoke runs leave
`OTEL_EXPORTER_OTLP_ENDPOINT` unset, so the OTLP exporter is disabled and
training does not depend on telemetry infrastructure. Use
`python3 scripts/otel_smoke.py` for an opt-in local collector smoke check.

## Training image

HF Jobs use `ghcr.io/abdelstark/lewm-rs:latest`, built from the checked-in
[`Dockerfile`](Dockerfile). The image contains `lewm-train`, the checked-in
configs, HF job specs, Python helpers, `hf`, `zstd`, and `bash`.

## Smoke training

The current smoke path validates the runnable training envelope: config load,
deterministic scalar training mechanics, checkpoint artifacts, Hub upload, and
optional telemetry wiring. It is not the full JEPA training loop yet.

```sh
cargo run -p lewm-train -- --config configs/pusht.toml --device cpu --output-dir /tmp/lewm-smoke smoke --steps 50 --batch-size 4
HF_TOKEN=dummy python3 python/upload_checkpoints.py --src /tmp/lewm-smoke --dst abdelstark/lewm-rs-pusht --path-prefix smoke/local --dry-run
scripts/launch_hf_job.py jobs/smoke_pusht.yaml
```

## Short PushT train

The bounded `train --max-steps` path is a real PushT data-plane train run for
`pusht-full-module-lewm`: a deterministic config-shaped host `LeWM` path with
encoder, projector, action encoder, predictor, and prediction-projection
components at the locked PushT dimensions, plus AdamW update, scheduler,
gradient clipping, JSONL losses, checkpoint sidecar, `.mpk`, `.safetensors`,
and parity JSON. It uses HDF5 PushT windows when a dataset path is provided, and
otherwise writes an explicitly marked PushT-compatible fixture run for local
plumbing checks. It is not the final Burn ViT parity stack and does not make
PushT success-rate claims. `--resume-if-present` restores the latest complete
checkpoint for this mode and validates the sidecar, `.mpk`, `.safetensors`,
config hash, seed, step, AdamW state, and RNG state before continuing.

The public `quentinll/lewm-pusht` HDF5 stores pixels with the Blosc HDF5 filter;
set `HDF5_PLUGIN_PATH` from the Python `hdf5plugin` package before reading that
file outside the container.

```sh
cargo run -p lewm-train -- --config configs/pusht.toml --device cpu --output-dir /tmp/lewm-train-pusht --max-steps 10 train
scripts/launch_hf_job.py jobs/short_pusht.yaml
```

## Reproducing

- Clone the repo and use the pinned Rust toolchain in `rust-toolchain.toml`.
- Run the local quality gate: `CARGO_INCREMENTAL=0 make check`.
- Run the focused train crate gate when changing training:
  `cargo test -p lewm-train --all-features --locked`.
- Follow the training runbook in
  [RFC 0005](specs/rfcs/0005-training-system.md#9-runbook) once the data,
  training, and job milestones are implemented.

## Project structure

```text
crates/     Rust workspace crates for core, data, training, planning, inference, telemetry, Hub
infra/      Optional self-hosted observability infrastructure
scripts/    Local validation and repository maintenance scripts
specs/      Accepted RFCs, ADR process, glossary, and traceability matrix
python/     Edge adapters for conversion, decoding, stats, cost, and upload
jobs/       Hugging Face Jobs launch files
reports/    Cost ledger and future training, parity, and inference reports
paper/      Planned paper-style writeup and figures
```

## License

Code is MIT licensed. Trained checkpoints are intended to be Apache-2.0.
The paper-style writeup is intended to be CC-BY-4.0.

## Citation

```bibtex
@software{lewm_rs_2026,
  title = {lewm-rs: Rust reproduction and extension of LeWorldModel},
  author = {Abdel},
  year = {2026},
  url = {https://github.com/AbdelStark/lewm-rs}
}
```

## Acknowledgments

This project builds on LeWorldModel by Maes, Le Lidec, Scieur, Balestriero,
and LeCun, the upstream reference code by Lucas Maes, and the Burn framework.

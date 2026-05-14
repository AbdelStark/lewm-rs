# lewm-rs

> Pure-Rust reproduction and extension of LeWorldModel (Maes et al., 2026).

[![Spec checks](https://github.com/AbdelStark/lewm-rs/actions/workflows/specs.yml/badge.svg)](https://github.com/AbdelStark/lewm-rs/actions/workflows/specs.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What

`lewm-rs` is a Rust workspace for reproducing LeWorldModel training, planning,
CPU inference, and artifact publication. The repository is currently in the
bootstrap phase: the spec set is accepted, the workspace skeleton exists, and
the implementation is landing issue by issue against the RFC contracts.

The binding product and engineering contract lives in [`PRD.md`](PRD.md) and
[`specs/`](specs/). The latest planned model artifact is
[abdelstark/lewm-rs-pusht](https://huggingface.co/abdelstark/lewm-rs-pusht),
and the planned demo Space is
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
| `make test` | Run workspace tests with all features. |
| `make test-fast` | Run lib/bin tests excluding `_slow_` tests. |
| `make bench` | Run workspace benchmarks. |
| `make docs` | Build rustdoc with warnings denied. |
| `make check` | Run format, lint, cargo check, spec/layer checks, deny, and audit. |
| `make accept` | Run the current release gate: check, test, docs, and available future hooks. |
| `make clean` | Remove Cargo build outputs. |

## Results

| Result | Current state | Target |
|--------|---------------|--------|
| PushT planning success | Not yet trained in this repo | >= 87% |
| SO-100 pick-and-place extension | Specified, not yet trained | Warm-start ablation report |
| CPU inference | Workspace scaffolded | Sub-second Tract cost computation |
| Hub publication | Repos named in spec | Model, dataset, and Space artifacts |

Final metrics will link to model cards and reports once the training and
evaluation milestones land.

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
training does not depend on telemetry infrastructure.

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

## Reproducing

- Clone the repo and use the pinned Rust toolchain in `rust-toolchain.toml`.
- Run the local quality gates as they land; today that is `cargo check`,
  `scripts/check_layers.py`, and `scripts/check_specs.py`.
- Follow the training runbook in
  [RFC 0005](specs/rfcs/0005-training-system.md#9-runbook) once the data,
  training, and job milestones are implemented.

## Project structure

```text
crates/     Rust workspace crates for core, data, training, planning, inference, telemetry, Hub
infra/      Optional self-hosted observability infrastructure
scripts/    Local validation and repository maintenance scripts
specs/      Accepted RFCs, ADR process, glossary, and traceability matrix
python/     Planned edge adapters for conversion, decoding, plotting, and upload
jobs/       Planned Hugging Face Jobs launch files
reports/    Planned training, parity, inference, and cost reports
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

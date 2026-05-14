# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- RFC 0013 RNG sub-stream state serialization and a nondeterminism lint for
  Rust sources.
- RFC 0009 system metric samplers for CPU utilization, process RSS, disk usage,
  and optional NVML GPU utilization/memory telemetry.
- Initial Rust workspace, spec validation, quality gate configuration, and OSS
  scaffolding.
- Optional `lewm-telemetry/nvtx` profiling layer and local profiling artifact
  workflow for RFC 0014.
- Cost-ledger append, integrity-check, pricing, backfill, and CI cap gates.
- Hugging Face model-card renderer with YAML frontmatter, provenance, and
  attribution blocks.
- Hugging Face Hub client upload pipeline with SHA-256 idempotency and retry
  policy.
- PushT evaluation driver plumbing, pinned eval config, `lewm-eval pusht`
  CLI, and JSON/Markdown/Parquet eval artifacts.
- PushT JSON-RPC sidecar with a deterministic mock backend and pinned
  `gym-pusht` simulator extra for real eval runs.
- RFC 0006 CEM planner with deterministic `rng:cem` sampling, chunked cost
  evaluation fallback, and toy-quadratic convergence tests.
- Resume-aware PushT full-training HF Jobs spec and the ml-intern approval
  leash entry for human-gated full training launches.
- PushT smoke and short HF Jobs specs with local schema checks for hardware,
  timeout, environment passthrough, and checkpoint upload steps.
- Root TOML config loader with layered environment/CLI overrides, validation,
  canonical BLAKE3 hashing, and the `configs/pusht.toml` fixture.
- `lewm-train` clap CLI contract for train, smoke, parity, eval, and convert
  commands, including config overrides and provenance preamble formatting.
- Added `lewm-train` trainer state-machine, transition, parity-probe, smoke
  slope, and eval-cadence primitives.
- Added `lewm-train` resume detection, RNG restoration, and shutdown-signal
  checkpoint handler primitives.
- Added `lewm-train` checkpoint sidecar, atomic write, safetensors mirror, and
  pruning primitives.
- Added `lewm-train` inner-step accumulation, clipping, and non-finite guard
  primitives.
- Added `lewm-train` mixed-precision policy contracts.
- Added `lewm-train` cosine warmup learning-rate schedule.
- Added `lewm-train` AdamW RFC defaults and decay/no-decay parameter
  partitioning.
- Direct `lewm-core` Burn `0.20.1` dependency and compile smoke for the
  Rust `1.89.0` parity implementation path.
- Burn-backed `lewm-core::vit` encoder modules with RFC 0002 shape coverage.
- Burn-backed `lewm-core::embedder` action encoder with preserved Conv1d-k1
  smoothing and shape coverage.
- Burn-backed `lewm-core::mlp` projector heads with feature-axis normalization
  and RFC 0002 shape coverage.
- Burn-backed `lewm-core::ada_ln::AdaLNZero` with zero-initialized
  modulation heads and RFC 0002 shape coverage.
- Burn-backed `lewm-core::predictor::{ConditionalBlock, ArPredictor}` with
  AdaLN-zero conditioning, causal attention, and RFC 0002 shape coverage.
- Burn-backed `lewm-core::losses::SigReg` with RFC 0003 constants,
  deterministic sketch sampling, fixed-projection parity entry point, and
  gradient/RNG coverage.
- Burn tensor `lewm-core::losses::prediction_loss` with shape, value, and
  bidirectional-gradient coverage.
- Burn-backed `lewm-core::Jepa` top-level wrapper with encode, predict,
  rollout, criterion, and per-candidate cost contracts.
- RFC 0008 `lewm-core` parity initialization shape audit for the top-level
  JEPA wrapper.
- Python reference-checkpoint parameter-name map for the locked PushT source
  state dict and Burn record conversion preflight.
- `python/convert_reference.py audit` for validating the downloaded PushT
  reference checkpoint keys against the locked conversion map.
- `python/convert_reference.py convert`, `python/verify_conversion.py`, and
  `lewm-reference-record` for emitting a deterministic Safetensors mirror,
  load-checked Burn `NamedMpk` record, and Safetensors-vs-record drift check
  from the locked PushT reference checkpoint.
- `lewm_core::export` Safetensors export helpers for deterministic `Jepa`
  parameter mirrors, including BatchNorm running state and integer counters.

### Changed

- Expanded GitHub issue templates, the pull request traceability checklist, and
  CODEOWNERS mappings for crate-level review routing.
- Bumped the pinned Rust toolchain, CI toolchain, and training image builder to
  Rust `1.89.0` to satisfy the Burn `0.20.1` MSRV.
- Aligned the Burn predictor attention and feed-forward submodules with the
  upstream PushT checkpoint layout, including predictor qkv bias removal and
  affine pre-norm parameters.

### Deprecated

### Removed

### Fixed

### Security

- Added a gitleaks-backed secret scan wrapper and CI gate for
  `TST-0016-SECRETS-001`.
- Documented ADR 0002's date-bounded `cargo audit` waiver for Burn's transitive
  `RUSTSEC-2025-0141` `bincode` dependency.

## [0.1.0] - TBD

Initial public release target.

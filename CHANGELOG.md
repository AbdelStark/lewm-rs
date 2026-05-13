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

### Changed

### Deprecated

### Removed

### Fixed

### Security

- Added a gitleaks-backed secret scan wrapper and CI gate for
  `TST-0016-SECRETS-001`.

## [0.1.0] - TBD

Initial public release target.

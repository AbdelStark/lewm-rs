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

## [0.1.0] - TBD

Initial public release target.
